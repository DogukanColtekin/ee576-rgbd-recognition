// RGB-D recognition and tracking - parts (a) and (b)
// Segmentation -> RGB-D descriptors -> per-class binary SVM + MLP -> P/R at three tau
// values, 3-tau precision-recall curves, ROC, best/worst class.
//
// Usage:
//   ./train_eval --data <rgbd-dataset dir> --out <output dir>
//                [--seg plane|depth|gt] [--every 8] [--aug 1] [--bg 2]
//                [--hsv-weight 0] [--svm-c 0.01] [--mlp-iters 40]
//   ./train_eval --features <out dir>/features.yml.gz --out <dir>   (skip part (a))
//
//   --seg plane (default): OUR segmentation (support-plane removal, depth-Otsu as the
//               alternative candidate) feeds the descriptor for training AND test
//               images. The provided *_mask.png is read only to report segmentation IoU.
//   --seg depth: depth-window Otsu segmentation feeds the descriptor (ablation).
//   --seg gt   : provided mask feeds the descriptor (upper bound).
//   --hsv-weight: weight of the colour block after normalisation (0 = colour unused).
//   --onnx F --prop-neg K: run FastSAM on every training frame and add up to K of its
//               segments that do not touch the object as background negatives, so the
//               classifiers see the same kind of segments they will meet in the scene
//               (tray, turntable, table, cables...). Only training frames are used.
#include "common.hpp"
#include <opencv2/ml.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace rgbd;
namespace ml = cv::ml;

struct Models { cv::Ptr<ml::SVM> svm; cv::Ptr<ml::ANN_MLP> mlp; };

static Models trainBinary(const cv::Mat& Z, const std::vector<int>& pos01, double svmC, int mlpIters) {
    const int N = Z.rows;
    int np = 0;
    for (int v : pos01) np += v;
    const int nn = N - np;
    cv::Mat lab(N, 1, CV_32S);
    for (int i = 0; i < N; ++i) lab.at<int>(i) = pos01[i];

    // linear SVM, C=1, balanced class weights (= sklearn class_weight="balanced")
    Models m;
    m.svm = ml::SVM::create();
    m.svm->setType(ml::SVM::C_SVC);
    m.svm->setKernel(ml::SVM::LINEAR);
    m.svm->setC(svmC);
    cv::Mat w(2, 1, CV_64F);
    w.at<double>(0) = (double)N / (2.0 * std::max(1, nn));
    w.at<double>(1) = (double)N / (2.0 * std::max(1, np));
    m.svm->setClassWeights(w);
    m.svm->setTermCriteria({cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-6});
    m.svm->train(Z, ml::ROW_SAMPLE, lab);

    // MLP 719 -> 256 -> 64 -> 2, SGD back-propagation with momentum, balanced sample
    // weights; inputs are already normalised by FeatureNorm (NO_INPUT_SCALE keeps the
    // block weighting, OpenCV would otherwise re-standardise every input).
    m.mlp = ml::ANN_MLP::create();
    cv::Mat layers = (cv::Mat_<int>(4, 1) << DESC_DIM, 256, 64, 2);
    m.mlp->setLayerSizes(layers);
    m.mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
    m.mlp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001, 0.9);
    m.mlp->setTermCriteria({cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, mlpIters, 1e-6});
    cv::Mat targets(N, 2, CV_32F), sw(N, 1, CV_32F);
    for (int i = 0; i < N; ++i) {
        targets.at<float>(i, 0) = pos01[i] ? -1.f : 1.f;
        targets.at<float>(i, 1) = pos01[i] ?  1.f : -1.f;
        sw.at<float>(i) = pos01[i] ? (float)N / (2.f * std::max(1, np)) : (float)N / (2.f * std::max(1, nn));
    }
    m.mlp->train(ml::TrainData::create(Z, ml::ROW_SAMPLE, targets, cv::noArray(), cv::noArray(), sw),
                 ml::ANN_MLP::NO_INPUT_SCALE);
    return m;
}

int main(int argc, char** argv) {
    const std::string dataDir = argval(argc, argv, "--data", "rgbd-dataset");
    const std::string outDir  = argval(argc, argv, "--out", "out_train");
    std::string segMode       = argval(argc, argv, "--seg", "plane");
    if (segMode == "own") segMode = "plane";                             // old flag name
    const std::string featIn  = argval(argc, argv, "--features", "");
    const float HSV_W  = std::stof(argval(argc, argv, "--hsv-weight", "0"));
    const double SVM_C = std::stod(argval(argc, argv, "--svm-c", "0.01"));
    const int MLP_IT   = std::stoi(argval(argc, argv, "--mlp-iters", "40"));
    const std::string onnxPath = argval(argc, argv, "--onnx", "FastSAM-s.onnx");
    const int PROP_NEG = std::stoi(argval(argc, argv, "--prop-neg", "6"));
    const int EVERY_N      = std::stoi(argval(argc, argv, "--every", "8"));
    const int AUG_COPIES   = std::stoi(argval(argc, argv, "--aug", "1"));
    const int BG_PER_FRAME = std::stoi(argval(argc, argv, "--bg", "2"));
    const int NC = (int)CLASSES.size(), BG_LABEL = NC;
    if (segMode != "plane" && segMode != "depth" && segMode != "gt") {
        std::cerr << "--seg must be plane|depth|gt\n"; return 1;
    }
    std::filesystem::create_directories(outDir);
    std::cout << "data=" << (featIn.empty() ? dataDir : featIn) << " out=" << outDir << " seg=" << segMode
              << " hsv-weight=" << HSV_W << " svm-C=" << SVM_C << "\n";

    // ---------------- PART (a): segmentation + descriptors ----------------
    std::vector<std::vector<float>> Xtr, Xte;
    std::vector<int> ytr, yte, gtr;                          // gtr = training instance index (CV group)
    std::vector<std::string> teName;
    cv::RNG rng(0);
    cv::dnn::Net fastsam;
    if (featIn.empty() && PROP_NEG > 0) {
        if (std::filesystem::exists(onnxPath)) fastsam = cv::dnn::readNetFromONNX(onnxPath);
        else std::cerr << "WARNING: " << onnxPath << " not found - no proposal negatives\n";
    }
    int nPropNeg = 0;
    cv::Mat XtrM, XteM;
    if (!featIn.empty()) {                                   // reuse descriptors of an earlier run
        cv::FileStorage ff(featIn, cv::FileStorage::READ);
        if (!ff.isOpened()) { std::cerr << "cannot open " << featIn << "\n"; return 1; }
        ff["Xtr"] >> XtrM; ff["ytr"] >> ytr; ff["gtr"] >> gtr; ff["Xte"] >> XteM; ff["yte"] >> yte;
        if (!ff["teName"].empty()) ff["teName"] >> teName;
        teName.resize(yte.size(), "?");
        printf("loaded descriptors: train %d, test %d\n", XtrM.rows, XteM.rows);
    } else {
    std::ofstream segCsv(outDir + "/seg_iou.csv");
    segCsv << "class,frames,iou_plane,iou_depth,plane_fallback_rate\n";
    double allP = 0, allD = 0; int allN = 0;

    for (int ci = 0; ci < NC; ++ci) {
        const std::string& cls = CLASSES[ci];
        auto insts = listDirsNatural(dataDir + "/" + cls, cls + "_");
        if (insts.empty()) { std::cerr << "WARNING: no instances for " << cls << " under " << dataDir << "\n"; continue; }
        const int nTrain = std::min(3, std::max(1, (int)insts.size() - 1));   // 3 train, rest test
        int trN = 0, teN = 0, nf = 0, nFallback = 0;
        double sP = 0, sD = 0;
        for (int ii = 0; ii < (int)insts.size(); ++ii) {
            const bool isTrain = ii < nTrain;
            auto frames = listRgbFramesNatural(insts[ii]);
            for (size_t fi = 0; fi < frames.size(); fi += EVERY_N) {
                ObjectFrame fr;
                if (!readObjectFrame(frames[fi], fr)) continue;
                bool fb = false;
                cv::Mat segP = segmentObject(fr.depth, fr.loc, &fb);
                cv::Mat segD = segmentDepth(fr.depth, objectROI(fr.loc, fr.depth.size()));
                // ground truth used ONLY for this diagnostic (unless --seg gt)
                sP += maskIoU(segP, fr.maskGT); sD += maskIoU(segD, fr.maskGT); ++nf; nFallback += fb;

                const cv::Mat& seg = segMode == "gt" ? fr.maskGT : segMode == "depth" ? segD : segP;
                std::vector<float> d;
                if (!describe(fr.rgb, fr.depth, seg, d)) continue;
                if (!isTrain) {
                    Xte.push_back(d); yte.push_back(ci); ++teN;
                    teName.push_back(std::filesystem::path(frames[fi]).stem().string());
                    continue;
                }
                Xtr.push_back(d); ytr.push_back(ci); gtr.push_back(ii); ++trN;
                for (int a = 0; a < AUG_COPIES; ++a) {           // perturbed-mask copies
                    std::vector<float> da;
                    if (describe(fr.rgb, fr.depth, perturbMask(seg, rng), da)) {
                        Xtr.push_back(da); ytr.push_back(ci); gtr.push_back(ii); ++trN;
                    }
                }
                // background negatives: random boxes that do NOT touch the object.
                // (The first iteration accepted a box when mask-IoU <= 0.05, but a large box
                // that contains half of a small object still has IoU < 0.05, so
                // object pixels leaked into the "background" class.)
                cv::Mat keepOut;
                cv::dilate(seg, keepOut, cv::getStructuringElement(cv::MORPH_RECT, {31, 31}));
                const int Hh = fr.rgb.rows, Ww = fr.rgb.cols;
                for (int b = 0, tries = 0; b < BG_PER_FRAME && tries < 30; ++tries) {
                    int bw = rng.uniform(70, 160), bh = rng.uniform(70, 160);
                    if (Ww - bw <= 0 || Hh - bh <= 0) break;
                    cv::Rect r(rng.uniform(0, Ww - bw), rng.uniform(0, Hh - bh), bw, bh);
                    if (cv::countNonZero(keepOut(r)) > 0) continue;
                    cv::Mat bgm = cv::Mat::zeros(Hh, Ww, CV_8U);
                    bgm(r) = 255;
                    std::vector<float> dbg;
                    if (describe(fr.rgb, fr.depth, bgm, dbg)) { Xtr.push_back(dbg); ytr.push_back(BG_LABEL); gtr.push_back(ii); }
                    ++b;
                }
                // proposal negatives: FastSAM segments of this training frame that stay
                // clear of the object
                if (!fastsam.empty()) {
                    std::vector<cv::Mat> neg;
                    for (auto& pr : runFastSAM(fastsam, fr.rgb, 0.4f, 0.9f))
                        if (goodProposal(pr.mask, fr.depth) && cv::countNonZero(pr.mask & keepOut) == 0)
                            neg.push_back(pr.mask);
                    for (int i = (int)neg.size() - 1; i > 0; --i) std::swap(neg[i], neg[rng.uniform(0, i + 1)]);
                    for (int i = 0; i < (int)neg.size() && i < PROP_NEG; ++i) {
                        std::vector<float> dn;
                        if (describe(fr.rgb, fr.depth, neg[i], dn)) {
                            Xtr.push_back(dn); ytr.push_back(BG_LABEL); gtr.push_back(ii); ++nPropNeg;
                        }
                    }
                }
            }
        }
        double mP = nf ? sP / nf : 0, mD = nf ? sD / nf : 0;
        allP += sP; allD += sD; allN += nf;
        segCsv << cls << "," << nf << "," << mP << "," << mD << "," << (nf ? (double)nFallback / nf : 0) << "\n";
        printf("%-12s instances=%zu  train(+aug)=%d  test=%d  segIoU plane=%.3f depth=%.3f\n",
               cls.c_str(), insts.size(), trN, teN, mP, mD);
    }
    const int nBG = (int)std::count(ytr.begin(), ytr.end(), BG_LABEL);
    printf("\nbackground negatives: %d (of which FastSAM proposal negatives: %d)\nmean segmentation IoU vs provided mask: plane=%.3f  depth=%.3f  (%d frames)\n",
           nBG, nPropNeg, allN ? allP / allN : 0, allN ? allD / allN : 0, allN);
    segCsv << "ALL," << allN << "," << (allN ? allP / allN : 0) << "," << (allN ? allD / allN : 0) << ",\n";

    auto toMat = [](const std::vector<std::vector<float>>& V) {
        cv::Mat M((int)V.size(), DESC_DIM, CV_32F);
        for (int r = 0; r < M.rows; ++r) std::copy(V[r].begin(), V[r].end(), M.ptr<float>(r));
        return M;
    };
    XtrM = toMat(Xtr); XteM = toMat(Xte);
    printf("train %dx%d  test %dx%d\n", XtrM.rows, XtrM.cols, XteM.rows, XteM.cols);
    if (XtrM.rows == 0 || XteM.rows == 0) { std::cerr << "no data - check --data path\n"; return 1; }
    {   // raw descriptors: rerun part (b) later with --features
        cv::FileStorage ff(outDir + "/features.yml.gz", cv::FileStorage::WRITE);
        ff << "Xtr" << XtrM << "ytr" << ytr << "gtr" << gtr << "Xte" << XteM << "yte" << yte << "teName" << teName;
    }
    }   // end part (a)
    FeatureNorm norm;
    norm.fit(XtrM, HSV_W);
    cv::Mat Ztr = norm.transform(XtrM), Zte = norm.transform(XteM);

    // ---------------- PART (b): per-class binary SVM + MLP, cross-validated ensemble ----------------
    // fold f = training instance f of every class (and the background patches mined from
    // its frames). The models of fold f never see instance f; their Platt sigmoid is
    // fitted on it. Test probability = mean over folds.
    std::vector<int> folds(gtr.begin(), gtr.end());
    std::sort(folds.begin(), folds.end());
    folds.erase(std::unique(folds.begin(), folds.end()), folds.end());
    if (folds.size() < 2) { std::cerr << "need at least 2 training instances per class\n"; return 1; }
    ClassifierBank bank;
    bank.nClasses = NC; bank.nFolds = (int)folds.size();
    bank.svm.assign(NC, {}); bank.mlp.assign(NC, {});
    bank.calSvm.assign(NC, std::vector<Platt>(bank.nFolds)); bank.calMlp.assign(NC, std::vector<Platt>(bank.nFolds));
    std::ofstream cvCsv(outDir + "/cv_heldout_ap.csv");
    cvCsv << "class,fold,AP_SVM,AP_MLP\n";
    for (int ci = 0; ci < NC; ++ci) {
        double apS = 0, apM = 0;
        for (int fi = 0; fi < bank.nFolds; ++fi) {
            const int f = folds[fi];
            std::vector<int> tr, ho;
            for (int i = 0; i < Ztr.rows; ++i) (gtr[i] == f ? ho : tr).push_back(i);
            cv::Mat Zf((int)tr.size(), DESC_DIM, CV_32F);
            std::vector<int> pf(tr.size());
            for (size_t k = 0; k < tr.size(); ++k) { Ztr.row(tr[k]).copyTo(Zf.row((int)k)); pf[k] = ytr[tr[k]] == ci; }
            Models m = trainBinary(Zf, pf, SVM_C, MLP_IT);
            std::vector<float> mS, mM; std::vector<int> yh;
            for (int i : ho) { mS.push_back(svmMargin(m.svm, Ztr.row(i))); mM.push_back(mlpMargin(m.mlp, Ztr.row(i))); yh.push_back(ytr[i] == ci); }
            bank.calSvm[ci][fi].fit(mS, yh);
            bank.calMlp[ci][fi].fit(mM, yh);
            bank.svm[ci].push_back(m.svm); bank.mlp[ci].push_back(m.mlp);
            // AP of the calibrated held-out scores (OpenCV's raw SVM output has the
            // opposite sign for the positive class; Platt's A absorbs it)
            std::vector<float> pS, pM;
            for (size_t k = 0; k < yh.size(); ++k) { pS.push_back(bank.calSvm[ci][fi].prob(mS[k])); pM.push_back(bank.calMlp[ci][fi].prob(mM[k])); }
            double a1 = averagePrecision(pS, yh), a2 = averagePrecision(pM, yh);
            apS += a1; apM += a2;
            cvCsv << CLASSES[ci] << "," << f << "," << a1 << "," << a2 << "\n";
        }
        printf("trained %-12s %d folds   held-out-instance AP: SVM %.3f  MLP %.3f\n", CLASSES[ci].c_str(),
               bank.nFolds, apS / bank.nFolds, apM / bank.nFolds);
    }
    {
        cv::FileStorage fsModels(outDir + "/models.yml", cv::FileStorage::WRITE);
        norm.write(fsModels);
        bank.save(outDir, fsModels);
        fsModels << "seg_mode" << segMode << "hsv_weight" << HSV_W;
    }

    // ---------------- test-set evaluation ----------------
    auto scoreAll = [&](bool useSvm) {
        std::vector<std::vector<float>> S(NC, std::vector<float>(Zte.rows));
        for (int i = 0; i < Zte.rows; ++i)
            for (int ci = 0; ci < NC; ++ci) S[ci][i] = bank.prob(ci, Zte.row(i), useSvm);
        return S;
    };

    auto evaluate = [&](const std::vector<std::vector<float>>& S, const std::string& name) {
        std::ofstream tab(outDir + "/tau_table_" + name + ".csv");
        tab << "class,tau,precision,recall,tp,fp,fn\n";
        std::ofstream sum(outDir + "/summary_" + name + ".csv");
        sum << "class,n_test_pos,AP,ROC_AUC\n";
        printf("\n[%s] class, tau, precision, recall\n", name.c_str());
        std::vector<Series> prSeries, rocSeries;
        std::map<std::string, double> aps;
        std::vector<std::vector<double>> tauP(REPORT_TAUS.size(), std::vector<double>(NC)),
                                         tauR(REPORT_TAUS.size(), std::vector<double>(NC));
        for (int ci = 0; ci < NC; ++ci) {
            std::vector<int> yb(yte.size());
            for (size_t i = 0; i < yte.size(); ++i) yb[i] = yte[i] == ci;
            auto sweep = sweepClassification(S[ci], yb, denseTaus());
            auto marks = sweepClassification(S[ci], yb, REPORT_TAUS);
            double ap = averagePrecision(S[ci], yb), auc = rocAUC(S[ci], yb);
            aps[CLASSES[ci]] = ap;
            sum << CLASSES[ci] << "," << std::count(yb.begin(), yb.end(), 1) << "," << ap << "," << auc << "\n";
            Series pr, roc;
            char lbl[96];
            snprintf(lbl, 96, "%s  AP=%.2f  AUC=%.2f", CLASSES[ci].c_str(), ap, auc);
            pr.label = roc.label = lbl;
            roc.pts.push_back({1, 1});
            for (auto& p : sweep) { pr.pts.push_back({p.recall, p.precision}); roc.pts.push_back({p.fpr, p.recall}); }
            roc.pts.push_back({0, 0});
            for (size_t ti = 0; ti < marks.size(); ++ti) {
                const auto& p = marks[ti];
                tauP[ti][ci] = p.precision; tauR[ti][ci] = p.recall;
                pr.markers.push_back({p.recall, p.precision});
                char ml[96];
                if (p.tp + p.fp == 0) snprintf(ml, 96, "tau=%.1f  P=n/a   R=%.2f  (no detections)", p.tau, p.recall);
                else snprintf(ml, 96, "tau=%.1f  P=%.2f  R=%.2f", p.tau, p.precision, p.recall);
                pr.markerLabels.push_back(ml);
                roc.markers.push_back({p.fpr, p.recall});
                printf("  %-12s %.1f  P=%.3f  R=%.3f  (TP=%d FP=%d FN=%d)\n", CLASSES[ci].c_str(), p.tau,
                       p.precision, p.recall, p.tp, p.fp, p.fn);
                tab << CLASSES[ci] << "," << p.tau << "," << p.precision << "," << p.recall << ","
                    << p.tp << "," << p.fp << "," << p.fn << "\n";
            }
            prSeries.push_back(pr); rocSeries.push_back(roc);
        }
        plotTauPR(outDir + "/pr_tau_" + name + ".png",
                  name + " (object test set): recall vs precision at tau = 0.3 / 0.5 / 0.7", prSeries);
        plotXY(outDir + "/roc_" + name + ".png", name + ": ROC (tau markers 0.3/0.5/0.7)",
               "FPR", "TPR", rocSeries, true);
        for (size_t ti = 0; ti < REPORT_TAUS.size(); ++ti) {
            char fn[256], tt[128];
            snprintf(fn, 256, "%s/bars_tau%.1f_%s.png", outDir.c_str(), REPORT_TAUS[ti], name.c_str());
            snprintf(tt, 128, "%s: precision/recall per class @ tau=%.1f", name.c_str(), REPORT_TAUS[ti]);
            plotTauBars(fn, tt, CLASSES, tauP[ti], tauR[ti]);
        }
        // confusion matrix: true class vs argmax of the five binary scores
        // ("none" if no detector fires at tau=0.5) - explains which classes get confused
        std::ofstream cm(outDir + "/confusion_" + name + ".csv");
        cm << "true\\pred";
        for (auto& c : CLASSES) cm << "," << c;
        cm << ",none\n";
        std::vector<std::vector<int>> M(NC, std::vector<int>(NC + 1, 0));
        for (size_t i = 0; i < yte.size(); ++i) {
            int best = NC; float bs = 0.5f;
            for (int ci = 0; ci < NC; ++ci) if (S[ci][i] >= bs) { bs = S[ci][i]; best = ci; }
            M[yte[i]][best]++;
        }
        for (int r = 0; r < NC; ++r) {
            cm << CLASSES[r];
            for (int c = 0; c <= NC; ++c) cm << "," << M[r][c];
            cm << "\n";
        }
        std::ofstream sc(outDir + "/test_scores_" + name + ".csv");
        sc << "frame,true";
        for (auto& c : CLASSES) sc << "," << c;
        sc << "\n";
        for (size_t i = 0; i < yte.size(); ++i) {
            sc << teName[i] << "," << CLASSES[yte[i]];
            for (int ci = 0; ci < NC; ++ci) sc << "," << S[ci][i];
            sc << "\n";
        }
        auto best = std::max_element(aps.begin(), aps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        auto worst = std::min_element(aps.begin(), aps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        printf("[%s] best class: %s (AP=%.2f) | worst: %s (AP=%.2f)\n",
               name.c_str(), best->first.c_str(), best->second, worst->first.c_str(), worst->second);
    };

    evaluate(scoreAll(true),  "SVM");
    evaluate(scoreAll(false), "MLP");
    std::cout << "\nsaved models, tables and plots to " << outDir << "\n";
    return 0;
}
