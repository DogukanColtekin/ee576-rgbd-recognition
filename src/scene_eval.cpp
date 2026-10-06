// RGB-D recognition and tracking - part (c)
// FastSAM (ONNX, cv::dnn) proposals on the scene sequence -> RGB-D descriptors ->
// learned binary models of part (b) -> proposal IoU evaluation + recognition P/R at
// three tau values (VOC-style matching, IoU >= 0.5).
//
// Usage:
//   ./scene_eval --scene <kitchen_small_1 dir> --gt <gt.csv> --model <train_eval out dir>
//                --onnx FastSAM-s.onnx --out <output dir>
//                [--every 2] [--conf 0.4] [--nmsiou 0.9] [--dedup 0.85] [--evalnms 0.5]
//                [--reject-border] [--no-gate] [--no-argmax]
//
// gt.csv comes from the dataset's .mat file via scripts/convert_gt.py:
//   frame,category,x1,y1,x2,y2     (frame 1-based, pixel coords 0-based INCLUSIVE)
#include "common.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace rgbd;
namespace ml = cv::ml;

struct GTBox { std::string cat; cv::Rect2d box; };
static std::map<int, std::vector<GTBox>> loadGT(const std::string& csv) {
    std::map<int, std::vector<GTBox>> gt;
    std::ifstream f(csv);
    if (!f) { std::cerr << "cannot open " << csv << "\n"; return gt; }
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> t;
        while (std::getline(ss, tok, ',')) t.push_back(tok);
        if (t.size() < 6) continue;
        if (std::find(CLASSES.begin(), CLASSES.end(), t[1]) == CLASSES.end()) continue;
        double x1 = std::stod(t[2]), y1 = std::stod(t[3]), x2 = std::stod(t[4]), y2 = std::stod(t[5]);
        gt[std::stoi(t[0])].push_back({t[1], cv::Rect2d(x1, y1, x2 - x1 + 1, y2 - y1 + 1)});  // inclusive
    }
    return gt;
}

int main(int argc, char** argv) {
    const std::string sceneDir = argval(argc, argv, "--scene", "rgbd-scenes/kitchen_small/kitchen_small_1");
    const std::string gtCsv    = argval(argc, argv, "--gt", "gt.csv");
    const std::string modelDir = argval(argc, argv, "--model", "out_train");
    const std::string onnxPath = argval(argc, argv, "--onnx", "FastSAM-s.onnx");
    const std::string outDir   = argval(argc, argv, "--out", "out_scene");
    const int   EVERY_N  = std::stoi(argval(argc, argv, "--every", "2"));
    const float CONF     = std::stof(argval(argc, argv, "--conf", "0.4"));
    const float NMS_IOU  = std::stof(argval(argc, argv, "--nmsiou", "0.9"));
    const float DEDUP    = std::stof(argval(argc, argv, "--dedup", "0.85"));
    const float EVAL_NMS = std::stof(argval(argc, argv, "--evalnms", "0.5"));
    const bool  REJ_BORDER = hasFlag(argc, argv, "--reject-border");
    const bool  NO_GATE    = hasFlag(argc, argv, "--no-gate");
    const bool  NO_ARGMAX  = hasFlag(argc, argv, "--no-argmax");
    const int NC = (int)CLASSES.size();
    std::filesystem::create_directories(outDir);

    // ---------------- models of part (b) ----------------
    cv::FileStorage fs(modelDir + "/models.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) { std::cerr << "cannot open " << modelDir << "/models.yml\n"; return 1; }
    FeatureNorm norm; norm.read(fs);
    ClassifierBank bank;
    if (norm.w.empty() || !bank.load(modelDir, fs)) {
        std::cerr << "models in " << modelDir << " are from an older version - rerun train_eval\n"; return 1;
    }
    fs.release();

    auto gtAll = loadGT(gtCsv);
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxPath);

    // ---------------- frames ----------------
    std::vector<std::pair<int, std::string>> frames;
    for (auto& p : listRgbFramesNatural(sceneDir)) {
        int k = frameIndexFromPath(p);
        if (k >= 0 && k % EVERY_N == 0) frames.push_back({k, p});
    }
    std::sort(frames.begin(), frames.end());
    printf("processing %zu frames (every %d)\n", frames.size(), EVERY_N);
    if (frames.empty()) { std::cerr << "no frames found in " << sceneDir << "\n"; return 1; }

    // Only processed frames contribute ground truth. (The first iteration counted the GT
    // boxes of ALL frames while scoring only every 2nd one, so recall could never
    // exceed ~0.5.)
    std::map<int, std::vector<GTBox>> gt;
    for (auto& [k, p] : frames) if (gtAll.count(k)) gt[k] = gtAll[k];
    {
        std::ofstream ff(outDir + "/frames.txt");
        for (auto& [k, p] : frames) ff << k << "\n";
    }

    // ---------------- detection pass ----------------
    struct Det { int frame; cv::Rect2d box; float conf, svm[5], mlp[5]; SegmentGeometry geo; };
    std::vector<Det> dets;
    std::map<std::string, std::vector<double>> bestIoUCls;    // proposal quality per class
    size_t nRaw = 0, nGood = 0;
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const int k = frames[fi].first;
        cv::Mat rgb, depth;
        if (!readSceneFrame(frames[fi].second, rgb, depth)) continue;
        auto props = runFastSAM(net, rgb, CONF, NMS_IOU);
        const PlaneModel table = fitFramePlane(depth);
        nRaw += props.size();
        // geometric filter, then class-agnostic de-duplication of near-identical masks
        std::vector<Proposal> good;
        for (auto& pr : props)
            if (goodProposal(pr.mask, depth, REJ_BORDER)) { pr.box = maskBox(pr.mask); good.push_back(pr); }
        std::vector<cv::Rect2d> gb; std::vector<float> gc;
        for (auto& g : good) { gb.push_back(g.box); gc.push_back(g.conf); }
        auto keep = nmsGreedy(gb, gc, DEDUP);
        nGood += keep.size();
        std::vector<cv::Rect2d> frameBoxes;
        for (int i : keep) {
            std::vector<float> feat;
            if (!describe(rgb, depth, good[i].mask, feat)) continue;
            cv::Mat z = norm.transform(cv::Mat(1, DESC_DIM, CV_32F, feat.data()));
            Det d; d.frame = k; d.box = good[i].box; d.conf = good[i].conf;
            d.geo = segmentGeometry(good[i].mask, depth, &table);
            for (int ci = 0; ci < NC; ++ci) { d.svm[ci] = bank.prob(ci, z, true); d.mlp[ci] = bank.prob(ci, z, false); }
            dets.push_back(d);
            frameBoxes.push_back(d.box);
        }
        // IoU role 1: proposal-generation quality vs GT (class-agnostic)
        auto it = gt.find(k);
        if (it != gt.end())
            for (auto& g : it->second) {
                double best = 0;
                for (auto& b : frameBoxes) best = std::max(best, boxIoU(g.box, b));
                bestIoUCls[g.cat].push_back(best);
            }
        if (fi % 10 == 0) printf("  frame %d  proposals kept so far %zu\n", k, dets.size());
    }

    // ---------------- proposal quality ----------------
    std::ofstream pq(outDir + "/proposal_quality.csv");
    pq << "class,n_gt,mean_best_iou,recall_iou0.5\n";
    double sIoU = 0; int sN = 0, sHit = 0;
    std::map<std::string, double> propRecall;
    printf("\nproposals: %zu raw FastSAM masks -> %zu after filter+dedup (%.1f / frame)\n",
           nRaw, nGood, (double)nGood / frames.size());
    printf("%-12s %6s %10s %10s\n", "class", "n_gt", "meanIoU", "rec@0.5");
    for (auto& c : CLASSES) {
        auto& v = bestIoUCls[c];
        double m = 0; int hit = 0;
        for (double x : v) { m += x; hit += x >= 0.5; }
        sIoU += m; sN += (int)v.size(); sHit += hit;
        propRecall[c] = v.empty() ? 0 : (double)hit / v.size();
        printf("%-12s %6zu %10.3f %10.3f\n", c.c_str(), v.size(), v.empty() ? 0 : m / v.size(), propRecall[c]);
        pq << c << "," << v.size() << "," << (v.empty() ? 0 : m / v.size()) << "," << propRecall[c] << "\n";
    }
    printf("%-12s %6d %10.3f %10.3f\n", "ALL", sN, sN ? sIoU / sN : 0, sN ? (double)sHit / sN : 0);
    pq << "ALL," << sN << "," << (sN ? sIoU / sN : 0) << "," << (sN ? (double)sHit / sN : 0) << "\n";

    // ---------------- detections.csv (consumed by part d and visualize) ----------------
    {
        std::ofstream f(outDir + "/detections.csv");
        f << "frame,x1,y1,x2,y2,conf";                       // x2,y2 exclusive
        for (auto& c : CLASSES) f << ",svm_" << c;
        for (auto& c : CLASSES) f << ",mlp_" << c;
        f << ",e1,e2,e3,h_low,h_high\n";
        for (auto& d : dets) {
            f << d.frame << "," << d.box.x << "," << d.box.y << "," << d.box.x + d.box.width << ","
              << d.box.y + d.box.height << "," << d.conf;
            for (int c = 0; c < NC; ++c) f << "," << d.svm[c];
            for (int c = 0; c < NC; ++c) f << "," << d.mlp[c];
            f << "," << d.geo.e1 << "," << d.geo.e2 << "," << d.geo.e3 << "," << d.geo.hLow << "," << d.geo.hHigh << "\n";
        }
    }

    // ---------------- IoU role 2: recognition P/R per class at tau ----------------
    // Per class: detections scored by that class's binary model, per-frame NMS on that
    // score (a detector reports one box per object), then greedy matching in score
    // order to unmatched same-class GT with IoU >= 0.5.
    // Post-processing (defaults on, each can be switched off for the ablation table):
    //   gate    - GeometryGate: physically implausible segments are not detections
    //   argmax  - a segment is reported only for the class whose model scores it highest
    //   contain - per-frame NMS also suppresses a box lying >80% inside a better one
    struct Post { bool gate, argmax; double contain; };
    const Post FULL{!NO_GATE, !NO_ARGMAX, 0.8};
    GeometryGate gateCfg;
    auto evalDetections = [&](bool useSvm, const std::string& name, Post post, bool report) -> double {
        std::map<std::string, int> ngt;
        for (auto& [k, v] : gt) for (auto& g : v) ngt[g.cat]++;
        std::ofstream tab, sum;
        if (report) {
            tab.open(outDir + "/scene_tau_table_" + name + ".csv");
            tab << "class,tau,precision,recall,tp,fp,n_gt\n";
            sum.open(outDir + "/scene_summary_" + name + ".csv");
            sum << "class,n_gt,AP,proposal_recall_ceiling\n";
            printf("\n[%s] scene: class, tau, precision, recall\n", name.c_str());
        }
        std::vector<Series> prSeries;
        std::vector<std::vector<double>> tauP(REPORT_TAUS.size(), std::vector<double>(NC)),
                                         tauR(REPORT_TAUS.size(), std::vector<double>(NC));
        std::map<std::string, double> aps;

        for (int ci = 0; ci < NC; ++ci) {
            const std::string& cls = CLASSES[ci];
            // per-frame NMS on this class's score
            std::map<int, std::vector<int>> byFrame;
            for (int i = 0; i < (int)dets.size(); ++i) byFrame[dets[i].frame].push_back(i);
            std::vector<std::pair<float, int>> ranked;       // (score, det index)
            for (auto& [k, idx] : byFrame) {
                std::vector<cv::Rect2d> b; std::vector<float> s; std::vector<int> id;
                for (int i : idx) {
                    const float* sc = useSvm ? dets[i].svm : dets[i].mlp;
                    if (post.gate && !gateCfg.pass(dets[i].geo)) continue;
                    if (post.argmax && std::max_element(sc, sc + NC) - sc != ci) continue;
                    b.push_back(dets[i].box); s.push_back(sc[ci]); id.push_back(i);
                }
                for (int j : nmsGreedy(b, s, EVAL_NMS, post.contain)) ranked.push_back({s[j], id[j]});
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
            // TP/FP label of every ranked detection (greedy, score order)
            std::vector<int> lab(ranked.size(), 0);
            std::map<int, std::vector<char>> used;
            for (size_t r = 0; r < ranked.size(); ++r) {
                const Det& d = dets[ranked[r].second];
                auto it = gt.find(d.frame);
                if (it == gt.end()) continue;
                auto& fl = used[d.frame];
                fl.resize(it->second.size(), 0);
                int bj = -1; double bi = 0.5;
                for (size_t j = 0; j < it->second.size(); ++j) {
                    if (it->second[j].cat != cls || fl[j]) continue;
                    double iou = boxIoU(d.box, it->second[j].box);
                    if (iou >= bi) { bi = iou; bj = (int)j; }
                }
                if (bj >= 0) { lab[r] = 1; fl[bj] = 1; }
            }
            auto prAt = [&](float tau, int& tp, int& fp) {
                tp = fp = 0;
                for (size_t r = 0; r < ranked.size() && ranked[r].first >= tau; ++r) (lab[r] ? ++tp : ++fp);
            };
            const int G = ngt[cls];
            double ap = 0, ctp = 0, cfp = 0, prevR = 0;      // all-point AP over the ranking
            for (size_t r = 0; r < ranked.size(); ++r) {
                (lab[r] ? ++ctp : ++cfp);
                if (lab[r] && G) { double R = ctp / G; ap += (R - prevR) * ctp / (ctp + cfp); prevR = R; }
            }
            aps[cls] = ap;
            Series s;
            char lbl[96];
            snprintf(lbl, 96, "%s  AP=%.2f  (n_gt=%d, ceiling R=%.2f)", cls.c_str(), ap, G, propRecall[cls]);
            s.label = lbl;
            for (float tau : denseTaus()) {
                int tp, fp; prAt(tau, tp, fp);
                s.pts.push_back({G ? (float)tp / G : 0.f, (tp + fp) ? (float)tp / (tp + fp) : 1.f});
            }
            for (size_t ti = 0; ti < REPORT_TAUS.size(); ++ti) {
                int tp, fp; prAt(REPORT_TAUS[ti], tp, fp);
                float P = (tp + fp) ? (float)tp / (tp + fp) : 1.f, R = G ? (float)tp / G : 0.f;
                tauP[ti][ci] = P; tauR[ti][ci] = R;
                s.markers.push_back({R, P});
                char ml[96];
                if (tp + fp == 0) snprintf(ml, 96, "tau=%.1f  P=n/a   R=%.2f  (no detections)", REPORT_TAUS[ti], R);
                else snprintf(ml, 96, "tau=%.1f  P=%.2f  R=%.2f", REPORT_TAUS[ti], P, R);
                s.markerLabels.push_back(ml);
                if (report) printf("  %-12s %.1f  P=%.3f  R=%.3f  (TP=%d FP=%d nGT=%d)\n", cls.c_str(), REPORT_TAUS[ti], P, R, tp, fp, G);
                tab << cls << "," << REPORT_TAUS[ti] << "," << P << "," << R << "," << tp << "," << fp << "," << G << "\n";
            }
            sum << cls << "," << G << "," << ap << "," << propRecall[cls] << "\n";
            prSeries.push_back(s);
        }
        double mAP = 0;
        for (auto& [c, a] : aps) mAP += a;
        mAP /= std::max<size_t>(1, aps.size());
        if (!report) return mAP;
        plotTauPR(outDir + "/scene_pr_tau_" + name + ".png",
                  name + " (scene proposals): recall vs precision at tau = 0.3 / 0.5 / 0.7", prSeries);
        for (size_t ti = 0; ti < REPORT_TAUS.size(); ++ti) {
            char fn[256], tt[160];
            snprintf(fn, 256, "%s/scene_bars_tau%.1f_%s.png", outDir.c_str(), REPORT_TAUS[ti], name.c_str());
            snprintf(tt, 160, "%s scene: precision/recall per class @ tau=%.1f", name.c_str(), REPORT_TAUS[ti]);
            plotTauBars(fn, tt, CLASSES, tauP[ti], tauR[ti]);
        }
        auto best = std::max_element(aps.begin(), aps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        auto worst = std::min_element(aps.begin(), aps.end(), [](auto& a, auto& b) { return a.second < b.second; });
        printf("[%s] best: %s (AP=%.2f) | worst: %s (AP=%.2f) | mAP=%.3f\n", name.c_str(),
               best->first.c_str(), best->second, worst->first.c_str(), worst->second, mAP);
        return mAP;
    };
    evalDetections(true,  "SVM", FULL, true);
    evalDetections(false, "MLP", FULL, true);

    // ablation: contribution of each post-processing step (mAP over the five classes)
    {
        std::ofstream ab(outDir + "/scene_ablation.csv");
        ab << "setting,mAP_SVM,mAP_MLP\n";
        printf("\nablation (scene mAP)              SVM     MLP\n");
        const std::vector<std::pair<std::string, Post>> settings = {
            {"no post-processing", {false, false, 1.0}},
            {"+ geometry gate", {true, false, 1.0}},
            {"+ one class per segment", {true, true, 1.0}},
            {"+ containment NMS (full)", {true, true, 0.8}}};
        for (auto& [nm, ps] : settings) {
            double a = evalDetections(true, "SVM", ps, false), b = evalDetections(false, "MLP", ps, false);
            printf("  %-30s %.3f   %.3f\n", nm.c_str(), a, b);
            ab << nm << "," << a << "," << b << "\n";
        }
    }
    std::cout << "\nsaved detections.csv, frames.txt, proposal_quality.csv, scene tables and plots to " << outDir << "\n";
    return 0;
}
