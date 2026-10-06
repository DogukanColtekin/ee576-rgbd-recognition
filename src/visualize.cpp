// visualize.cpp - see what the pipeline does.
//
// Mode 1: object dataset - GT mask (green), our plane segmentation (red), depth-Otsu
//         segmentation (yellow) and the segmentation ROI (grey)
//   ./visualize --mode object --data <rgbd-dataset> --out viz_obj [--every 16]
//
// Mode 2: scene - GT boxes (white) and recognised detections (class colour, after
//         class-agnostic NMS, best class >= tau) on the processed scene frames
//   ./visualize --mode scene --scene <kitchen_small_1 dir> --dets out_scene/detections.csv
//               --gt gt.csv --out viz_scene [--tau 0.5 --scorer svm]
#include "common.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
using namespace rgbd;

static void drawContourOf(cv::Mat& img, const cv::Mat& mask, const cv::Scalar& col, int th = 2) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(mask.clone(), cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img, cs, -1, col, th);
}

int main(int argc, char** argv) {
    const std::string mode = argval(argc, argv, "--mode", "object");
    const std::string out  = argval(argc, argv, "--out", "viz");
    fs::create_directories(out);

    if (mode == "object") {
        const std::string data = argval(argc, argv, "--data", "rgbd-dataset");
        const int EVERY = std::stoi(argval(argc, argv, "--every", "16"));
        int n = 0;
        for (const auto& cls : CLASSES) {
            for (auto& inst : listDirsNatural((fs::path(data) / cls).string(), cls + "_")) {
                auto frames = listRgbFramesNatural(inst);
                for (size_t i = 0; i < frames.size(); i += EVERY) {
                    ObjectFrame f;
                    if (!readObjectFrame(frames[i], f)) continue;
                    cv::Rect roi = objectROI(f.loc, f.depth.size());
                    cv::Mat segP = segmentObject(f.depth, f.loc);
                    cv::Mat segD = segmentDepth(f.depth, roi);
                    cv::Mat vis = f.rgb.clone();
                    cv::rectangle(vis, roi, {160, 160, 160}, 1);
                    drawContourOf(vis, f.maskGT, {0, 255, 0});
                    drawContourOf(vis, segD, {0, 220, 255}, 1);
                    drawContourOf(vis, segP, {0, 0, 255});
                    char buf[96];
                    snprintf(buf, sizeof(buf), "IoU plane=%.2f  depth=%.2f", maskIoU(segP, f.maskGT), maskIoU(segD, f.maskGT));
                    cv::rectangle(vis, {0, 0, 330, 32}, {0, 0, 0}, -1);
                    cv::putText(vis, buf, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);
                    cv::imwrite((fs::path(out) / (fs::path(frames[i]).stem().string() + "_seg.png")).string(), vis);
                    ++n;
                }
                break;                                       // first instance per class
            }
        }
        printf("wrote %d overlays to %s  (green=GT, red=plane seg, yellow=depth-Otsu)\n", n, out.c_str());
        return 0;
    }

    // ---------------- scene ----------------
    const std::string scene = argval(argc, argv, "--scene", "");
    const std::string dets  = argval(argc, argv, "--dets", "out_scene/detections.csv");
    const std::string gtCsv = argval(argc, argv, "--gt", "");
    const float TAU = std::stof(argval(argc, argv, "--tau", "0.5"));
    const int S = argval(argc, argv, "--scorer", "svm") == "mlp" ? 1 : 0;

    std::map<int, std::vector<std::pair<cv::Rect2d, std::array<float, 5>>>> byFrame;
    std::ifstream in(dets);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::stringstream ss(line); std::string tok; std::vector<double> v;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        if (v.size() < 16) continue;
        std::array<float, 5> s;
        for (int c = 0; c < 5; ++c) s[c] = (float)v[6 + 5 * S + c];
        byFrame[(int)v[0]].push_back({cv::Rect2d(v[1], v[2], v[3] - v[1], v[4] - v[2]), s});
    }
    std::map<int, std::vector<std::pair<std::string, cv::Rect>>> gt;
    if (!gtCsv.empty()) {
        std::ifstream g(gtCsv);
        std::getline(g, line);
        while (std::getline(g, line)) {
            std::stringstream ss(line); std::string tok; std::vector<std::string> t;
            while (std::getline(ss, tok, ',')) t.push_back(tok);
            if (t.size() < 6) continue;
            int x1 = std::stoi(t[2]), y1 = std::stoi(t[3]), x2 = std::stoi(t[4]), y2 = std::stoi(t[5]);
            gt[std::stoi(t[0])].push_back({t[1], cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)});
        }
    }
    // frame index -> file, matched by the number in the file name (the first iteration
    // indexed a 0-based vector with the 1-based frame number: off by one frame)
    std::map<int, std::string> path;
    for (auto& p : listRgbFramesNatural(scene)) path[frameIndexFromPath(p)] = p;
    int n = 0;
    for (auto& [k, rows] : byFrame) {
        if (!path.count(k)) continue;
        cv::Mat img = cv::imread(path[k]);
        if (img.empty()) continue;
        for (auto& [cat, r] : gt[k]) {
            cv::rectangle(img, r, {255, 255, 255}, 1);
            cv::putText(img, cat, {r.x + 2, r.y + r.height - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.38, {255, 255, 255}, 1, cv::LINE_AA);
        }
        std::vector<cv::Rect2d> b; std::vector<float> sc; std::vector<int> cl;
        for (auto& [box, s] : rows) {
            int c = (int)(std::max_element(s.begin(), s.end()) - s.begin());
            if (s[c] < TAU) continue;
            b.push_back(box); sc.push_back(s[c]); cl.push_back(c);
        }
        for (int i : nmsGreedy(b, sc, 0.5)) {
            cv::rectangle(img, b[i], classColor(cl[i]), 2);
            char buf[64]; snprintf(buf, sizeof(buf), "%s %.2f", CLASSES[cl[i]].c_str(), sc[i]);
            cv::putText(img, buf, {(int)b[i].x, std::max(12, (int)b[i].y - 4)}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        classColor(cl[i]), 1, cv::LINE_AA);
        }
        char name[32]; snprintf(name, sizeof(name), "frame_%04d.png", k);
        cv::imwrite((fs::path(out) / name).string(), img);
        ++n;
    }
    printf("wrote %d annotated scene frames to %s (tau=%.2f; white = GT)\n", n, out.c_str(), TAU);
    return 0;
}
