// RGB-D recognition and tracking - part (d)
// Frame-to-frame association of recognised objects. For every pair of consecutive
// processed frames it reports which object IDs CONTINUE, which APPEAR (new or
// re-appearing) and which DISAPPEAR, and draws the timeline (x = frame, y = object id).
//
// Pipeline per frame:
//   0. segments failing the physical tabletop-object prior (GeometryGate) are dropped
//   1. each detection gets the class with the highest calibrated score; kept if >= tau
//   2. class-agnostic NMS - FastSAM returns nested/overlapping masks of one object
//      (mug + handle + mug-with-shadow ...); without this every duplicate opened a
//      new track (the first iteration ended with 964 IDs for six objects)
//   3. association with existing tracks on IoU (or normalised centre distance)
//      against a constant-velocity predicted box (the camera moves, boxes shift by
//      tens of pixels between processed frames); greedy in decreasing affinity; a
//      class change is accepted only on strong overlap, and the track's class is the
//      argmax of its accumulated scores
//   4. M-of-N confirmation: a track receives an ID only after M hits in its last N
//      frames, which suppresses one-frame false positives; unconfirmed tracks and
//      tracks missed for more than --maxmiss frames are dropped
//
// Usage:
//   ./tracking --dets out_scene/detections.csv --frames out_scene/frames.txt --out out_track
//              [--gt gt.csv] [--scorer mlp|svm] [--tau 0.5] [--iou 0.3] [--nms 0.5]
//              [--confirm 3] [--window 4] [--maxmiss 3] [--cdist 0.5] [--contain 0.8] [--no-gate] [--debug]
#include "common.hpp"
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace rgbd;

struct Det { int frame; cv::Rect2d box; std::array<float, 5> s; SegmentGeometry geo; bool hasGeo = false; };
struct Track {
    cv::Rect2d box; cv::Point2d vel{0, 0};
    std::array<double, 5> acc{};                            // accumulated class scores
    std::deque<int> hist;                                   // 1 = hit, 0 = miss (last N)
    int misses = 0, id = -1, lastFrameIdx = -1;
    std::vector<std::pair<int, cv::Rect2d>> pending;        // hits before confirmation
    int cls() const { return (int)(std::max_element(acc.begin(), acc.end()) - acc.begin()); }
};

static std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main(int argc, char** argv) {
    const std::string detsCsv  = argval(argc, argv, "--dets", "out_scene/detections.csv");
    std::string framesTxt      = argval(argc, argv, "--frames", "");
    const std::string outDir   = argval(argc, argv, "--out", "out_track");
    const std::string gtCsv    = argval(argc, argv, "--gt", "");
    const std::string scorer   = argval(argc, argv, "--scorer", "mlp");
    const float TAU     = std::stof(argval(argc, argv, "--tau", "0.5"));
    const float IOU_TRK = std::stof(argval(argc, argv, "--iou", "0.3"));
    const float NMS_IOU = std::stof(argval(argc, argv, "--nms", "0.5"));
    const int   M       = std::stoi(argval(argc, argv, "--confirm", "3"));
    const int   N       = std::stoi(argval(argc, argv, "--window", "4"));
    const int   MAXMISS = std::stoi(argval(argc, argv, "--maxmiss", "3"));
    const float CDIST   = std::stof(argval(argc, argv, "--cdist", "0.5"));
    const float CONTAIN = std::stof(argval(argc, argv, "--contain", "0.8"));
    const int S = scorer == "mlp" ? 1 : 0;
    const bool DEBUG = hasFlag(argc, argv, "--debug");
    const bool USE_GATE = !hasFlag(argc, argv, "--no-gate");
    const GeometryGate gate;
    std::filesystem::create_directories(outDir);
    if (framesTxt.empty()) framesTxt = (std::filesystem::path(detsCsv).parent_path() / "frames.txt").string();

    // ---------------- load ----------------
    std::ifstream f(detsCsv);
    if (!f) { std::cerr << "cannot open " << detsCsv << "\n"; return 1; }
    std::string line;
    std::getline(f, line);
    if (line.find("conf") == std::string::npos) { std::cerr << "detections.csv is from an older version - rerun scene_eval\n"; return 1; }
    std::map<int, std::vector<Det>> byFrame;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> v;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        if (v.size() < 16) continue;
        Det d;
        d.frame = (int)v[0];
        d.box = cv::Rect2d(v[1], v[2], v[3] - v[1], v[4] - v[2]);
        for (int c = 0; c < 5; ++c) d.s[c] = (float)v[6 + 5 * S + c];
        if (v.size() >= 21) {
            d.geo.e1 = (float)v[16]; d.geo.e2 = (float)v[17]; d.geo.e3 = (float)v[18];
            d.geo.hLow = (float)v[19]; d.geo.hHigh = (float)v[20]; d.hasGeo = true;
        }
        byFrame[d.frame].push_back(d);
    }
    std::vector<int> frames;
    {
        std::ifstream ff(framesTxt);
        int k;
        while (ff >> k) frames.push_back(k);
    }
    if (frames.empty()) {                                   // fallback: frames with detections
        std::cerr << "WARNING: " << framesTxt << " not found - using only frames that have detections\n";
        for (auto& [k, v] : byFrame) frames.push_back(k);
    }
    std::sort(frames.begin(), frames.end());

    // ---------------- tracking ----------------
    std::vector<Track> tracks;
    int nextId = 0;
    std::vector<TimelinePoint> timeline;
    std::map<int, std::map<int, cv::Rect2d>> visible;       // frame -> id -> box
    std::map<int, int> idCls;
    std::ofstream ev(outDir + "/events.csv");
    ev << "prev_frame,frame,n_continued,n_appeared,n_disappeared,continued_ids,appeared_ids,disappeared_ids,new_ids\n";
    std::set<int> prevVis, everSeen;
    int prevFrame = -1;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const int k = frames[fi];
        // 1+2: best class >= tau, class-agnostic NMS
        std::vector<Det> obs;
        {
            std::vector<Det> cand;
            for (auto& d : byFrame[k]) {
                float best = *std::max_element(d.s.begin(), d.s.end());
                if (USE_GATE && d.hasGeo && !gate.pass(d.geo)) continue;   // same prior as scene_eval
                if (best >= TAU) cand.push_back(d);
            }
            std::vector<cv::Rect2d> b; std::vector<float> sc;
            for (auto& d : cand) { b.push_back(d.box); sc.push_back(*std::max_element(d.s.begin(), d.s.end())); }
            for (int i : nmsGreedy(b, sc, NMS_IOU, CONTAIN)) obs.push_back(cand[i]);
        }
        // 3: association against predicted boxes
        struct Pair { double iou; int oi, ti; };
        std::vector<Pair> pairs;
        for (int ti = 0; ti < (int)tracks.size(); ++ti) {
            const Track& t = tracks[ti];
            const double dt = (double)fi - t.lastFrameIdx;
            cv::Rect2d pred(t.box.x + t.vel.x * dt, t.box.y + t.vel.y * dt, t.box.width, t.box.height);
            for (int oi = 0; oi < (int)obs.size(); ++oi) {
                const cv::Rect2d& o = obs[oi].box;
                double iou = std::max(boxIoU(o, pred), boxIoU(o, t.box));
                // second gate: centre distance relative to object size, for fast camera
                // motion / partially segmented boxes where IoU drops although it is
                // clearly the same object; similar size required
                double dx = (o.x + o.width / 2) - (pred.x + pred.width / 2);
                double dy = (o.y + o.height / 2) - (pred.y + pred.height / 2);
                double scale = std::max({pred.width, pred.height, 1.0});
                double sizeRatio = std::sqrt(o.area() / std::max(1.0, pred.area()));
                double dn = std::sqrt(dx * dx + dy * dy) / scale;
                bool near = dn < CDIST && sizeRatio > 0.5 && sizeRatio < 2.0;
                // class consistency: a detection whose best class differs from the
                // track's class may only continue it on strong overlap (class flicker
                // of the same object); otherwise a nearby false positive could steal it
                int oc = (int)(std::max_element(obs[oi].s.begin(), obs[oi].s.end()) - obs[oi].s.begin());
                bool same = oc == t.cls();
                bool ok = same ? (iou >= IOU_TRK || near) : iou >= 0.5;
                if (ok) pairs.push_back({iou + std::max(0.0, 1.0 - dn) * 0.5 + (same ? 0.5 : 0.0), oi, ti});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.iou > b.iou; });
        std::vector<int> assign(obs.size(), -1);
        std::vector<char> taken(tracks.size(), 0);
        for (auto& p : pairs)
            if (assign[p.oi] < 0 && !taken[p.ti]) { assign[p.oi] = p.ti; taken[p.ti] = 1; }

        if (DEBUG) {
            printf("frame %d:\n", k);
            for (int oi = 0; oi < (int)obs.size(); ++oi) {
                const auto& b = obs[oi].box;
                int c = (int)(std::max_element(obs[oi].s.begin(), obs[oi].s.end()) - obs[oi].s.begin());
                printf("  obs %d [%4.0f %4.0f %4.0f %4.0f] %-10s -> ", oi, b.x, b.y, b.x + b.width, b.y + b.height, CLASSES[c].c_str());
                if (assign[oi] >= 0) printf("track(id=%d, prev [%4.0f %4.0f %4.0f %4.0f])\n", tracks[assign[oi]].id,
                    tracks[assign[oi]].box.x, tracks[assign[oi]].box.y, tracks[assign[oi]].box.x + tracks[assign[oi]].box.width,
                    tracks[assign[oi]].box.y + tracks[assign[oi]].box.height);
                else printf("NEW\n");
            }
        }
        // update matched / unmatched tracks
        for (int ti = 0; ti < (int)tracks.size(); ++ti) {
            Track& t = tracks[ti];
            if (!taken[ti]) { t.misses++; t.hist.push_back(0); }
        }
        for (int oi = 0; oi < (int)obs.size(); ++oi) {
            if (assign[oi] >= 0) {
                Track& t = tracks[assign[oi]];
                const double dt = std::max(1.0, (double)fi - t.lastFrameIdx);
                cv::Point2d c0(t.box.x + t.box.width / 2, t.box.y + t.box.height / 2);
                cv::Point2d c1(obs[oi].box.x + obs[oi].box.width / 2, obs[oi].box.y + obs[oi].box.height / 2);
                t.vel = 0.5 * t.vel + 0.5 * (c1 - c0) * (1.0 / dt);
                t.box = obs[oi].box; t.misses = 0; t.lastFrameIdx = (int)fi; t.hist.push_back(1);
                for (int c = 0; c < 5; ++c) t.acc[c] += obs[oi].s[c];
                if (t.id < 0) t.pending.push_back({k, t.box});
            } else {
                Track t;
                t.box = obs[oi].box; t.lastFrameIdx = (int)fi; t.hist.push_back(1);
                for (int c = 0; c < 5; ++c) t.acc[c] = obs[oi].s[c];
                t.pending.push_back({k, t.box});
                tracks.push_back(t);
            }
        }
        // 4: confirmation, pruning
        std::vector<Track> alive;
        for (auto& t : tracks) {
            while ((int)t.hist.size() > N) t.hist.pop_front();
            int hits = (int)std::count(t.hist.begin(), t.hist.end(), 1);
            if (t.id < 0 && hits >= M) {
                t.id = nextId++;
                for (auto& [pf, pb] : t.pending) visible[pf][t.id] = pb;   // back-fill
                t.pending.clear();
            } else if (t.id >= 0 && t.lastFrameIdx == (int)fi) {
                visible[k][t.id] = t.box;
            }
            if (t.id >= 0) idCls[t.id] = t.cls();
            bool drop = t.id < 0 ? (int)t.hist.size() >= N && hits < M : t.misses > MAXMISS;
            if (!drop) alive.push_back(t);
        }
        tracks.swap(alive);
    }

    // ---------------- events for consecutive processed frames ----------------
    // (computed after the pass so back-filled first sightings are included)
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const int k = frames[fi];
        std::set<int> cur;
        for (auto& [id, b] : visible[k]) cur.insert(id);
        std::vector<int> cont, app, dis, fresh;
        for (int id : cur) {
            (prevVis.count(id) ? cont : app).push_back(id);
            if (!everSeen.count(id)) fresh.push_back(id);
        }
        for (int id : prevVis) if (!cur.count(id)) dis.push_back(id);
        if (fi > 0)
            ev << prevFrame << "," << k << "," << cont.size() << "," << app.size() << "," << dis.size() << ","
               << join(cont) << "," << join(app) << "," << join(dis) << "," << join(fresh) << "\n";
        everSeen.insert(cur.begin(), cur.end());
        prevVis = cur; prevFrame = k;
        for (int id : cur) timeline.push_back({k, id, idCls[id]});
    }

    // ---------------- tracks.csv, optional GT check ----------------
    std::map<int, std::vector<int>> framesOf;
    for (auto& p : timeline) framesOf[p.id].push_back(p.frame);
    {
        std::ofstream tf(outDir + "/tracks.csv");
        tf << "id,class,first_frame,last_frame,n_frames\n";
        for (auto& [id, v] : framesOf)
            tf << id << "," << CLASSES[idCls[id]] << "," << v.front() << "," << v.back() << "," << v.size() << "\n";
    }
    printf("processed frames: %zu   confirmed object IDs: %d\n", frames.size(), nextId);
    for (auto& [id, v] : framesOf)
        printf("  id %3d  %-11s frames %d..%d (%zu)\n", id, CLASSES[idCls[id]].c_str(), v.front(), v.back(), v.size());

    if (!gtCsv.empty()) {
        std::ifstream g(gtCsv);
        std::getline(g, line);
        std::map<int, std::vector<std::pair<int, cv::Rect2d>>> gt;  // frame -> (cls, box)
        while (std::getline(g, line)) {
            std::stringstream ss(line); std::string tok; std::vector<std::string> t;
            while (std::getline(ss, tok, ',')) t.push_back(tok);
            if (t.size() < 6) continue;
            auto it = std::find(CLASSES.begin(), CLASSES.end(), t[1]);
            if (it == CLASSES.end()) continue;
            double x1 = std::stod(t[2]), y1 = std::stod(t[3]), x2 = std::stod(t[4]), y2 = std::stod(t[5]);
            gt[std::stoi(t[0])].push_back({(int)(it - CLASSES.begin()), cv::Rect2d(x1, y1, x2 - x1 + 1, y2 - y1 + 1)});
        }
        int tp = 0, nTrk = 0, nGT = 0, covered = 0;
        std::map<int, std::vector<int>> gtCls;
        for (int k : frames) {
            auto& gv = gt[k];
            for (auto& [c, b] : gv) gtCls[k].push_back(c);
            std::vector<char> used(gv.size(), 0);
            for (auto& [id, b] : visible[k]) {
                ++nTrk;
                int bj = -1; double bi = 0.5;
                for (size_t j = 0; j < gv.size(); ++j)
                    if (!used[j] && gv[j].first == idCls[id]) { double u = boxIoU(b, gv[j].second); if (u >= bi) { bi = u; bj = (int)j; } }
                if (bj >= 0) { used[bj] = 1; ++tp; }
            }
            nGT += (int)gv.size();
            covered += (int)std::count(used.begin(), used.end(), 1);
        }
        printf("GT check (same class, IoU>=0.5): tracked boxes correct %d/%d (%.2f), GT object-frames covered %d/%d (%.2f)\n",
               tp, nTrk, nTrk ? (double)tp / nTrk : 0, covered, nGT, nGT ? (double)covered / nGT : 0);
        std::ofstream gs(outDir + "/gt_check.csv");
        gs << "tracked_boxes,correct,precision,gt_object_frames,covered,recall\n"
           << nTrk << "," << tp << "," << (nTrk ? (double)tp / nTrk : 0) << "," << nGT << "," << covered << ","
           << (nGT ? (double)covered / nGT : 0) << "\n";
        plotPresence(outDir + "/gt_presence.png", "Ground truth: class presence per processed frame", gtCls, frames);
    }
    char tt[160];
    snprintf(tt, 160, "Object presence over time (%s, tau=%.2f, %d IDs)", scorer.c_str(), TAU, nextId);
    plotTimeline(outDir + "/timeline.png", tt, timeline, frames);
    std::cout << "saved events.csv, tracks.csv, timeline.png to " << outDir << "\n";
    return 0;
}
