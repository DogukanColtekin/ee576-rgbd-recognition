// RGB-D recognition and tracking - common implementation (C++17 / OpenCV).
#include "common.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <numeric>
#include <random>
#include <regex>
#include <set>

namespace fs = std::filesystem;

namespace rgbd {

const std::vector<std::string> CLASSES = {"bowl", "cap", "cereal_box", "coffee_mug", "soda_can"};
const std::vector<float> REPORT_TAUS = {0.3f, 0.5f, 0.7f};

std::vector<float> denseTaus() {
    std::vector<float> t;
    for (int i = 1; i <= 99; ++i) t.push_back(i / 100.f);
    return t;
}

std::string argval(int argc, char** argv, const std::string& key, const std::string& dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return dflt;
}
bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (key == argv[i]) return true;
    return false;
}

// ============================ natural sort ============================
static bool naturalLess(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (std::isdigit((unsigned char)a[i]) && std::isdigit((unsigned char)b[j])) {
            size_t i2 = i, j2 = j;
            while (i2 < a.size() && std::isdigit((unsigned char)a[i2])) ++i2;
            while (j2 < b.size() && std::isdigit((unsigned char)b[j2])) ++j2;
            long long na = std::stoll(a.substr(i, i2 - i));
            long long nb = std::stoll(b.substr(j, j2 - j));
            if (na != nb) return na < nb;
            i = i2; j = j2;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i; ++j;
        }
    }
    return a.size() < b.size();
}

std::vector<std::string> listDirsNatural(const std::string& dir, const std::string& prefix) {
    std::vector<std::string> out;
    if (!fs::is_directory(dir)) return out;
    for (auto& e : fs::directory_iterator(dir))
        if (e.is_directory() && e.path().filename().string().rfind(prefix, 0) == 0)
            out.push_back(e.path().string());
    std::sort(out.begin(), out.end(), naturalLess);
    return out;
}

static bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::vector<std::string> listRgbFramesNatural(const std::string& dir) {
    std::vector<std::string> out;
    if (!fs::is_directory(dir)) return out;
    for (auto& e : fs::directory_iterator(dir)) {
        std::string p = e.path().string();
        if (endsWith(p, ".png") && !endsWith(p, "_depth.png") && !endsWith(p, "_mask.png"))
            out.push_back(p);
    }
    std::sort(out.begin(), out.end(), naturalLess);
    return out;
}

// ============================ IO ============================
cv::Point readLoc(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {1, 1};
    std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::regex num("\\d+");
    auto it = std::sregex_iterator(txt.begin(), txt.end(), num);
    std::vector<int> v;
    for (; it != std::sregex_iterator() && v.size() < 2; ++it) v.push_back(std::stoi(it->str()));
    if (v.size() < 2) return {1, 1};
    return {v[0], v[1]};
}

static std::string replaceSuffix(const std::string& p, const std::string& suf) {
    return p.substr(0, p.size() - 4) + suf;                // strip ".png"
}

bool readObjectFrame(const std::string& rgbPath, ObjectFrame& out) {
    out.rgb = cv::imread(rgbPath, cv::IMREAD_COLOR);
    cv::Mat d16 = cv::imread(replaceSuffix(rgbPath, "_depth.png"), cv::IMREAD_UNCHANGED);
    cv::Mat mk  = cv::imread(replaceSuffix(rgbPath, "_mask.png"), cv::IMREAD_GRAYSCALE);
    if (out.rgb.empty() || d16.empty() || mk.empty()) return false;
    d16.convertTo(out.depth, CV_32F);
    out.maskGT = mk > 0;
    out.loc = readLoc(replaceSuffix(rgbPath, "_loc.txt"));
    return true;
}

bool readSceneFrame(const std::string& rgbPath, cv::Mat& rgb, cv::Mat& depth) {
    rgb = cv::imread(rgbPath, cv::IMREAD_COLOR);
    cv::Mat d16 = cv::imread(replaceSuffix(rgbPath, "_depth.png"), cv::IMREAD_UNCHANGED);
    if (rgb.empty() || d16.empty()) return false;
    d16.convertTo(depth, CV_32F);
    return true;
}

int frameIndexFromPath(const std::string& path) {
    std::smatch m;
    static const std::regex re("_(\\d+)\\.png$");
    if (std::regex_search(path, m, re)) return std::stoi(m[1].str());
    return -1;
}

// ============================ metrics ============================
double maskIoU(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat ia, ua;
    cv::bitwise_and(a, b, ia);
    cv::bitwise_or(a, b, ua);
    double u = cv::countNonZero(ua);
    return u > 0 ? cv::countNonZero(ia) / u : 0.0;
}

double boxIoU(const cv::Rect2d& a, const cv::Rect2d& b) {
    double xa = std::max(a.x, b.x), ya = std::max(a.y, b.y);
    double xb = std::min(a.x + a.width,  b.x + b.width);
    double yb = std::min(a.y + a.height, b.y + b.height);
    double inter = std::max(0.0, xb - xa) * std::max(0.0, yb - ya);
    double uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0;
}

cv::Rect2d maskBox(const cv::Mat& mask) {
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    if (nz.empty()) return {};
    cv::Rect r = cv::boundingRect(nz);                      // width = xmax - xmin + 1
    return cv::Rect2d(r.x, r.y, r.width, r.height);
}

std::vector<int> nmsGreedy(const std::vector<cv::Rect2d>& boxes,
                           const std::vector<float>& scores, double iouThr, double containThr) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    std::vector<char> sup(boxes.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        int a = order[i];
        if (sup[a]) continue;
        keep.push_back(a);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int b = order[j];
            if (sup[b]) continue;
            if (boxIoU(boxes[a], boxes[b]) > iouThr) { sup[b] = 1; continue; }
            if (containThr < 1.0 && boxes[b].area() > 0 &&
                (boxes[a] & boxes[b]).area() / boxes[b].area() > containThr) sup[b] = 1;
        }
    }
    return keep;
}

// ============================ segmentation helpers ============================
cv::Rect objectROI(cv::Point loc, cv::Size img, cv::Size win) {
    int x0 = std::clamp(loc.x, 0, img.width - 1), y0 = std::clamp(loc.y, 0, img.height - 1);
    int x1 = std::min(img.width, x0 + win.width), y1 = std::min(img.height, y0 + win.height);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

static void fillHoles(cv::Mat& m) {
    cv::Mat inv = cv::Mat::zeros(m.rows + 2, m.cols + 2, CV_8U);
    m.copyTo(inv(cv::Rect(1, 1, m.cols, m.rows)));
    cv::floodFill(inv, cv::Point(0, 0), 255);
    cv::Mat holes = ~inv(cv::Rect(1, 1, m.cols, m.rows));
    m |= holes;
}

// ============================ segmentation A (depth + Otsu) ============================
cv::Mat segmentDepth(const cv::Mat& depthMM, const cv::Rect& roi, int minArea) {
    const int H = depthMM.rows, W = depthMM.cols;
    cv::Mat out = cv::Mat::zeros(H, W, CV_8U);
    if (roi.area() <= 0) return out;
    cv::Mat d = depthMM(roi);
    cv::Mat valid = d > 0;
    if (cv::countNonZero(valid) < minArea) return out;

    // Otsu on the VALID ROI pixels only. (The first iteration thresholded the whole image,
    // where the zeroed outside-ROI area dominated the histogram, so the threshold
    // ended up selecting most of the ROI including the support surface.)
    double vmin, vmax;
    cv::minMaxLoc(d, &vmin, &vmax, nullptr, nullptr, valid);
    if (vmax - vmin < 1) return out;
    cv::Mat inv8;                                          // near = bright
    d.convertTo(inv8, CV_8U, -255.0 / (vmax - vmin), 255.0 * vmax / (vmax - vmin));
    std::vector<uchar> vals;
    for (int y = 0; y < d.rows; ++y)
        for (int x = 0; x < d.cols; ++x)
            if (valid.at<uchar>(y, x)) vals.push_back(inv8.at<uchar>(y, x));
    cv::Mat v1(1, (int)vals.size(), CV_8U, vals.data());
    cv::Mat dummy;
    double thr = cv::threshold(v1, dummy, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat th = (inv8 > thr) & valid;

    cv::morphologyEx(th, th, cv::MORPH_OPEN,  cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    cv::morphologyEx(th, th, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));

    cv::Mat lbl, stats, cent;
    int n = cv::connectedComponentsWithStats(th, lbl, stats, cent);
    if (n <= 1) return out;
    int best = -1; double bestDepth = 1e18;               // nearest blob with enough area
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < minArea) continue;
        std::vector<float> dv;
        for (int y = 0; y < d.rows; ++y)
            for (int x = 0; x < d.cols; ++x)
                if (lbl.at<int>(y, x) == i && valid.at<uchar>(y, x)) dv.push_back(d.at<float>(y, x));
        if (dv.empty()) continue;
        std::nth_element(dv.begin(), dv.begin() + dv.size() / 2, dv.end());
        if (dv[dv.size() / 2] < bestDepth) { bestDepth = dv[dv.size() / 2]; best = i; }
    }
    if (best < 0) return out;
    cv::Mat sel = (lbl == best);
    fillHoles(sel);
    sel.copyTo(out(roi));
    return out;
}

// ============================ segmentation B (RANSAC plane) ============================
struct Plane { cv::Vec3f n; float d; };                    // n.p + d = signed height (m)

static bool fitPlaneRansac(const std::vector<cv::Point3f>& pts, const PlaneSegParams& P,
                           Plane& out, bool supportScore = true,
                           const cv::Vec3f* parallelTo = nullptr, float maxAngleDeg = 12.f) {
    const float cosMax = std::cos(maxAngleDeg * (float)CV_PI / 180.f);
    if (pts.size() < 50) return false;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> uni(0, pts.size() - 1);
    // score hypotheses on a random subset (fast), refine on all inliers
    std::vector<cv::Point3f> sub;
    const size_t NS = std::min<size_t>(4000, pts.size());
    for (size_t i = 0; i < NS; ++i) sub.push_back(pts[uni(rng)]);

    int bestCnt = -1; Plane best{};
    for (int it = 0; it < P.iters; ++it) {
        cv::Point3f p1 = pts[uni(rng)], p2 = pts[uni(rng)], p3 = pts[uni(rng)];
        cv::Point3f n = (p2 - p1).cross(p3 - p1);
        float nn = std::sqrt(n.dot(n));
        if (nn < 1e-9f) continue;
        n *= 1.f / nn;
        float d = -n.dot(p1);
        if (d < 0) { n = -n; d = -d; }                     // camera (origin) on the + side
        // The support surface faces the camera AND points "up" in the image (n.y < 0
        // in camera coords: image y grows downwards). This rejects the back wall /
        // background planes, which face the camera but tilt the other way.
        if (n.y > -0.25f) continue;
        // The capture rig has no camera roll, so a horizontal support plane has a
        // normal with almost no image-x component. This rejects oblique planes that
        // slice diagonally through a large flat object face (cereal box).
        if (std::fabs(n.x) > 0.3f) continue;
        if (parallelTo && parallelTo->dot(cv::Vec3f(n.x, n.y, n.z)) < cosMax) continue;
        // Support score = inliers + points standing on the plane (minH..maxH above it).
        // Pure inlier counting picks the largest flat patch, which is often the top or
        // front of the object itself (cereal box); the surface the object stands on is
        // the one with the object above it.
        int inl = 0, above = 0;
        for (const auto& q : sub) {
            float h = n.dot(q) + d;
            if (std::fabs(h) < P.inlierThr) ++inl;
            else if (h > P.minH && h < P.maxH) ++above;
        }
        if (inl < 0.05 * sub.size()) continue;
        int cnt = supportScore ? inl + above : inl;
        if (cnt > bestCnt) { bestCnt = cnt; best = {cv::Vec3f(n.x, n.y, n.z), d}; }
    }
    if (bestCnt < 20) return false;

    // least-squares refinement on all inliers (smallest-eigenvector of covariance)
    std::vector<cv::Point3f> inl;
    for (const auto& q : pts)
        if (std::fabs(best.n.dot(cv::Vec3f(q.x, q.y, q.z)) + best.d) < P.inlierThr) inl.push_back(q);
    if (inl.size() >= 3) {
        cv::Mat M((int)inl.size(), 3, CV_32F, inl.data());
        cv::PCA pca(M, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Vec3f n(pca.eigenvectors.at<float>(2, 0), pca.eigenvectors.at<float>(2, 1),
                    pca.eigenvectors.at<float>(2, 2));
        cv::Vec3f c(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
        float d = -n.dot(c);
        if (d < 0) { n = -n; d = -d; }
        if (n[1] < -0.25f && std::fabs(n[0]) <= 0.3f &&
            (!parallelTo || parallelTo->dot(n) >= cosMax)) best = {n, d};
    }
    out = best;
    return true;
}

static const char* segDbg() { static const char* d = std::getenv("RGBD_SEGDBG"); return d; }

cv::Mat segmentPlane(const cv::Mat& depthMM, const cv::Rect& roi, const PlaneSegParams& P) {
    const int H = depthMM.rows, W = depthMM.cols;
    static int dbgId = 0;
    const bool DBG = segDbg() != nullptr;
    if (DBG) ++dbgId;
    cv::Mat out = cv::Mat::zeros(H, W, CV_8U);
    if (roi.area() <= 0) return out;

    std::vector<cv::Point3f> pts;
    std::vector<cv::Point> px;
    for (int v = roi.y; v < roi.y + roi.height; ++v) {
        const float* dr = depthMM.ptr<float>(v);
        for (int u = roi.x; u < roi.x + roi.width; ++u)
            if (dr[u] > 0) { pts.push_back(backproject((float)u, (float)v, dr[u])); px.emplace_back(u, v); }
    }
    Plane pl;
    if (!fitPlaneRansac(pts, P, pl)) { if (DBG) printf("  [seg] no P1\n"); return out; }

    // keep points between minH and maxH above the support plane
    auto height = [](const Plane& q, const cv::Point3f& p) { return q.n.dot(cv::Vec3f(p.x, p.y, p.z)) + q.d; };
    std::vector<int> keep;
    for (size_t i = 0; i < pts.size(); ++i) {
        float h = height(pl, pts[i]);
        if (h > P.minH && h < P.maxH) keep.push_back((int)i);
    }
    if (DBG) {
        int inl = 0; for (auto& q : pts) if (std::fabs(height(pl, q)) < P.inlierThr) ++inl;
        printf("  [seg] P1 n=(%.2f %.2f %.2f) d=%.3f inl=%d/%zu keep=%zu\n", pl.n[0], pl.n[1], pl.n[2], pl.d, inl, pts.size(), keep.size());
    }
    // Stacked support: the best support plane in the ROI is usually the TABLE, and the
    // turntable standing on it is then "above the plane" and swallowed into the object.
    // Look ONCE for a second support among the kept points that is parallel to the
    // table and 3-30 cm above it (the turntable top). If it holds a fair share of the
    // points and enough remain above it, remove it and everything below it.
    // (Real data showed that a second pass, or non-parallel planes, cut into the object
    // itself: bowl interior, cap crown, leaning cereal-box face.)
    if (keep.size() > 400) {
        std::vector<cv::Point3f> kp;
        kp.reserve(keep.size());
        for (int i : keep) kp.push_back(pts[i]);
        Plane p2;
        if (fitPlaneRansac(kp, P, p2, /*supportScore=*/false, &pl.n, 12.f)) {
            int inl = 0;
            std::vector<int> above;
            std::vector<float> lift;                        // height of P2 inliers over P1
            for (int i : keep) {
                float h = height(p2, pts[i]);
                if (std::fabs(h) < P.inlierThr) { ++inl; lift.push_back(height(pl, pts[i])); }
                else if (h > P.minH && h < P.maxH) above.push_back(i);
            }
            float off = 0;
            if (!lift.empty()) {
                std::nth_element(lift.begin(), lift.begin() + lift.size() / 2, lift.end());
                off = lift[lift.size() / 2];
            }
            bool ok = inl >= 0.10 * keep.size() && above.size() >= 300 && off > 0.03f && off < 0.30f;
            if (DBG) printf("  [seg] P2 n=(%.2f %.2f %.2f) d=%.3f lift=%.3f inl=%d above=%zu of %zu -> %s\n",
                            p2.n[0], p2.n[1], p2.n[2], p2.d, off, inl, above.size(), keep.size(), ok ? "removed" : "kept");
            if (ok) keep.swap(above);
        }
    }
    cv::Mat m = cv::Mat::zeros(roi.size(), CV_8U);
    for (int i : keep) m.at<uchar>(px[i].y - roi.y, px[i].x - roi.x) = 255;
    // cut along depth discontinuities so the object does not merge with background
    // that happens to be adjacent in the image (e.g. the wall behind the turntable)
    const float jumpMM = P.edgeJump * 1000.f;
    cv::Mat cut = m.clone();
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x) {
            if (!m.at<uchar>(y, x)) continue;
            float dc = depthMM.at<float>(y + roi.y, x + roi.x);
            const int nb[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
            for (auto& o : nb) {
                int xx = x + o[0], yy = y + o[1];
                if (xx < 0 || yy < 0 || xx >= m.cols || yy >= m.rows) continue;
                float dn = depthMM.at<float>(yy + roi.y, xx + roi.x);
                if (dn > 0 && std::fabs(dn - dc) > jumpMM) { cut.at<uchar>(y, x) = 0; break; }
            }
        }
    cv::morphologyEx(cut, cut, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    // 4-connectivity: with 8-connectivity the background leaked through diagonal
    // gaps of the one-pixel boundary cut
    cv::Mat lbl, stats, cent;
    int n = cv::connectedComponentsWithStats(cut, lbl, stats, cent, 4);
    if (n <= 1) return out;
    // Washington *_loc.txt is the top-left of the object crop, so the object's own
    // bounding box starts near the ROI origin: prefer big components whose box
    // top-left is close to it. (The first iteration measured the distance from the
    // component CENTROID to loc, i.e. to a corner of the object, not to the object.)
    // per-component contact with the ROI frame (support/background pieces run along it)
    std::vector<int> onFrame(n, 0), perim(n, 0);
    for (int y = 0; y < lbl.rows; ++y)
        for (int x = 0; x < lbl.cols; ++x) {
            int l = lbl.at<int>(y, x);
            if (!l) continue;
            bool frame = x == 0 || y == 0 || x == lbl.cols - 1 || y == lbl.rows - 1;
            bool edge = frame || lbl.at<int>(y, x - 1) != l || lbl.at<int>(y, x + 1) != l ||
                        lbl.at<int>(y - 1, x) != l || lbl.at<int>(y + 1, x) != l;
            onFrame[l] += frame; perim[l] += edge;
        }
    auto contact = [&](int l) { return perim[l] ? (double)onFrame[l] / perim[l] : 1.0; };
    int best = -1; double bestScore = 0;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < P.minArea) continue;
        double dx = stats.at<int>(i, cv::CC_STAT_LEFT), dy = stats.at<int>(i, cv::CC_STAT_TOP);
        double score = area * std::exp(-std::sqrt(dx * dx + dy * dy) / 60.0) * std::max(0.05, 1.0 - 2.0 * contact(i));
        if (score > bestScore) { bestScore = score; best = i; }
    }
    if (DBG) {
        printf("  [seg] components=%d best=%d\n", n - 1, best);
        cv::Mat vis;
        cv::Mat d8; depthMM(roi).convertTo(d8, CV_8U, 255.0 / 2000.0);
        cv::cvtColor(d8, vis, cv::COLOR_GRAY2BGR);
        vis.setTo(cv::Scalar(0, 180, 255), m);
        vis.setTo(cv::Scalar(0, 0, 255), cut);
        if (best >= 0) vis.setTo(cv::Scalar(0, 255, 0), lbl == best);
        cv::imwrite(std::string(segDbg()) + "/seg_" + std::to_string(dbgId) + ".png", vis);
    }
    if (best < 0) return out;
    cv::Mat sel = (lbl == best);
    // The depth-edge cut also splits one object into pieces (bowl interior vs. front
    // rim, cap crown vs. brim, mug inside vs. outside). Re-attach pieces that touch the
    // selected component and stay clear of the ROI frame.
    for (int it = 0; it < 3; ++it) {
        cv::Mat dil;
        cv::dilate(sel, dil, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
        std::set<int> touch;
        for (int y = 0; y < lbl.rows; ++y)
            for (int x = 0; x < lbl.cols; ++x) {
                int l = lbl.at<int>(y, x);
                if (l && dil.at<uchar>(y, x) && !sel.at<uchar>(y, x)) touch.insert(l);
            }
        bool added = false;
        for (int l : touch)
            if (stats.at<int>(l, cv::CC_STAT_AREA) >= 30 && contact(l) < 0.10) { sel |= (lbl == l); added = true; }
        if (!added) break;
    }
    // undo the boundary cut / fill sensor holes, but only into pixels that are
    // above the plane or have no depth
    cv::Mat grown;
    cv::dilate(sel, grown, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    cv::Mat allowed = cut | (depthMM(roi) <= 0);   // never re-admit pixels across a depth edge
    sel = (grown & allowed) | sel;
    cv::morphologyEx(sel, sel, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    fillHoles(sel);
    sel.copyTo(out(roi));
    return out;
}

// Fraction of the mask outline that lies on the ROI frame. A true object sits inside
// its crop window (loc = crop top-left, with a margin); support surfaces / table that
// leaked into a mask run along the window border.
double borderContact(const cv::Mat& mask, const cv::Rect& roi) {
    cv::Mat m = mask(roi), er;
    cv::erode(m, er, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}), {-1, -1}, 1,
              cv::BORDER_CONSTANT, 0);
    int perim = cv::countNonZero(m) - cv::countNonZero(er);
    if (perim <= 0) return 1.0;
    int onFrame = cv::countNonZero(m.row(0)) + cv::countNonZero(m.row(m.rows - 1)) +
                  cv::countNonZero(m.col(0)) + cv::countNonZero(m.col(m.cols - 1));
    return (double)onFrame / perim;
}

cv::Mat segmentObject(const cv::Mat& depthMM, cv::Point loc, bool* usedFallback) {
    // Two independent segmentations, the plausible one is chosen WITHOUT ground truth:
    //  - lower border contact wins (leaked support surface touches the ROI frame)
    //  - if both are equally clean, the larger mask wins (a plane cut through a
    //    flat-faced object such as a cereal box only loses object pixels)
    cv::Rect roi = objectROI(loc, depthMM.size());
    cv::Mat mp = segmentPlane(depthMM, roi), md = segmentDepth(depthMM, roi);
    const double A = roi.area();
    int ap = cv::countNonZero(mp), ad = cv::countNonZero(md);
    bool vp = ap >= 200 && ap <= 0.85 * A, vd = ad >= 200 && ad <= 0.85 * A;
    bool useDepth;
    if (!vp || !vd) useDepth = !vp && vd;
    else {
        double cp = borderContact(mp, roi), cd = borderContact(md, roi);
        useDepth = std::fabs(cp - cd) > 0.05 ? cd < cp : ad > ap;
    }
    if (usedFallback) *usedFallback = useDepth;
    return useDepth ? md : mp;
}

// ============================ segment geometry ============================
PlaneModel fitFramePlane(const cv::Mat& depthMM) {
    std::vector<cv::Point3f> pts;
    for (int v = 0; v < depthMM.rows; v += 2)
        for (int u = 0; u < depthMM.cols; u += 2) {
            float d = depthMM.at<float>(v, u);
            if (d > 0) pts.push_back(backproject((float)u, (float)v, d));
        }
    PlaneModel pm;
    Plane pl;
    PlaneSegParams P;
    P.maxH = 0.6f;
    if (fitPlaneRansac(pts, P, pl)) { pm.n = pl.n; pm.d = pl.d; pm.ok = true; }
    return pm;
}

SegmentGeometry segmentGeometry(const cv::Mat& mask, const cv::Mat& depthMM, const PlaneModel* plane) {
    SegmentGeometry g;
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    std::vector<cv::Point3f> P;
    const size_t step = std::max<size_t>(1, nz.size() / 4000);
    for (size_t i = 0; i < nz.size(); i += step) {
        float d = depthMM.at<float>(nz[i]);
        if (d > 0) P.push_back(backproject((float)nz[i].x, (float)nz[i].y, d));
    }
    g.nPts = (int)P.size();
    if (P.size() < 20) return g;
    cv::Mat M((int)P.size(), 3, CV_32F, P.data());
    cv::PCA pca(M, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Mat proj = pca.project(M);
    auto spread = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[(size_t)(0.95 * (v.size() - 1))] - v[(size_t)(0.05 * (v.size() - 1))];
    };
    float e[3];
    for (int k = 0; k < 3; ++k) {
        std::vector<float> v(proj.rows);
        for (int r = 0; r < proj.rows; ++r) v[r] = proj.at<float>(r, k);
        e[k] = spread(v);
    }
    std::sort(e, e + 3, std::greater<float>());
    g.e1 = e[0]; g.e2 = e[1]; g.e3 = e[2];
    if (plane && plane->ok) {
        std::vector<float> h;
        for (auto& p : P) h.push_back(plane->n.dot(cv::Vec3f(p.x, p.y, p.z)) + plane->d);
        std::sort(h.begin(), h.end());
        g.hLow = h[(size_t)(0.05 * (h.size() - 1))];
        g.hHigh = h[(size_t)(0.95 * (h.size() - 1))];
    }
    return g;
}

// ============================ augmentation ============================
cv::Mat perturbMask(const cv::Mat& mask, cv::RNG& rng) {
    cv::Mat m = mask.clone();
    int op = rng.uniform(0, 3);
    int ks = rng.uniform(1, 3) * 2 + 1;                    // 3 or 5
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ks, ks});
    if (op == 0)      cv::dilate(m, m, k);
    else if (op == 1) cv::erode(m, m, k);
    for (int y = 0; y < m.rows; ++y) {                     // 5% pixel dropout
        uchar* r = m.ptr<uchar>(y);
        for (int x = 0; x < m.cols; ++x)
            if (r[x] && rng.uniform(0.f, 1.f) < 0.05f) r[x] = 0;
    }
    return m;
}

// ============================ descriptor ============================
// Normal from the cross product of back-projected neighbours. The first iteration used
// np.gradient of the depth image, i.e. dZ/dpixel with Z in metres and x,y in
// pixels; one pixel is ~1.6 mm at 0.9 m, so every normal collapsed to ~(0,0,1)
// and the "spin image" carried almost no surface-orientation information.
cv::Point3f normalAt(const cv::Mat& D, int u, int v, int step) {
    auto P = [&](int x, int y, cv::Point3f& p) {
        if (x < 0 || y < 0 || x >= D.cols || y >= D.rows) return false;
        float d = D.at<float>(y, x);
        if (d <= 0) return false;
        p = backproject((float)x, (float)y, d);
        return true;
    };
    for (int s = step; s >= 1; --s) {
        cv::Point3f c, l, r, t, b;
        if (!P(u, v, c)) break;
        bool hl = P(u - s, v, l), hr = P(u + s, v, r), ht = P(u, v - s, t), hb = P(u, v + s, b);
        if (!(hl || hr) || !(ht || hb)) continue;
        cv::Point3f dx = (hr ? r : c) - (hl ? l : c);
        cv::Point3f dy = (hb ? b : c) - (ht ? t : c);
        cv::Point3f n = dx.cross(dy);
        float nn = std::sqrt(n.dot(n));
        if (nn < 1e-12f) continue;
        n *= 1.f / nn;
        if (n.dot(c) > 0) n = -n;                          // face the camera
        return n;
    }
    return {0.f, 0.f, -1.f};
}

static std::vector<float> spinDescriptor(const std::vector<cv::Point3f>& cloudIn,
                                         const std::vector<cv::Point>& pxIn,
                                         const cv::Mat& depthMM,
                                         int nSamples = 20, float radius = 0.04f, int bins = 16) {
    std::vector<float> acc(bins * bins, 0.f);
    if ((int)cloudIn.size() < 30) return acc;
    std::vector<cv::Point3f> cloud = cloudIn;
    std::vector<cv::Point> px = pxIn;
    if (cloud.size() > 2000) {                             // subsample to 2000 points
        std::mt19937 r0(0);
        std::vector<size_t> idx(cloud.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), r0);
        idx.resize(2000);
        std::vector<cv::Point3f> c2; std::vector<cv::Point> p2;
        for (size_t i : idx) { c2.push_back(cloud[i]); p2.push_back(px[i]); }
        cloud.swap(c2); px.swap(p2);
    }
    std::mt19937 r1(42);
    std::vector<size_t> idx(cloud.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), r1);
    int nSel = std::min<int>(nSamples, (int)cloud.size());

    int cnt = 0;
    const float r2 = radius * radius;
    std::vector<float> hist(bins * bins);
    for (int s = 0; s < nSel; ++s) {
        const cv::Point3f p = cloud[idx[s]];
        const cv::Point3f nv = normalAt(depthMM, px[idx[s]].x, px[idx[s]].y);
        std::fill(hist.begin(), hist.end(), 0.f);
        float hsum = 0;
        for (const auto& q : cloud) {
            cv::Point3f diff = q - p;
            float d2 = diff.dot(diff);
            if (d2 >= r2 || d2 <= 1e-12f) continue;
            float beta  = diff.dot(nv);                    // elevation along the normal
            float alpha = std::sqrt(std::max(d2 - beta * beta, 0.f));   // radial distance
            int ai = std::min(bins - 1, (int)(alpha / radius * bins));
            int bi = std::clamp((int)((beta + radius) / (2 * radius) * bins), 0, bins - 1);
            hist[ai * bins + bi] += 1.f; hsum += 1.f;
        }
        if (hsum < 10) continue;
        for (int i = 0; i < bins * bins; ++i) acc[i] += hist[i] / hsum;
        ++cnt;
    }
    if (cnt > 0) for (auto& v : acc) v /= cnt;
    return acc;
}

bool describe(const cv::Mat& rgb, const cv::Mat& depthMM, const cv::Mat& mask,
              std::vector<float>& out) {
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    if ((int)nz.size() < 20) return false;
    cv::Rect bb = cv::boundingRect(nz);
    if (bb.width < 5 || bb.height < 5) return false;

    out.clear(); out.reserve(DESC_DIM);

    // 1) HSV colour histogram on masked pixels (8x4x4 = 128, L1-normalised)
    cv::Mat hsv;
    cv::cvtColor(rgb(bb), hsv, cv::COLOR_BGR2HSV);
    int histSize[] = {8, 4, 4};
    float hr[] = {0, 180}, sr[] = {0, 256}, vr[] = {0, 256};
    const float* ranges[] = {hr, sr, vr};
    int channels[] = {0, 1, 2};
    cv::Mat hist, mroi = mask(bb);
    cv::calcHist(&hsv, 1, channels, mroi, hist, 3, histSize, ranges);
    double hs = cv::sum(hist)[0];
    for (int h = 0; h < 8; ++h)
        for (int s = 0; s < 4; ++s)
            for (int v = 0; v < 4; ++v)
                out.push_back(hs > 0 ? hist.at<float>(h, s, v) / (float)hs : 0.f);

    // 2) HOG on the bbox crop, 64x64 grey (3x3 blocks * 2x2 cells * 9 = 324)
    cv::Mat g;
    cv::resize(rgb(bb), g, {64, 64});
    cv::cvtColor(g, g, cv::COLOR_BGR2GRAY);
    static const cv::HOGDescriptor hog(cv::Size(64, 64), cv::Size(32, 32),
                                       cv::Size(16, 16), cv::Size(16, 16), 9);
    std::vector<float> hogF;
    hog.compute(g, hogF);
    out.insert(out.end(), hogF.begin(), hogF.end());

    // 3) percentile-normalised depth histogram (10) + aspect ratio (1)
    std::vector<float> dv;
    std::vector<cv::Point3f> cloud;
    std::vector<cv::Point> cloudPx;
    for (const auto& p : nz) {
        float d = depthMM.at<float>(p);
        if (d <= 0) continue;
        dv.push_back(d);
        // pixel coordinates are absolute in both datasets -> no offset is added
        // (the first iteration added *_loc.txt again for object frames but not for scene
        // proposals, so training and test point clouds lived in different frames)
        cloud.push_back(backproject((float)p.x, (float)p.y, d));
        cloudPx.push_back(p);
    }
    float dh[10] = {0};
    if (!dv.empty()) {
        std::vector<float> sorted = dv;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&](double q) {
            double pos = q * (sorted.size() - 1);
            size_t lo = (size_t)pos; double f = pos - lo;
            return lo + 1 < sorted.size() ? sorted[lo] * (1 - f) + sorted[lo + 1] * f : sorted[lo];
        };
        float plo = (float)pct(0.05), phi = (float)pct(0.95);
        float rng = std::max(1.f, phi - plo);
        for (float d : dv) {
            float t = std::min(1.f, std::max(0.f, (d - plo) / rng));
            dh[std::min(9, (int)(t * 10))] += 1.f;
        }
        for (auto& v : dh) v /= dv.size();
    }
    out.insert(out.end(), dh, dh + 10);
    out.push_back((float)bb.height / (float)bb.width);

    // 4) mean spin image over sampled oriented points (16x16 = 256)
    std::vector<float> spin = spinDescriptor(cloud, cloudPx, depthMM);
    out.insert(out.end(), spin.begin(), spin.end());
    return (int)out.size() == DESC_DIM;
}

// ============================ feature normalisation ============================
const std::vector<Block> BLOCKS = {{"hsv", 0, 128}, {"hog", 128, 452}, {"depth", 452, 462},
                                   {"aspect", 462, 463}, {"spin", 463, 719}};

static void powerNormalise(cv::Mat& X) {
    for (int r = 0; r < X.rows; ++r) {
        float* p = X.ptr<float>(r);
        for (const auto& b : BLOCKS) {
            if (std::string(b.name) == "aspect") continue;
            for (int k = b.begin; k < b.end; ++k) p[k] = std::sqrt(std::max(0.f, p[k]));
        }
    }
}

void FeatureNorm::fit(const cv::Mat& Xin, float hsvWeight) {
    cv::Mat X = Xin.clone();
    powerNormalise(X);
    mean.create(1, X.cols, CV_32F); stdv.create(1, X.cols, CV_32F); w.create(1, X.cols, CV_32F);
    for (int c = 0; c < X.cols; ++c) {
        cv::Scalar m, s;
        cv::meanStdDev(X.col(c), m, s);
        mean.at<float>(0, c) = (float)m[0];
        stdv.at<float>(0, c) = s[0] > 1e-9 ? (float)s[0] : 1.f;
    }
    for (const auto& b : BLOCKS) {
        float bw = std::string(b.name) == "hsv" ? hsvWeight : 1.f;
        // x2: keeps the whole vector at unit-ish scale for the MLP (4 active blocks)
        float v = 2.f * bw / std::sqrt((float)(b.end - b.begin));
        for (int k = b.begin; k < b.end; ++k) w.at<float>(0, k) = v;
    }
}
cv::Mat FeatureNorm::transform(const cv::Mat& Xin) const {
    cv::Mat X = Xin.clone();
    powerNormalise(X);
    cv::Mat Z(X.size(), CV_32F);
    const float *m = mean.ptr<float>(0), *s = stdv.ptr<float>(0), *wt = w.ptr<float>(0);
    for (int r = 0; r < X.rows; ++r) {
        const float* x = X.ptr<float>(r);
        float* z = Z.ptr<float>(r);
        for (int c = 0; c < X.cols; ++c) z[c] = (x[c] - m[c]) / s[c] * wt[c];
    }
    return Z;
}
void FeatureNorm::write(cv::FileStorage& fs) const { fs << "norm_mean" << mean << "norm_std" << stdv << "norm_w" << w; }
void FeatureNorm::read(const cv::FileStorage& fs)  { fs["norm_mean"] >> mean; fs["norm_std"] >> stdv; fs["norm_w"] >> w; }

// ============================ Platt (Newton) ============================
void Platt::fit(const std::vector<float>& f, const std::vector<int>& y) {
    const int N = (int)f.size();
    double prior1 = 0, prior0 = 0;
    for (int t : y) (t ? prior1 : prior0) += 1;
    if (prior1 == 0 || prior0 == 0) { A = -1; B = 0; return; }
    const double hi = (prior1 + 1) / (prior1 + 2), lo = 1 / (prior0 + 2);
    std::vector<double> t(N);
    for (int i = 0; i < N; ++i) t[i] = y[i] ? hi : lo;
    double a = 0, b = std::log((prior0 + 1) / (prior1 + 1));
    auto fval = [&](double A_, double B_) {
        double s = 0;
        for (int i = 0; i < N; ++i) {
            double fApB = f[i] * A_ + B_;
            s += fApB >= 0 ? t[i] * fApB + std::log1p(std::exp(-fApB))
                           : (t[i] - 1) * fApB + std::log1p(std::exp(fApB));
        }
        return s;
    };
    double fv = fval(a, b);
    for (int it = 0; it < 100; ++it) {
        double h11 = 1e-12, h22 = 1e-12, h21 = 0, g1 = 0, g2 = 0;
        for (int i = 0; i < N; ++i) {
            double fApB = f[i] * a + b, p, q;
            if (fApB >= 0) { p = std::exp(-fApB) / (1 + std::exp(-fApB)); q = 1 / (1 + std::exp(-fApB)); }
            else           { p = 1 / (1 + std::exp(fApB)); q = std::exp(fApB) / (1 + std::exp(fApB)); }
            double d2 = p * q;
            h11 += f[i] * f[i] * d2; h22 += d2; h21 += f[i] * d2;
            double d1 = t[i] - p;
            g1 += f[i] * d1; g2 += d1;
        }
        if (std::fabs(g1) < 1e-5 && std::fabs(g2) < 1e-5) break;
        double det = h11 * h22 - h21 * h21;
        double dA = -(h22 * g1 - h21 * g2) / det, dB = -(-h21 * g1 + h11 * g2) / det;
        double gd = g1 * dA + g2 * dB, step = 1;
        while (step >= 1e-10) {
            double na = a + step * dA, nb = b + step * dB, nf = fval(na, nb);
            if (nf < fv + 1e-4 * step * gd) { a = na; b = nb; fv = nf; break; }
            step /= 2;
        }
        if (step < 1e-10) break;
    }
    A = (float)a; B = (float)b;
}
float Platt::prob(float m) const {
    double fApB = (double)m * A + B;
    return (float)(fApB >= 0 ? std::exp(-fApB) / (1 + std::exp(-fApB)) : 1 / (1 + std::exp(fApB)));
}

// ============================ FastSAM proposals ============================

std::vector<Proposal> runFastSAM(cv::dnn::Net& net, const cv::Mat& bgr, float confTh, float nmsIoU) {
    const int S = 640, H = bgr.rows, W = bgr.cols;
    const float r = std::min((float)S / W, (float)S / H);  // letterbox
    const int nw = (int)std::round(W * r), nh = (int)std::round(H * r);
    const int dw = (S - nw) / 2, dh = (S - nh) / 2;
    cv::Mat resized, canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(bgr, resized, {nw, nh});
    resized.copyTo(canvas(cv::Rect(dw, dh, nw, nh)));
    net.setInput(cv::dnn::blobFromImage(canvas, 1.0 / 255.0, {S, S}, {}, true, false));
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    cv::Mat det, proto;                                     // (1,37,8400) and (1,32,160,160)
    for (auto& o : outs) (o.dims == 4 ? proto : det) = o;
    CV_Assert(!det.empty() && !proto.empty());
    const int nattr = det.size[1], nanch = det.size[2], ncoef = 32, nclass = nattr - 4 - ncoef;
    cv::Mat detM(nattr, nanch, CV_32F, det.ptr<float>());

    std::vector<cv::Rect2d> boxes; std::vector<cv::Rect> iboxes;
    std::vector<float> confs; std::vector<int> anchor;
    for (int a = 0; a < nanch; ++a) {
        float best = 0;
        for (int c = 0; c < nclass; ++c) best = std::max(best, detM.at<float>(4 + c, a));
        if (best < confTh) continue;
        float cx = detM.at<float>(0, a), cy = detM.at<float>(1, a), bw = detM.at<float>(2, a), bh = detM.at<float>(3, a);
        double x1 = std::clamp((cx - bw / 2 - dw) / r, 0.f, (float)W), y1 = std::clamp((cy - bh / 2 - dh) / r, 0.f, (float)H);
        double x2 = std::clamp((cx + bw / 2 - dw) / r, 0.f, (float)W), y2 = std::clamp((cy + bh / 2 - dh) / r, 0.f, (float)H);
        if (x2 - x1 < 2 || y2 - y1 < 2) continue;
        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        iboxes.emplace_back((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
        confs.push_back(best); anchor.push_back(a);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(iboxes, confs, confTh, nmsIoU, keep);

    const int mh = proto.size[2], mw = proto.size[3];
    cv::Mat protoM(ncoef, mh * mw, CV_32F, proto.ptr<float>());
    std::vector<Proposal> props;
    for (int k : keep) {
        cv::Mat cf(1, ncoef, CV_32F);
        for (int c = 0; c < ncoef; ++c) cf.at<float>(c) = detM.at<float>(4 + nclass + c, anchor[k]);
        cv::Mat lin = cf * protoM;
        cv::Mat m = lin.reshape(1, mh), e;
        cv::exp(-m, e);
        m = 1.0 / (1.0 + e);                                // sigmoid
        const float ms = (float)mh / S;                     // proto grid = letterbox / 4
        cv::Rect crop((int)std::round(dw * ms), (int)std::round(dh * ms),
                      (int)std::round(nw * ms), (int)std::round(nh * ms));
        crop &= cv::Rect(0, 0, mw, mh);
        if (crop.area() <= 0) continue;
        cv::Mat mFull;
        cv::resize(m(crop), mFull, {W, H}, 0, 0, cv::INTER_LINEAR);
        cv::Mat mb = mFull > 0.5, boxed = cv::Mat::zeros(H, W, CV_8U);
        cv::Rect bi((int)boxes[k].x, (int)boxes[k].y, (int)std::ceil(boxes[k].width), (int)std::ceil(boxes[k].height));
        bi &= cv::Rect(0, 0, W, H);
        mb(bi).copyTo(boxed(bi));                            // ultralytics crops masks to boxes
        if (cv::countNonZero(boxed) == 0) continue;
        props.push_back({boxes[k], boxed, confs[k]});
    }
    return props;
}

bool goodProposal(const cv::Mat& mask, const cv::Mat& depth, bool rejectBorder) {
    const int H = mask.rows, W = mask.cols;
    int area = cv::countNonZero(mask);
    if (area < 500 || area > 0.30 * H * W) return false;
    cv::Rect2d bb = maskBox(mask);
    // Truncated objects at the image border are part of the ground truth, so they are
    // kept by default (the first iteration always rejected them, capping recall).
    if (rejectBorder && (bb.x <= 1 || bb.y <= 1 || bb.x + bb.width >= W - 1 || bb.y + bb.height >= H - 1))
        return false;
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    double s = 0, s2 = 0; int n = 0;
    for (auto& p : nz) { float d = depth.at<float>(p); if (d > 0) { s += d; s2 += (double)d * d; ++n; } }
    if (n < 50) return false;
    double sd = std::sqrt(std::max(0.0, s2 / n - (s / n) * (s / n)));
    return sd * 0.001 <= 0.5;                              // reject segments spanning > 0.5 m depth spread
}


// ============================ classifier bank ============================
float svmMargin(const cv::Ptr<cv::ml::SVM>& s, const cv::Mat& row) {
    cv::Mat r;
    s->predict(row, r, cv::ml::StatModel::RAW_OUTPUT);
    return r.at<float>(0);                                 // sign absorbed by Platt's A
}
float mlpMargin(const cv::Ptr<cv::ml::ANN_MLP>& m, const cv::Mat& row) {
    cv::Mat r;
    m->predict(row, r);
    return r.at<float>(0, 1) - r.at<float>(0, 0);
}
float ClassifierBank::prob(int c, const cv::Mat& z, bool useSvm) const {
    double p = 0;
    for (int f = 0; f < nFolds; ++f)
        p += useSvm ? calSvm[c][f].prob(svmMargin(svm[c][f], z)) : calMlp[c][f].prob(mlpMargin(mlp[c][f], z));
    return (float)(p / nFolds);
}
void ClassifierBank::save(const std::string& dir, cv::FileStorage& fs) const {
    fs << "n_classes" << nClasses << "n_folds" << nFolds;
    for (int c = 0; c < nClasses; ++c)
        for (int f = 0; f < nFolds; ++f) {
            std::string k = std::to_string(c) + "_" + std::to_string(f);
            svm[c][f]->save(dir + "/svm_" + k + ".yml");
            mlp[c][f]->save(dir + "/mlp_" + k + ".yml");
            fs << ("cal_svm_" + k) << "[" << calSvm[c][f].A << calSvm[c][f].B << "]";
            fs << ("cal_mlp_" + k) << "[" << calMlp[c][f].A << calMlp[c][f].B << "]";
        }
}
bool ClassifierBank::load(const std::string& dir, const cv::FileStorage& fs) {
    if (fs["n_folds"].empty()) return false;
    fs["n_classes"] >> nClasses; fs["n_folds"] >> nFolds;
    svm.assign(nClasses, {}); mlp.assign(nClasses, {});
    calSvm.assign(nClasses, std::vector<Platt>(nFolds)); calMlp.assign(nClasses, std::vector<Platt>(nFolds));
    for (int c = 0; c < nClasses; ++c)
        for (int f = 0; f < nFolds; ++f) {
            std::string k = std::to_string(c) + "_" + std::to_string(f);
            svm[c].push_back(cv::ml::SVM::load(dir + "/svm_" + k + ".yml"));
            mlp[c].push_back(cv::ml::ANN_MLP::load(dir + "/mlp_" + k + ".yml"));
            std::vector<float> a, b;
            fs["cal_svm_" + k] >> a; fs["cal_mlp_" + k] >> b;
            if (a.size() < 2 || b.size() < 2 || svm[c].back().empty() || mlp[c].back().empty()) return false;
            calSvm[c][f].A = a[0]; calSvm[c][f].B = a[1];
            calMlp[c][f].A = b[0]; calMlp[c][f].B = b[1];
        }
    return true;
}

// ============================ evaluation ============================
std::vector<PRPoint> sweepClassification(const std::vector<float>& scores,
                                         const std::vector<int>& y,
                                         const std::vector<float>& taus) {
    int npos = 0, nneg = 0;
    for (int t : y) (t ? ++npos : ++nneg);
    std::vector<PRPoint> out;
    for (float tau : taus) {
        int tp = 0, fp = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if (scores[i] >= tau) (y[i] ? ++tp : ++fp);
        PRPoint p;
        p.tau = tau; p.tp = tp; p.fp = fp; p.fn = npos - tp;
        p.precision = (tp + fp) ? (float)tp / (tp + fp) : 1.f;   // no predictions -> P := 1
        p.recall    = npos ? (float)tp / npos : 0.f;
        p.fpr       = nneg ? (float)fp / nneg : 0.f;
        out.push_back(p);
    }
    return out;
}

double averagePrecision(const std::vector<float>& scores, const std::vector<int>& y) {
    std::vector<size_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    int npos = 0; for (int t : y) npos += t;
    if (!npos) return 0.0;
    double ap = 0, tp = 0, fp = 0, prevRec = 0;
    for (size_t i = 0; i < idx.size();) {                  // tied scores = one threshold
        size_t j = i;
        while (j < idx.size() && scores[idx[j]] == scores[idx[i]]) { (y[idx[j]] ? ++tp : ++fp); ++j; }
        double rec = tp / npos, prec = tp / (tp + fp);
        ap += (rec - prevRec) * prec;
        prevRec = rec;
        i = j;
    }
    return ap;
}

double rocAUC(const std::vector<float>& s, const std::vector<int>& y) {
    std::vector<size_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return s[a] < s[b]; });
    double rankSum = 0; long np = 0, nn = 0;
    for (size_t i = 0; i < idx.size();) {
        size_t j = i;
        while (j < idx.size() && s[idx[j]] == s[idx[i]]) ++j;
        double r = (i + 1 + j) / 2.0;                       // average rank of the tie group
        for (size_t k = i; k < j; ++k) if (y[idx[k]]) rankSum += r;
        i = j;
    }
    for (int t : y) (t ? ++np : ++nn);
    if (!np || !nn) return 0.5;
    return (rankSum - np * (np + 1) / 2.0) / ((double)np * nn);
}

// ============================ plotting ============================
cv::Scalar classColor(int i) {
    static const cv::Scalar pal[] = {                      // BGR, tab10-like
        {180, 119, 31}, {14, 127, 255}, {44, 160, 44}, {40, 39, 214}, {189, 103, 148},
        {75, 86, 140}, {194, 119, 227}, {127, 127, 127}, {34, 189, 188}, {207, 190, 23}};
    return pal[((i % 10) + 10) % 10];
}

static cv::Scalar lighten(const cv::Scalar& c, double t) {
    return {c[0] + (255 - c[0]) * t, c[1] + (255 - c[1]) * t, c[2] + (255 - c[2]) * t};
}

static void text(cv::Mat& img, const std::string& s, cv::Point p, double sc = 0.5,
                 cv::Scalar col = {20, 20, 20}, int th = 1) {
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, sc, col, th, cv::LINE_AA);
}

// draws axes for a [0,1]^2 panel; returns data->pixel mapper
static std::function<cv::Point(cv::Point2f)> unitAxes(cv::Mat& img, cv::Rect area,
                                                      const std::string& xl, const std::string& yl) {
    auto mp = [area](cv::Point2f p) {
        return cv::Point(area.x + (int)std::round(std::clamp(p.x, 0.f, 1.f) * area.width),
                         area.y + area.height - (int)std::round(std::clamp(p.y, 0.f, 1.f) * area.height));
    };
    for (int i = 0; i <= 5; ++i) {
        float v = i / 5.f;
        cv::line(img, mp({v, 0}), mp({v, 1}), {228, 228, 228}, 1);
        cv::line(img, mp({0, v}), mp({1, v}), {228, 228, 228}, 1);
        char buf[8]; snprintf(buf, 8, "%.1f", v);
        text(img, buf, {mp({v, 0}).x - 12, area.y + area.height + 18}, 0.42, {80, 80, 80});
        text(img, buf, {area.x - 32, mp({0, v}).y + 5}, 0.42, {80, 80, 80});
    }
    cv::rectangle(img, mp({0, 1}), mp({1, 0}), {90, 90, 90}, 1);
    text(img, xl, {area.x + area.width / 2 - 25, area.y + area.height + 38}, 0.5);
    text(img, yl, {area.x - 34, area.y - 8}, 0.5);
    return mp;
}

void plotXY(const std::string& path, const std::string& title,
            const std::string& xlab, const std::string& ylab,
            const std::vector<Series>& series, bool diagonal) {
    const int W = 1080, H = 680;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Rect area(80, 70, W - 80 - 380, H - 70 - 70);
    auto mp = unitAxes(img, area, xlab, ylab);
    text(img, title, {80, 32}, 0.7, {20, 20, 20}, 2);
    if (diagonal) cv::line(img, mp({0, 0}), mp({1, 1}), {200, 200, 200}, 1, cv::LINE_AA);
    for (size_t si = 0; si < series.size(); ++si) {
        cv::Scalar col = classColor((int)si);
        const auto& s = series[si];
        for (size_t i = 1; i < s.pts.size(); ++i)
            cv::line(img, mp(s.pts[i - 1]), mp(s.pts[i]), col, 2, cv::LINE_AA);
        for (size_t i = 0; i < s.markers.size(); ++i) {
            cv::circle(img, mp(s.markers[i]), 6, col, -1, cv::LINE_AA);
            cv::circle(img, mp(s.markers[i]), 6, {0, 0, 0}, 1, cv::LINE_AA);
        }
        int ly = area.y + 15 + (int)si * 26;
        cv::line(img, {W - 360, ly}, {W - 325, ly}, col, 3, cv::LINE_AA);
        text(img, s.label, {W - 318, ly + 5}, 0.48);
    }
    cv::imwrite(path, img);
}

// marker shape per tau index: circle, square, triangle
static void tauMarker(cv::Mat& img, cv::Point q, int i, const cv::Scalar& col, bool hollow = false) {
    const int r = 7;
    int fill = hollow ? 2 : -1;
    if (i == 0) {
        cv::circle(img, q, r, col, fill, cv::LINE_AA);
        cv::circle(img, q, r, {0, 0, 0}, 1, cv::LINE_AA);
    } else if (i == 1) {
        cv::rectangle(img, {q.x - r + 1, q.y - r + 1}, {q.x + r - 1, q.y + r - 1}, col, fill, cv::LINE_AA);
        cv::rectangle(img, {q.x - r + 1, q.y - r + 1}, {q.x + r - 1, q.y + r - 1}, {0, 0, 0}, 1, cv::LINE_AA);
    } else {
        std::vector<cv::Point> tri = {{q.x, q.y - r - 1}, {q.x - r, q.y + r - 2}, {q.x + r, q.y + r - 2}};
        if (hollow) cv::polylines(img, tri, true, col, 2, cv::LINE_AA);
        else cv::fillConvexPoly(img, tri, col, cv::LINE_AA);
        cv::polylines(img, tri, true, {0, 0, 0}, 1, cv::LINE_AA);
    }
}

void plotTauPR(const std::string& path, const std::string& title, const std::vector<Series>& perClass) {
    const int PW = 330, PH = 290, ML = 70, MT = 80, GX = 80, GY = 165;
    const int cols = 3, rows = (int)((perClass.size() + 1 + cols - 1) / cols);
    const int W = ML + cols * (PW + GX), H = MT + rows * (PH + GY);
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    text(img, title, {ML - 40, 38}, 0.75, {20, 20, 20}, 2);
    for (size_t ci = 0; ci < perClass.size(); ++ci) {
        int r = (int)ci / cols, c = (int)ci % cols;
        cv::Rect area(ML + c * (PW + GX), MT + r * (PH + GY) + 20, PW, PH);
        auto mp = unitAxes(img, area, "Recall", "Precision");
        const auto& s = perClass[ci];
        cv::Scalar col = classColor((int)ci);
        text(img, s.label, {area.x, area.y - 26}, 0.5, col, 2);
        for (size_t i = 1; i < s.pts.size(); ++i)          // faint threshold sweep
            cv::line(img, mp(s.pts[i - 1]), mp(s.pts[i]), lighten(col, 0.65), 1, cv::LINE_AA);
        for (size_t i = 1; i < s.markers.size(); ++i)      // the 3-tau curve
            cv::line(img, mp(s.markers[i - 1]), mp(s.markers[i]), col, 3, cv::LINE_AA);
        for (size_t i = 0; i < s.markers.size(); ++i) {
            bool none = i < s.markerLabels.size() && s.markerLabels[i].find("n/a") != std::string::npos;
            tauMarker(img, mp(s.markers[i]), (int)i, col, none);
        }
        // value table under the panel
        for (size_t i = 0; i < s.markerLabels.size(); ++i) {
            cv::Point o(area.x + 14, area.y + area.height + 62 + (int)i * 22);
            tauMarker(img, {o.x - 4, o.y - 5}, (int)i, col);
            text(img, s.markerLabels[i], {o.x + 10, o.y}, 0.48, {40, 40, 40});
        }
    }
    int k = (int)perClass.size();
    cv::Point o(ML + (k % cols) * (PW + GX), MT + (k / cols) * (PH + GY) + 50);
    text(img, "Bold line: precision/recall at", o, 0.55);
    text(img, "tau = 0.3 (circle), 0.5 (square),", {o.x, o.y + 26}, 0.55);
    text(img, "0.7 (triangle)", {o.x, o.y + 52}, 0.55);
    text(img, "Faint line: same, tau = 0.01 ... 0.99", {o.x, o.y + 90}, 0.55, {120, 120, 120});
    text(img, "Hollow marker: no detection at that", {o.x, o.y + 128}, 0.55);
    text(img, "tau (precision undefined, drawn as 1)", {o.x, o.y + 154}, 0.55);
    text(img, "Raising tau: fewer, more reliable hits", {o.x, o.y + 192}, 0.55);
    cv::imwrite(path, img);
}

void plotTauBars(const std::string& path, const std::string& title,
                 const std::vector<std::string>& classes,
                 const std::vector<double>& precision,
                 const std::vector<double>& recall) {
    const int n = (int)classes.size();
    const int L = 90, R = 40, T = 70, B = 110;
    const int groupW = 130, pw = groupW * n, ph = 420;
    const int W = L + pw + R, H = T + ph + B;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    auto mapY = [&](double v) { return T + ph - (int)std::round(std::clamp(v, 0.0, 1.0) * ph); };
    for (int i = 0; i <= 5; ++i) {
        double v = i / 5.0;
        cv::line(img, {L, mapY(v)}, {L + pw, mapY(v)}, {225, 225, 225}, 1);
        char buf[8]; snprintf(buf, 8, "%.1f", v);
        text(img, buf, {L - 45, mapY(v) + 5}, 0.5, {60, 60, 60});
    }
    cv::rectangle(img, {L, T}, {L + pw, T + ph}, {80, 80, 80}, 1);
    const cv::Scalar colP(180, 119, 31), colR(14, 127, 255);
    const int barW = 38, gap = 8;
    for (int ci = 0; ci < n; ++ci) {
        int cx = L + ci * groupW + groupW / 2;
        int xP = cx - barW - gap / 2, xR = cx + gap / 2;
        cv::rectangle(img, {xP, mapY(precision[ci])}, {xP + barW, mapY(0)}, colP, -1);
        cv::rectangle(img, {xR, mapY(recall[ci])},    {xR + barW, mapY(0)}, colR, -1);
        char bp[8], br[8];
        snprintf(bp, 8, "%.2f", precision[ci]);
        snprintf(br, 8, "%.2f", recall[ci]);
        text(img, bp, {xP - 4, mapY(precision[ci]) - 6}, 0.45);
        text(img, br, {xR - 4, mapY(recall[ci]) - 6}, 0.45);
        text(img, classes[ci], {cx - (int)classes[ci].size() * 5, T + ph + 30}, 0.5);
    }
    cv::rectangle(img, {L, H - 55}, {L + 22, H - 38}, colP, -1);
    text(img, "Precision", {L + 30, H - 41}, 0.55);
    cv::rectangle(img, {L + 160, H - 55}, {L + 182, H - 38}, colR, -1);
    text(img, "Recall", {L + 190, H - 41}, 0.55);
    text(img, title, {L, 35}, 0.75, {20, 20, 20}, 2);
    cv::imwrite(path, img);
}

static void frameAxis(cv::Mat& img, int L, int T, int pw, int ph, int fmin, int fmax,
                      const std::function<int(int)>& mapX) {
    for (int i = 0; i <= 8; ++i) {
        int f = fmin + (int)std::round(i / 8.0 * (fmax - fmin));
        cv::line(img, {mapX(f), T}, {mapX(f), T + ph}, {235, 235, 235}, 1);
        text(img, std::to_string(f), {mapX(f) - 12, T + ph + 22}, 0.45, {60, 60, 60});
    }
    cv::rectangle(img, {L, T}, {L + pw, T + ph}, {80, 80, 80}, 1);
}

void plotTimeline(const std::string& path, const std::string& title,
                  const std::vector<TimelinePoint>& pts, const std::vector<int>& frames) {
    if (frames.empty()) return;
    int fmin = frames.front(), fmax = frames.back(), idMax = 0;
    for (auto& p : pts) idMax = std::max(idMax, p.id);
    const int L = 80, R = 40, T = 90, B = 70, pw = 1100;
    const int rowH = std::max(6, std::min(34, 520 / std::max(1, idMax + 1)));
    const int ph = rowH * (idMax + 1);
    cv::Mat img(T + ph + B, L + pw + R, CV_8UC3, cv::Scalar(255, 255, 255));
    std::function<int(int)> mapX = [&](int f) {
        return L + (fmax > fmin ? (int)std::round((double)(f - fmin) / (fmax - fmin) * pw) : pw / 2);
    };
    auto mapY = [&](int id) { return T + ph - rowH / 2 - id * rowH; };
    for (int id = 0; id <= idMax; ++id) {
        cv::line(img, {L, mapY(id)}, {L + pw, mapY(id)}, {242, 242, 242}, 1);
        int every = std::max(1, (idMax + 1) / 25);
        if (id % every == 0) text(img, std::to_string(id), {L - 35, mapY(id) + 4}, 0.42, {60, 60, 60});
    }
    frameAxis(img, L, T, pw, ph, fmin, fmax, mapX);
    std::map<int, size_t> pos;                              // frame -> processed index
    for (size_t i = 0; i < frames.size(); ++i) pos[frames[i]] = i;
    std::map<int, std::vector<std::pair<int, int>>> byId;   // id -> (frame, cls)
    for (auto& p : pts) byId[p.id].push_back({p.frame, p.cls});
    for (auto& [id, v] : byId) {
        std::sort(v.begin(), v.end());
        for (size_t i = 1; i < v.size(); ++i)               // join consecutive processed frames
            if (pos[v[i].first] == pos[v[i - 1].first] + 1)
                cv::line(img, {mapX(v[i - 1].first), mapY(id)}, {mapX(v[i].first), mapY(id)},
                         classColor(v[i].second), std::max(2, rowH / 3), cv::LINE_AA);
        for (auto& [f, c] : v)
            cv::circle(img, {mapX(f), mapY(id)}, std::max(2, std::min(5, rowH / 3)), classColor(c), -1, cv::LINE_AA);
    }
    text(img, title, {L, 30}, 0.7, {20, 20, 20}, 2);
    int lx = L;
    for (size_t c = 0; c < CLASSES.size(); ++c) {
        cv::circle(img, {lx + 8, 58}, 6, classColor((int)c), -1, cv::LINE_AA);
        text(img, CLASSES[c], {lx + 20, 63});
        lx += 22 + 11 * (int)CLASSES[c].size() + 30;
    }
    text(img, "Frame", {L + pw / 2 - 20, T + ph + 50}, 0.55);
    text(img, "ID", {L - 35, T - 10}, 0.55);
    cv::imwrite(path, img);
}

void plotPresence(const std::string& path, const std::string& title,
                  const std::map<int, std::vector<int>>& clsPerFrame, const std::vector<int>& frames) {
    if (frames.empty()) return;
    int fmin = frames.front(), fmax = frames.back();
    const int L = 120, R = 40, T = 70, B = 60, pw = 1060, rowH = 34;
    const int ph = rowH * (int)CLASSES.size();
    cv::Mat img(T + ph + B, L + pw + R, CV_8UC3, cv::Scalar(255, 255, 255));
    std::function<int(int)> mapX = [&](int f) {
        return L + (fmax > fmin ? (int)std::round((double)(f - fmin) / (fmax - fmin) * pw) : pw / 2);
    };
    frameAxis(img, L, T, pw, ph, fmin, fmax, mapX);
    for (size_t c = 0; c < CLASSES.size(); ++c) {
        int y = T + (int)c * rowH + rowH / 2;
        text(img, CLASSES[c], {8, y + 5}, 0.5, classColor((int)c), 1);
        for (size_t i = 0; i < frames.size(); ++i) {
            auto it = clsPerFrame.find(frames[i]);
            if (it == clsPerFrame.end()) continue;
            int n = (int)std::count(it->second.begin(), it->second.end(), (int)c);
            if (!n) continue;
            int x0 = mapX(frames[i]), x1 = i + 1 < frames.size() ? mapX(frames[i + 1]) : x0 + 4;
            cv::rectangle(img, {x0, y - rowH / 3}, {std::max(x0 + 2, x1 - 1), y + rowH / 3},
                          n > 1 ? classColor((int)c) : lighten(classColor((int)c), 0.35), -1);
        }
    }
    text(img, title, {L, 32}, 0.7, {20, 20, 20}, 2);
    text(img, "Frame   (darker = 2+ instances)", {L + pw / 2 - 120, T + ph + 48}, 0.5);
    cv::imwrite(path, img);
}

} // namespace rgbd
