// RGB-D recognition and tracking - common declarations
// RGB-D object recognition & tracking, Washington RGB-D Object + Scenes datasets.
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <array>
#include <climits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace rgbd {

// ---------------- constants ----------------
extern const std::vector<std::string> CLASSES;            // 5 object classes
constexpr float FX = 570.3f, FY = 570.3f;                  // Kinect intrinsics (UW depthToCloud)
constexpr float CX0 = 320.0f, CY0 = 240.0f;
constexpr int   DESC_DIM = 719;                            // 128 + 324 + 10 + 1 + 256
extern const std::vector<float> REPORT_TAUS;               // {0.3, 0.5, 0.7}
std::vector<float> denseTaus();                            // 0.01 .. 0.99

// ---------------- small CLI helper ----------------
std::string argval(int argc, char** argv, const std::string& key, const std::string& dflt);
bool hasFlag(int argc, char** argv, const std::string& key);

// ---------------- file IO ----------------
// Full-frame (640x480) object-dataset sample. Pixel coordinates in rgb/depth/mask
// are ABSOLUTE image coordinates; loc is only the top-left of the object's crop
// window and is used to place the segmentation ROI - never added to coordinates.
struct ObjectFrame {
    cv::Mat rgb;      // CV_8UC3 BGR
    cv::Mat depth;    // CV_32F, millimetres (0 = invalid)
    cv::Mat maskGT;   // CV_8U, 0/255 (provided dataset mask) - evaluation only
    cv::Point loc;    // crop top-left in the full frame (from *_loc.txt)
};

std::vector<std::string> listDirsNatural(const std::string& dir, const std::string& prefix);
std::vector<std::string> listRgbFramesNatural(const std::string& dir);  // *.png minus _depth/_mask
bool readObjectFrame(const std::string& rgbPath, ObjectFrame& out);
bool readSceneFrame(const std::string& rgbPath, cv::Mat& rgb, cv::Mat& depth);
cv::Point readLoc(const std::string& path);
int frameIndexFromPath(const std::string& path);           // ..._<k>.png -> k

// ---------------- geometry / metrics ----------------
double maskIoU(const cv::Mat& a, const cv::Mat& b);        // CV_8U masks
double boxIoU(const cv::Rect2d& a, const cv::Rect2d& b);
cv::Rect2d maskBox(const cv::Mat& mask);                   // tight box, width = xmax-xmin+1
// back-projection with the Kinect pinhole model (depth in mm -> metres)
inline cv::Point3f backproject(float u, float v, float depthMM) {
    const float Z = depthMM * 0.001f;
    return {(u - CX0) * Z / FX, (v - CY0) * Z / FY, Z};
}
// greedy NMS: indices of kept boxes, highest score first. A box is suppressed by a
// higher-scoring one if IoU > iouThr or, when containThr < 1, if more than containThr
// of its own area lies inside it (nested masks: mug handle inside mug, ...)
std::vector<int> nmsGreedy(const std::vector<cv::Rect2d>& boxes,
                           const std::vector<float>& scores, double iouThr,
                           double containThr = 1.0);

// ---------------- segmentation (object dataset) ----------------
// ROI where the object is searched: window of size `win` starting at loc.
cv::Rect objectROI(cv::Point loc, cv::Size imgSize, cv::Size win = cv::Size(280, 300));

// A: depth window + Otsu (threshold computed on ROI pixels only) + nearest blob.
cv::Mat segmentDepth(const cv::Mat& depthMM, const cv::Rect& roi, int minArea = 200);

// B: RANSAC support-plane removal. Points more than `minH` and less than `maxH`
// above the fitted table/turntable plane are kept, depth discontinuities are cut,
// and the connected component closest to the ROI centre is returned.
struct PlaneSegParams {
    float minH = 0.010f;          // m above the plane (turntable/noise margin)
    float maxH = 0.50f;           // m above the plane (rejects the wall behind the table)
    float inlierThr = 0.008f;     // m, RANSAC inlier band
    float edgeJump = 0.025f;      // m, neighbour depth jump treated as an object boundary
    int   iters = 300;
    int   minArea = 150;
};
cv::Mat segmentPlane(const cv::Mat& depthMM, const cv::Rect& roi,
                     const PlaneSegParams& p = PlaneSegParams());

// fraction of the mask outline lying on the ROI frame (0 = object fully inside)
double borderContact(const cv::Mat& mask, const cv::Rect& roi);
// Pipeline segmenter used for descriptors: runs the plane and the depth segmenter and
// picks the plausible one without ground truth. `usedFallback` = depth was chosen.
cv::Mat segmentObject(const cv::Mat& depthMM, cv::Point loc, bool* usedFallback = nullptr);

// ---------------- 3D geometry of a segment ----------------
struct PlaneModel { cv::Vec3f n; float d = 0; bool ok = false; };   // height = n.p + d (m)
// dominant support plane (table) of a whole frame, same RANSAC as the object segmenter
PlaneModel fitFramePlane(const cv::Mat& depthMM);
// robust physical size of a segment: extents (m) along its principal axes, sorted
// e1 >= e2 >= e3 (5th-95th percentile spread), and the 5th/95th percentile height of
// its points above the support plane (if given)
struct SegmentGeometry { float e1 = 0, e2 = 0, e3 = 0, hLow = 0, hHigh = 0; int nPts = 0; };
SegmentGeometry segmentGeometry(const cv::Mat& mask, const cv::Mat& depthMM, const PlaneModel* plane = nullptr);
// Physical "tabletop object" prior for scene proposals (loose, class independent):
// largest extent 6-40 cm, not flat (thinnest extent >= 1.5 cm), top 4-40 cm above the
// table, lowest points within 15 cm of the table. Rejects table patches, walls, cabinet
// fronts and small fragments - the bulk of FastSAM's segments in a kitchen scene.
struct GeometryGate {
    float e1Min = 0.06f, e1Max = 0.40f, e3Min = 0.015f, topMin = 0.04f, topMax = 0.40f, bottomMax = 0.15f;
    bool pass(const SegmentGeometry& g) const {
        return g.e1 >= e1Min && g.e1 <= e1Max && g.e3 >= e3Min && g.hHigh >= topMin && g.hHigh <= topMax &&
               g.hLow <= bottomMax;
    }
};

// training augmentation: dilate/erode + 5% pixel dropout
cv::Mat perturbMask(const cv::Mat& mask, cv::RNG& rng);

// ---------------- descriptor (719-D) ----------------
// HSV hist(128) + HOG 64x64(324) + depth hist(10) + aspect(1) + mean spin image(256).
// Works identically for object frames and scene proposals (absolute pixel coords).
bool describe(const cv::Mat& rgb, const cv::Mat& depthMM, const cv::Mat& mask,
              std::vector<float>& out);
// surface normals from back-projected 3D neighbours, oriented towards the camera
cv::Point3f normalAt(const cv::Mat& depthMM, int u, int v, int step = 2);

// ---------------- feature normalisation ----------------
// Descriptor block layout (719-D)
struct Block { const char* name; int begin, end; };
extern const std::vector<Block> BLOCKS;                    // hsv, hog, depth, aspect, spin
// 1) square root of every histogram block (Hellinger / power normalisation),
// 2) per-dimension standardisation,
// 3) each block scaled by weight / sqrt(block size), so every block has the same total
//    influence regardless of its length. The colour block gets --hsv-weight
//    (default 0: colour does not generalise across instances of a category, which the
//    leave-one-instance-out cross-validation confirmed).
struct FeatureNorm {
    cv::Mat mean, stdv, w;                                 // 1 x D, CV_32F
    void fit(const cv::Mat& X, float hsvWeight);
    cv::Mat transform(const cv::Mat& X) const;
    void write(cv::FileStorage& fs) const;
    void read(const cv::FileStorage& fs);
};

// ---------------- Platt sigmoid calibration ----------------
// p = 1 / (1 + exp(A*f + B)), fitted by Newton's method with Platt's smoothed
// targets (Lin, Lin & Weng 2007). Fit on OUT-OF-FOLD margins (see train_eval).
struct Platt {
    float A = -1.f, B = 0.f;
    void fit(const std::vector<float>& margins, const std::vector<int>& y);
    float prob(float margin) const;
};

// ---------------- FastSAM proposals (ONNX through cv::dnn) ----------------
struct Proposal { cv::Rect2d box; cv::Mat mask; float conf; };
// YOLOv8-seg decode: letterbox 640, detection head + 32 prototype masks, NMS
std::vector<Proposal> runFastSAM(cv::dnn::Net& net, const cv::Mat& bgr, float confTh, float nmsIoU);
// geometric sanity filter: area, valid depth, depth spread (optionally border contact)
bool goodProposal(const cv::Mat& mask, const cv::Mat& depth, bool rejectBorder = false);

// ---------------- per-class binary classifiers ----------------
// For every class and every cross-validation fold (= one held-out training instance)
// a linear SVM and an MLP are trained on the other instances; each gets its own Platt
// calibration fitted on the held-out instance. The probability of a sample is the
// mean of the calibrated fold probabilities (cross-validated ensemble, as in
// sklearn's CalibratedClassifierCV). Calibration is therefore always fitted on data
// the model has not seen, which keeps the meaning of tau stable on new instances.
float svmMargin(const cv::Ptr<cv::ml::SVM>& s, const cv::Mat& row);
float mlpMargin(const cv::Ptr<cv::ml::ANN_MLP>& m, const cv::Mat& row);
struct ClassifierBank {
    int nClasses = 0, nFolds = 0;
    std::vector<std::vector<cv::Ptr<cv::ml::SVM>>> svm;    // [class][fold]
    std::vector<std::vector<cv::Ptr<cv::ml::ANN_MLP>>> mlp;
    std::vector<std::vector<Platt>> calSvm, calMlp;
    float prob(int cls, const cv::Mat& zRow, bool useSvm) const;
    void save(const std::string& dir, cv::FileStorage& fs) const;
    bool load(const std::string& dir, const cv::FileStorage& fs);
};

// ---------------- threshold-sweep evaluation ----------------
struct PRPoint { float tau, precision, recall, fpr; int tp, fp, fn; };
std::vector<PRPoint> sweepClassification(const std::vector<float>& scores,
                                         const std::vector<int>& y,
                                         const std::vector<float>& taus);
// rank-based average precision (= sklearn average_precision_score, step-wise)
double averagePrecision(const std::vector<float>& scores, const std::vector<int>& y);
// area under the ROC curve (Mann-Whitney, ties counted 1/2)
double rocAUC(const std::vector<float>& scores, const std::vector<int>& y);

// ---------------- plotting (OpenCV-rendered PNGs, no matplotlib) ----------------
cv::Scalar classColor(int idx);                            // tab10-like palette (BGR)
struct Series {
    std::string label;
    std::vector<cv::Point2f> pts;                          // x,y in data coords (faint line)
    std::vector<cv::Point2f> markers;                      // tau operating points
    std::vector<std::string> markerLabels;
};
// generic [0,1]x[0,1] plot (used for ROC)
void plotXY(const std::string& path, const std::string& title,
            const std::string& xlab, const std::string& ylab,
            const std::vector<Series>& series, bool diagonal = false);
// assignment figure: one panel per class, recall (x) vs precision (y) at the three
// tau values connected and labelled, threshold sweep shown faintly for context
void plotTauPR(const std::string& path, const std::string& title,
               const std::vector<Series>& perClass);
// one PNG per tau: grouped bars, per class one Precision + one Recall bar
void plotTauBars(const std::string& path, const std::string& title,
                 const std::vector<std::string>& classes,
                 const std::vector<double>& precision,
                 const std::vector<double>& recall);
// timeline: x = frame, y = object id, colour = class; segments join consecutive
// processed frames in which the object is visible
struct TimelinePoint { int frame, id, cls; };
void plotTimeline(const std::string& path, const std::string& title,
                  const std::vector<TimelinePoint>& pts, const std::vector<int>& frames);
// ground-truth class presence per frame (reference for the tracking timeline)
void plotPresence(const std::string& path, const std::string& title,
                  const std::map<int, std::vector<int>>& clsPerFrame,
                  const std::vector<int>& frames);

} // namespace rgbd
