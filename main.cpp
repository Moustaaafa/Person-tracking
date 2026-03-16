// =============================================================
// YOLOv8-SEG (ONNX) person detection + instance masks
// YOLOv8-POSE (ONNX) per-person keypoints (run on crop)
// OSNet (ONNX) ReID embeddings
// Tracker (IoU + ReID) + Long-term ReID gallery memory (reuses IDs after long gaps)
//
// Build:
//   g++ main.cpp -O2 -std=c++17 `pkg-config --cflags --libs opencv4` -lonnxruntime -o app
//
// Notes:
// - YOLOv8 ONNX outputs can vary depending on export settings.
//   This code supports common YOLOv8-seg layouts:
//     pred:  [1, C, N] where C = 4+nc+nm  (no obj)  OR  5+nc+nm (with obj)
//     proto: [1, nm, mh, mw]
//   It assumes COCO nc=80 and person class id = 0.
//
// - Pose ONNX output also varies. We infer kp_start as 5 or 6 depending on C.
// =============================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ------------------------------ Detection struct ------------------------------
struct Det {
    cv::Rect box;
    float conf = 0.0f;
    std::vector<float> emb;      // L2-normalized ReID appearance embedding for tracking across frames

    // Segmentation (full-frame binary mask 0/255)
    cv::Mat mask_u8;             // CV_8U (same size as frame), may be empty

    // Pose (COCO-17)
    std::vector<cv::Point2f> kps; // size 17
    std::vector<float> kp_conf;   // size 17
};

// ------------------------------ Utils ------------------------------
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static float iou(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? (float)inter / (float)uni : 0.0f;
}

static float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 1.0f;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); i++) dot += (double)a[i] * (double)b[i];
    return (float)(1.0 - dot);
}

static std::vector<int> argsort_desc(const std::vector<float>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return v[i] > v[j]; });
    return idx;
}

static std::vector<Det> nms(const std::vector<Det>& dets, float iouThresh) {
    std::vector<Det> out;
    if (dets.empty()) return out;

    std::vector<float> scores(dets.size());
    for (size_t i = 0; i < dets.size(); i++) scores[i] = dets[i].conf;

    auto order = argsort_desc(scores);
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t _i = 0; _i < order.size(); _i++) {
        int i = order[_i];
        if (suppressed[i]) continue;
        out.push_back(dets[i]);

        for (size_t _j = _i + 1; _j < order.size(); _j++) {
            int j = order[_j];
            if (suppressed[j]) continue;
            if (iou(dets[i].box, dets[j].box) > iouThresh) suppressed[j] = true;
        }
    }
    return out;
}

static void print_shape(const Ort::Value& v, const std::string& name) {
    auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << name << " shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cerr << shape[i] << (i + 1 < shape.size() ? "," : "");
    }
    std::cerr << "]\n";
}

// ------------------------------ ReID (OSNet) ------------------------------
struct ReIDExtractor {
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    const int inH = 256;
    const int inW = 128;

    ReIDExtractor(Ort::Env& env, const std::string& modelPath, const Ort::SessionOptions& opt)
        : session(env, modelPath.c_str(), opt) {}

    static void l2_normalize(std::vector<float>& v) {
        double s = 0.0;
        for (float x : v) s += (double)x * x;
        s = std::sqrt(s) + 1e-12;
        for (float& x : v) x = (float)(x / s);
    }

    std::vector<float> extract(const cv::Mat& frameBGR, const cv::Rect& box) {
        cv::Rect r = box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
        if (r.width <= 1 || r.height <= 1) return {};

        cv::Mat crop = frameBGR(r).clone();
        cv::resize(crop, crop, cv::Size(inW, inH));
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        crop.convertTo(crop, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> ch(3);
        cv::split(crop, ch);
        ch[0] = (ch[0] - 0.485f) / 0.229f;
        ch[1] = (ch[1] - 0.456f) / 0.224f;
        ch[2] = (ch[2] - 0.406f) / 0.225f;

        std::vector<float> input(1 * 3 * inH * inW);
        for (int c = 0; c < 3; c++) {
            std::memcpy(input.data() + c * inH * inW, ch[c].data, (size_t)inH * inW * sizeof(float));
        }

        auto inNameA  = session.GetInputNameAllocated(0, allocator);
        auto outNameA = session.GetOutputNameAllocated(0, allocator);
        const char* inName  = inNameA.get();
        const char* outName = outNameA.get();

        std::vector<int64_t> shape = {1, 3, inH, inW};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value x = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                       shape.data(), shape.size());

        const char* inNames[]  = {inName};
        const char* outNames[] = {outName};

        auto y = session.Run(Ort::RunOptions{nullptr}, inNames, &x, 1, outNames, 1);

        float* embPtr = y[0].GetTensorMutableData<float>();
        auto embShape = y[0].GetTensorTypeAndShapeInfo().GetShape();

        int dim = 1;
        for (auto v : embShape) if (v > 0) dim *= (int)v;

        std::vector<float> emb(embPtr, embPtr + dim);
        l2_normalize(emb);
        return emb;
    }
};

// ------------------------------ YOLOv8-SEG person detector ------------------------------
static std::vector<Det> detectPersonsYOLOv8Seg(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const cv::Mat& frameBGR,
    int inW, int inH,
    float confThresh, float iouThresh,
    bool debug_shapes = false
    ) {


    // Preprocess
    cv::Mat resized;
    cv::resize(frameBGR, resized, cv::Size(inW, inH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensor(1 * 3 * inH * inW);
    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; c++)
        chw[c] = cv::Mat(inH, inW, CV_32F, inputTensor.data() + c * inH * inW);
    cv::split(resized, chw);


    auto inNameA = session.GetInputNameAllocated(0, allocator);
    const char* inName = inNameA.get();

    // Seg models usually have 2 outputs
    auto out0A = session.GetOutputNameAllocated(0, allocator);
    auto out1A = session.GetOutputNameAllocated(1, allocator);
    const char* out0 = out0A.get();
    const char* out1 = out1A.get();

    std::vector<int64_t> inShape = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memInfo, inputTensor.data(), inputTensor.size(),
                                                   inShape.data(), inShape.size());

    const char* inNames[] = {inName};
    const char* outNames[] = {out0, out1};

    auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &x, 1, outNames, 2);

    if (debug_shapes) { print_shape(outs[0], "seg_out0"); print_shape(outs[1], "seg_out1"); }

    // Identify pred/proto by rank
    int pred_idx = 0, proto_idx = 1;
    {
        auto s0 = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto s1 = outs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (s0.size() == 4 && s1.size() == 3) { pred_idx = 1; proto_idx = 0; }
        else if (s0.size() == 3 && s1.size() == 4) { pred_idx = 0; proto_idx = 1; }
    }

    auto predShape = outs[pred_idx].GetTensorTypeAndShapeInfo().GetShape(); // [1,C,N]
    auto protoShape = outs[proto_idx].GetTensorTypeAndShapeInfo().GetShape(); // [1,nm,mh,mw]
    if (predShape.size() != 3 || protoShape.size() != 4) {
        std::cerr << "Unexpected seg output shapes.\n";
        return {};
    }

    const int C  = (int)predShape[1];
    const int N  = (int)predShape[2];
    const int nm = (int)protoShape[1];
    const int mh = (int)protoShape[2];
    const int mw = (int)protoShape[3];

    float* pred  = outs[pred_idx].GetTensorMutableData<float>();
    float* proto = outs[proto_idx].GetTensorMutableData<float>();

    // COCO assumptions
    const int nc = 80;
    const int person_id = 0;

    // Infer layout: with or without objectness
    bool has_obj = false;
    int cls_start = 4;
    int mask_start = -1;

    if (C == 4 + nc + nm) {
        has_obj = false; cls_start = 4; mask_start = cls_start + nc;
    } else if (C == 5 + nc + nm) {
        has_obj = true; cls_start = 5; mask_start = cls_start + nc;
    } else {
        // fallback guess
        if (C - 4 - nc == nm) { has_obj = false; cls_start = 4; mask_start = 4 + nc; }
        else if (C - 5 - nc == nm) { has_obj = true; cls_start = 5; mask_start = 5 + nc; }
        else {
            std::cerr << "Cannot infer YOLOv8-seg channel layout. C=" << C << " nm=" << nm << "\n";
            return {};
        }
    }

    float xScale = (float)frameBGR.cols / (float)inW;
    float yScale = (float)frameBGR.rows / (float)inH;

    std::vector<Det> candidates;
    candidates.reserve(256);

    for (int i = 0; i < N; i++) {
        float cx = pred[0 * N + i];
        float cy = pred[1 * N + i];
        float w  = pred[2 * N + i];
        float h  = pred[3 * N + i];

        float obj = 1.0f;
        if (has_obj) obj = pred[4 * N + i];

        float cls = pred[(cls_start + person_id) * N + i];
        float score = has_obj ? (obj * cls) : cls;
        if (score < confThresh) continue;

        float x1 = (cx - 0.5f * w) * xScale;
        float y1 = (cy - 0.5f * h) * yScale;
        float x2 = (cx + 0.5f * w) * xScale;
        float y2 = (cy + 0.5f * h) * yScale;

        //Clamp box coordinates to image boundaries

        x1 = std::max(0.0f, std::min(x1, (float)frameBGR.cols - 1));
        y1 = std::max(0.0f, std::min(y1, (float)frameBGR.rows - 1));
        x2 = std::max(0.0f, std::min(x2, (float)frameBGR.cols - 1));
        y2 = std::max(0.0f, std::min(y2, (float)frameBGR.rows - 1));

        int bw = (int)(x2 - x1);
        int bh = (int)(y2 - y1);
        if (bw <= 2 || bh <= 2) continue;

        // mask coeffs
        std::vector<float> coeff(nm);
        for (int k = 0; k < nm; k++) {
            coeff[k] = pred[(mask_start + k) * N + i];
        }

        Det d;
        d.box = cv::Rect((int)x1, (int)y1, bw, bh);
        d.conf = score;

        // Build mask logits on proto grid (mh x mw): sum_k coeff[k]*proto[k]
        cv::Mat logits(mh, mw, CV_32F, cv::Scalar(0));
        float* lg = (float*)logits.data;
        for (int k = 0; k < nm; k++) {
            const float* pk = proto + k * mh * mw;
            float ck = coeff[k];
            for (int p = 0; p < mh * mw; p++) lg[p] += ck * pk[p];
        }
        for (int p = 0; p < mh * mw; p++) lg[p] = sigmoidf(lg[p]);

        // Resize to frame size
        cv::Mat mask_full_f;
        cv::resize(logits, mask_full_f, frameBGR.size(), 0, 0, cv::INTER_LINEAR);

        // Threshold inside bbox and write to full-frame mask_u8
        cv::Mat mask_u8(frameBGR.rows, frameBGR.cols, CV_8U, cv::Scalar(0));
        cv::Rect r = d.box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
        if (r.width > 1 && r.height > 1) {
            cv::Mat roi = mask_full_f(r);
            cv::Mat roi_bin;
            cv::threshold(roi, roi_bin, 0.5, 255.0, cv::THRESH_BINARY);
            roi_bin.convertTo(roi_bin, CV_8U);
            roi_bin.copyTo(mask_u8(r));
            d.mask_u8 = std::move(mask_u8);
        }

        candidates.push_back(std::move(d));
    }

    return nms(candidates, iouThresh);
}

// ------------------------------ YOLOv8-POSE on crop ------------------------------
static void runPoseOnCropYOLOv8(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const cv::Mat& frameBGR,
    const cv::Rect& box,
    int inW, int inH,
    float poseDetThresh,
    std::vector<cv::Point2f>& out_kps,
    std::vector<float>& out_kpconf,
    bool debug_shapes = false
    ) {
    out_kps.clear();
    out_kpconf.clear();

    cv::Rect r = box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
    if (r.width <= 2 || r.height <= 2) return;

    cv::Mat crop = frameBGR(r).clone();
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(inW, inH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensor(1 * 3 * inH * inW);
    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; c++)
        chw[c] = cv::Mat(inH, inW, CV_32F, inputTensor.data() + c * inH * inW);
    cv::split(resized, chw);

    auto inNameA = session.GetInputNameAllocated(0, allocator);
    auto outNameA = session.GetOutputNameAllocated(0, allocator);
    const char* inName = inNameA.get();
    const char* outName = outNameA.get();

    std::vector<int64_t> inShape = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memInfo, inputTensor.data(), inputTensor.size(),
                                                   inShape.data(), inShape.size());

    const char* inNames[] = {inName};
    const char* outNames[] = {outName};

    auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &x, 1, outNames, 1);

    if (debug_shapes) print_shape(outs[0], "pose_out");

    auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape(); // common [1,C,N]
    if (shape.size() != 3) return;

    const int C = (int)shape[1];
    const int N = (int)shape[2];
    float* out = outs[0].GetTensorMutableData<float>();

    // Infer kp_start: common YOLOv8-pose exports
    // case1: [cx,cy,w,h,score, 17*3]
    // case2: [cx,cy,w,h,obj,cls?, 17*3]
    int kp_start = -1;
    int score_ch = 4;
    if (C >= 5 && (C - 5) >= 17 * 3 && ((C - 5) % 3 == 0)) {
        kp_start = 5;
        score_ch = 4;
    } else if (C >= 6 && (C - 6) >= 17 * 3 && ((C - 6) % 3 == 0)) {
        kp_start = 6;
        score_ch = 4; // may vary if your export differs
    } else {
        return;
    }

    // Choose the best prediction in the crop
    int best_i = -1;
    float best_score = 0.0f;
    for (int i = 0; i < N; i++) {
        float s = out[score_ch * N + i];
        if (s > best_score) { best_score = s; best_i = i; }
    }
    if (best_i < 0 || best_score < poseDetThresh) return;

    const int K = 17;
    out_kps.resize(K);
    out_kpconf.resize(K);

    float xScale = (float)r.width / (float)inW;
    float yScale = (float)r.height / (float)inH;

    for (int k = 0; k < K; k++) {
        float xk = out[(kp_start + k * 3 + 0) * N + best_i];
        float yk = out[(kp_start + k * 3 + 1) * N + best_i];
        float ck = out[(kp_start + k * 3 + 2) * N + best_i];

        out_kps[k] = cv::Point2f(r.x + xk * xScale, r.y + yk * yScale);
        out_kpconf[k] = ck;
    }
}

// ------------------------------ Tracker with long-term gallery ------------------------------
struct Track {
    int id = -1;
    cv::Rect box;
    std::vector<float> emb;
    int time_since_update = 0;
};

struct Tracker {
    int next_id = 1;
    std::vector<Track> tracks;

    // short-term
    int max_age = 120;               // increase to survive black screens
    float iou_gate = 0.05f;
    float cos_dist_thresh = 0.35f;
    float emb_momentum = 0.9f;

    // long-term gallery
    struct GalleryItem { std::vector<float> emb; int last_seen_frame = 0; };
    std::unordered_map<int, GalleryItem> gallery;

    int gallery_max_age = 5000;
    float gallery_cos_thresh = 0.35f;
    float gallery_momentum = 0.9f;

    static std::vector<float> smooth_emb(const std::vector<float>& oldEmb,
                                         const std::vector<float>& newEmb,
                                         float m) {
        if (oldEmb.empty()) return newEmb;
        if (oldEmb.size() != newEmb.size()) return newEmb;

        std::vector<float> out(oldEmb.size());
        for (size_t i = 0; i < out.size(); i++)
            out[i] = m * oldEmb[i] + (1.0f - m) * newEmb[i];

        double s = 0.0;
        for (float x : out) s += (double)x * x;
        s = std::sqrt(s) + 1e-12;
        for (float& x : out) x = (float)(x / s);
        return out;
    }

    void cleanup_gallery(int frame_idx) {
        for (auto it = gallery.begin(); it != gallery.end();) {
            if (frame_idx - it->second.last_seen_frame > gallery_max_age) it = gallery.erase(it);
            else ++it;
        }
    }

    int match_gallery_id(const std::vector<float>& emb) const {
        if (emb.empty()) return -1;
        int best_id = -1;
        float best_cd = 1.0f;
        for (const auto& kv : gallery) {
            float cd = cosine_distance(kv.second.emb, emb);
            if (cd < best_cd) { best_cd = cd; best_id = kv.first; }
        }
        if (best_id != -1 && best_cd <= gallery_cos_thresh) return best_id;
        return -1;
    }

    void update_gallery(int id, const std::vector<float>& emb, int frame_idx) {
        if (emb.empty()) return;
        auto& item = gallery[id];
        if (item.emb.empty()) item.emb = emb;
        else item.emb = smooth_emb(item.emb, emb, gallery_momentum);
        item.last_seen_frame = frame_idx;
    }

    void step(std::vector<Det>& dets, int frame_idx) {
        for (auto& t : tracks) t.time_since_update++;

        const int T = (int)tracks.size();
        const int D = (int)dets.size();

        struct Pair { int ti, di; float cost; float iou_v; float cd; };
        std::vector<Pair> pairs;
        pairs.reserve((size_t)T * (size_t)D);

        const float w_iou = 0.4f;
        const float w_reid = 0.6f;

        for (int ti = 0; ti < T; ti++) {
            for (int di = 0; di < D; di++) {
                float iou_v = iou(tracks[ti].box, dets[di].box);
                float cd = cosine_distance(tracks[ti].emb, dets[di].emb);
                float cost = w_iou * (1.0f - iou_v) + w_reid * cd;
                pairs.push_back({ti, di, cost, iou_v, cd});
            }
        }

        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
            return a.cost < b.cost;
        });

        std::vector<int> det_assigned(D, -1);
        std::vector<int> track_assigned(T, -1);

        for (const auto& p : pairs) {
            if (track_assigned[p.ti] != -1) continue;
            if (det_assigned[p.di] != -1) continue;

            bool ok = false;
            if (p.iou_v >= iou_gate && p.cd <= (cos_dist_thresh + 0.15f)) ok = true;
            if (p.iou_v <  iou_gate && p.cd <= cos_dist_thresh) ok = true;
            if (!ok) continue;

            track_assigned[p.ti] = p.di;
            det_assigned[p.di] = p.ti;
        }

        // Update matched tracks + gallery  (this includes your requested line)
        for (int ti = 0; ti < T; ti++) {
            int di = track_assigned[ti];
            if (di == -1) continue;

            tracks[ti].box = dets[di].box;
            tracks[ti].emb = smooth_emb(tracks[ti].emb, dets[di].emb, emb_momentum);
            tracks[ti].time_since_update = 0;

            update_gallery(tracks[ti].id, dets[di].emb, frame_idx);
        }

        // Unmatched detections: reuse ID from gallery or create new
        for (int di = 0; di < D; di++) {
            if (det_assigned[di] != -1) continue;

            int reuse_id = match_gallery_id(dets[di].emb);

            Track t;
            t.id = (reuse_id != -1) ? reuse_id : next_id++;
            t.box = dets[di].box;
            t.emb = dets[di].emb;
            t.time_since_update = 0;
            tracks.push_back(std::move(t));

            update_gallery(tracks.back().id, dets[di].emb, frame_idx);
        }

        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                    [&](const Track& t) { return t.time_since_update > max_age; }),
                     tracks.end());

        cleanup_gallery(frame_idx);
    }

    int get_id_for_box(const cv::Rect& box) const {
        int best_id = -1;
        float best = 0.0f;
        for (const auto& t : tracks) {
            float v = iou(t.box, box);
            if (v > best) { best = v; best_id = t.id; }
        }
        return best_id;
    }
};

// ------------------------------ Drawing (pose + mask) ------------------------------
static const std::vector<std::pair<int,int>> COCO17_EDGES = {
    {0,1},{0,2},{1,3},{2,4},
    {5,6},
    {5,7},{7,9},
    {6,8},{8,10},
    {5,11},{6,12},
    {11,12},
    {11,13},{13,15},
    {12,14},{14,16}
};

static void draw_pose(cv::Mat& frame, const std::vector<cv::Point2f>& kps,
                      const std::vector<float>& kpconf, float kp_thresh = 0.30f) {
    if (kps.size() != 17 || kpconf.size() != 17) return;
    for (auto [a,b] : COCO17_EDGES) {
        if (kpconf[a] < kp_thresh || kpconf[b] < kp_thresh) continue;
        cv::line(frame, kps[a], kps[b], cv::Scalar(0, 255, 255), 2);
    }
    for (int i = 0; i < 17; i++) {
        if (kpconf[i] < kp_thresh) continue;
        cv::circle(frame, kps[i], 3, cv::Scalar(255, 0, 0), -1);
    }
}

// Blend masks ONCE per frame (prevents the “frame becomes black” problem)
static void blend_all_masks_once(cv::Mat& frame, const std::vector<Det>& dets, float alpha = 0.30f) {
    cv::Mat overlay = frame.clone();
    for (const auto& d : dets) {
        if (!d.mask_u8.empty() && d.mask_u8.type() == CV_8U) {
            overlay.setTo(cv::Scalar(0, 0, 255), d.mask_u8);
        }
    }
    cv::addWeighted(overlay, alpha, frame, 1.0f - alpha, 0.0, frame);
}



// ------------------------------ main ------------------------------
int main() {
    // ---- paths ----
    std::string yoloSegPath  = "/home/mosta/Downloads/yolo_env/yolov8n-seg.onnx";
    std::string yoloPosePath = "/home/mosta/Downloads/yolo_env/yolov8n-pose.onnx";
    std::string reidPath     = "/home/mosta/Downloads/yolo_env/osnet_x0_25.onnx";
    std::string videoPath    = "/home/mosta/Downloads/wensday.mp4";
    std::string outPath      = "/home/mosta/Downloads/Clips/wensday_output.mp4";

    // Model input sizes (use what you exported with; 640 is typical)
    const int segW = 640, segH = 640;
    const int poseW = 640, poseH = 640;

    // thresholds
    const float detConfThresh  = 0.35f;
    const float detIouThresh   = 0.45f;

    const float poseDetThresh  = 0.25f; // crop-level pose detection score threshold
    const float kpDrawThresh   = 0.30f;

    // Debug shapes for frame 1 (set true if masks/pose look wrong)
    const bool debug_shapes_once = false;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath << "\n";
        return -1;
    }

    //Metadata for the frames

    int frameW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25;

    //video encoding

    cv::VideoWriter writer;
    if (!outPath.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(outPath, fourcc, fps, cv::Size(frameW, frameH));
        if (!writer.isOpened()) {
            std::cerr << "Warning: could not open writer, will not save output.\n";
            outPath.clear();
        }
    }

    // ---- ONNX Runtime ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SEG_POSE_REID_TRACK_GALLERY");
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session segSession(env, yoloSegPath.c_str(), opt);
    Ort::AllocatorWithDefaultOptions segAlloc;

    Ort::Session poseSession(env, yoloPosePath.c_str(), opt);
    Ort::AllocatorWithDefaultOptions poseAlloc;

    ReIDExtractor reid(env, reidPath, opt);

    Tracker tracker;
    // if you have ~3 seconds black screen, ensure max_age >= 3*fps
    tracker.max_age = std::max(tracker.max_age, (int)std::round(3.5 * fps)); // safety margin
    // tune if needed:
    // tracker.gallery_cos_thresh = 0.35f; // if it fails to reuse ID, try 0.38
    // tracker.gallery_cos_thresh = 0.30f; // if it swaps IDs, try 0.30 or 0.25



    cv::Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        frameIdx++;

        // 1) SEG person detections + masks
        auto dets = detectPersonsYOLOv8Seg(
            segSession, segAlloc, frame, segW, segH,
            detConfThresh, detIouThresh,
            debug_shapes_once && frameIdx == 1
            );

        // 2) ReID + Pose for each person
        for (auto& d : dets) {
            d.emb = reid.extract(frame, d.box);

            runPoseOnCropYOLOv8(
                poseSession, poseAlloc, frame, d.box,
                poseW, poseH, poseDetThresh,
                d.kps, d.kp_conf,
                debug_shapes_once && frameIdx == 1
                );
        }

        // 3) Tracker association + long-term gallery memory
        tracker.step(dets, frameIdx);

        // 4) Draw: masks once, then boxes+IDs+pose
        blend_all_masks_once(frame, dets, 0.30f);

        for (const auto& d : dets) {
            int id = tracker.get_id_for_box(d.box);

            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);

            std::string label = "ID " + std::to_string(id) + " conf " + cv::format("%.2f", d.conf);
            cv::putText(frame, label,
                        cv::Point(d.box.x, std::max(0, d.box.y - 6)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2);

            draw_pose(frame, d.kps, d.kp_conf, kpDrawThresh);
        }

        cv::imshow("YOLOv8-SEG + YOLOv8-POSE + ReID + Gallery Tracker", frame);
        int key = cv::waitKey(1);
        if (key == 27) break;

        if (!outPath.empty()) writer.write(frame);

        if (frameIdx % 50 == 0) std::cout << "Processed frame " << frameIdx << "\n";
    }



    cap.release();
    if (!outPath.empty()) writer.release();
    cv::destroyAllWindows();

    std::cout << "Done.\n";
    if (!outPath.empty()) std::cout << "Saved: " << outPath << "\n";
    return 0;
}
