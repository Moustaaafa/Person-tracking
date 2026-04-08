#include "detector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

namespace {

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    const int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::vector<int> argsortDesc(const std::vector<float>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return v[i] > v[j]; });
    return idx;
}

std::vector<Det> nms(const std::vector<Det>& dets, float iouThresh) {
    std::vector<Det> out;
    if (dets.empty()) {
        return out;
    }

    std::vector<float> scores(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        scores[i] = dets[i].conf;
    }

    const auto order = argsortDesc(scores);
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t orderIdx = 0; orderIdx < order.size(); ++orderIdx) {
        const int i = order[orderIdx];
        if (suppressed[i]) {
            continue;
        }

        out.push_back(dets[i]);

        for (size_t nextIdx = orderIdx + 1; nextIdx < order.size(); ++nextIdx) {
            const int j = order[nextIdx];
            if (!suppressed[j] && iou(dets[i].box, dets[j].box) > iouThresh) {
                suppressed[j] = true;
            }
        }
    }

    return out;
}

void printShape(const Ort::Value& value, const std::string& name) {
    const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << name << " shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cerr << shape[i] << (i + 1 < shape.size() ? "," : "");
    }
    std::cerr << "]\n";
}

}  // namespace

std::vector<Det> detectPersonsYOLOv8Seg(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const cv::Mat& frameBGR,
    int inW,
    int inH,
    float confThresh,
    float iouThresh,
    bool debugShapes) {
    cv::Mat resized;
    cv::resize(frameBGR, resized, cv::Size(inW, inH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensor(3 * inH * inW);
    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; ++c) {
        chw[c] = cv::Mat(inH, inW, CV_32F, inputTensor.data() + c * inH * inW);
    }
    cv::split(resized, chw);

    auto inNameA = session.GetInputNameAllocated(0, allocator);
    auto out0A = session.GetOutputNameAllocated(0, allocator);
    auto out1A = session.GetOutputNameAllocated(1, allocator);
    const char* inName = inNameA.get();
    const char* out0 = out0A.get();
    const char* out1 = out1A.get();

    const std::vector<int64_t> inShape = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memInfo, inputTensor.data(), inputTensor.size(), inShape.data(), inShape.size());

    const char* inNames[] = {inName};
    const char* outNames[] = {out0, out1};
    auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 2);

    if (debugShapes) {
        printShape(outs[0], "seg_out0");
        printShape(outs[1], "seg_out1");
    }

    int predIdx = 0;
    int protoIdx = 1;
    {
        const auto s0 = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto s1 = outs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (s0.size() == 4 && s1.size() == 3) {
            predIdx = 1;
            protoIdx = 0;
        } else if (s0.size() == 3 && s1.size() == 4) {
            predIdx = 0;
            protoIdx = 1;
        }
    }

    const auto predShape = outs[predIdx].GetTensorTypeAndShapeInfo().GetShape();
    const auto protoShape = outs[protoIdx].GetTensorTypeAndShapeInfo().GetShape();
    if (predShape.size() != 3 || protoShape.size() != 4) {
        std::cerr << "Unexpected seg output shapes.\n";
        return {};
    }

    const int c = static_cast<int>(predShape[1]);
    const int n = static_cast<int>(predShape[2]);
    const int nm = static_cast<int>(protoShape[1]);
    const int mh = static_cast<int>(protoShape[2]);
    const int mw = static_cast<int>(protoShape[3]);

    float* pred = outs[predIdx].GetTensorMutableData<float>();
    float* proto = outs[protoIdx].GetTensorMutableData<float>();

    const int nc = 80;
    const int personId = 0;

    bool hasObj = false;
    int clsStart = 4;
    int maskStart = -1;

    if (c == 4 + nc + nm) {
        clsStart = 4;
        maskStart = clsStart + nc;
    } else if (c == 5 + nc + nm) {
        hasObj = true;
        clsStart = 5;
        maskStart = clsStart + nc;
    } else if (c - 4 - nc == nm) {
        clsStart = 4;
        maskStart = 4 + nc;
    } else if (c - 5 - nc == nm) {
        hasObj = true;
        clsStart = 5;
        maskStart = 5 + nc;
    } else {
        std::cerr << "Cannot infer YOLOv8-seg channel layout. C=" << c << " nm=" << nm << "\n";
        return {};
    }

    const float xScale = static_cast<float>(frameBGR.cols) / static_cast<float>(inW);
    const float yScale = static_cast<float>(frameBGR.rows) / static_cast<float>(inH);

    std::vector<Det> candidates;
    candidates.reserve(256);

    for (int i = 0; i < n; ++i) {
        const float cx = pred[0 * n + i];
        const float cy = pred[1 * n + i];
        const float w = pred[2 * n + i];
        const float h = pred[3 * n + i];

        float obj = 1.0f;
        if (hasObj) {
            obj = pred[4 * n + i];
        }

        const float cls = pred[(clsStart + personId) * n + i];
        const float score = hasObj ? (obj * cls) : cls;
        if (score < confThresh) {
            continue;
        }

        float x1 = (cx - 0.5f * w) * xScale;
        float y1 = (cy - 0.5f * h) * yScale;
        float x2 = (cx + 0.5f * w) * xScale;
        float y2 = (cy + 0.5f * h) * yScale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(frameBGR.cols - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(frameBGR.rows - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(frameBGR.cols - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(frameBGR.rows - 1)));

        const int bw = static_cast<int>(x2 - x1);
        const int bh = static_cast<int>(y2 - y1);
        if (bw <= 2 || bh <= 2) {
            continue;
        }

        std::vector<float> coeff(nm);
        for (int k = 0; k < nm; ++k) {
            coeff[k] = pred[(maskStart + k) * n + i];
        }

        Det det;
        det.box = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), bw, bh);
        det.conf = score;

        cv::Mat logits(mh, mw, CV_32F, cv::Scalar(0));
        float* logitsData = reinterpret_cast<float*>(logits.data);
        for (int k = 0; k < nm; ++k) {
            const float* protoK = proto + k * mh * mw;
            const float coeffK = coeff[k];
            for (int p = 0; p < mh * mw; ++p) {
                logitsData[p] += coeffK * protoK[p];
            }
        }
        for (int p = 0; p < mh * mw; ++p) {
            logitsData[p] = sigmoidf(logitsData[p]);
        }

        cv::Mat maskFullFloat;
        cv::resize(logits, maskFullFloat, frameBGR.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat mask(frameBGR.rows, frameBGR.cols, CV_8U, cv::Scalar(0));
        const cv::Rect roi = det.box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
        if (roi.width > 1 && roi.height > 1) {
            cv::Mat roiFloat = maskFullFloat(roi);
            cv::Mat roiBinary;
            cv::threshold(roiFloat, roiBinary, 0.5, 255.0, cv::THRESH_BINARY);
            roiBinary.convertTo(roiBinary, CV_8U);
            roiBinary.copyTo(mask(roi));
            det.mask_u8 = std::move(mask);
        }

        candidates.push_back(std::move(det));
    }

    return nms(candidates, iouThresh);
}
