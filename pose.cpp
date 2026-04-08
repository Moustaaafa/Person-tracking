#include "pose.h"

#include <iostream>
#include <string>

namespace {

void printShape(const Ort::Value& value, const std::string& name) {
    const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << name << " shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cerr << shape[i] << (i + 1 < shape.size() ? "," : "");
    }
    std::cerr << "]\n";
}

}  // namespace

void runPoseOnCropYOLOv8(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const cv::Mat& frameBGR,
    const cv::Rect& box,
    int inW,
    int inH,
    float poseDetThresh,
    std::vector<cv::Point2f>& outKps,
    std::vector<float>& outKpConf,
    bool debugShapes) {
    outKps.clear();
    outKpConf.clear();

    const cv::Rect roi = box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
    if (roi.width <= 2 || roi.height <= 2) {
        return;
    }

    cv::Mat crop = frameBGR(roi).clone();
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(inW, inH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensor(3 * inH * inW);
    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; ++c) {
        chw[c] = cv::Mat(inH, inW, CV_32F, inputTensor.data() + c * inH * inW);
    }
    cv::split(resized, chw);

    auto inNameA = session.GetInputNameAllocated(0, allocator);
    auto outNameA = session.GetOutputNameAllocated(0, allocator);
    const char* inName = inNameA.get();
    const char* outName = outNameA.get();

    const std::vector<int64_t> inShape = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memInfo, inputTensor.data(), inputTensor.size(), inShape.data(), inShape.size());

    const char* inNames[] = {inName};
    const char* outNames[] = {outName};
    auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);

    if (debugShapes) {
        printShape(outs[0], "pose_out");
    }

    const auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3) {
        return;
    }

    const int c = static_cast<int>(shape[1]);
    const int n = static_cast<int>(shape[2]);
    float* output = outs[0].GetTensorMutableData<float>();

    int kpStart = -1;
    int scoreChannel = 4;
    if (c >= 5 && (c - 5) >= 17 * 3 && ((c - 5) % 3 == 0)) {
        kpStart = 5;
    } else if (c >= 6 && (c - 6) >= 17 * 3 && ((c - 6) % 3 == 0)) {
        kpStart = 6;
    } else {
        return;
    }

    int bestIndex = -1;
    float bestScore = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float score = output[scoreChannel * n + i];
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }

    if (bestIndex < 0 || bestScore < poseDetThresh) {
        return;
    }

    const int keypointCount = 17;
    outKps.resize(keypointCount);
    outKpConf.resize(keypointCount);

    const float xScale = static_cast<float>(roi.width) / static_cast<float>(inW);
    const float yScale = static_cast<float>(roi.height) / static_cast<float>(inH);

    for (int k = 0; k < keypointCount; ++k) {
        const float xk = output[(kpStart + k * 3 + 0) * n + bestIndex];
        const float yk = output[(kpStart + k * 3 + 1) * n + bestIndex];
        const float ck = output[(kpStart + k * 3 + 2) * n + bestIndex];

        outKps[k] = cv::Point2f(roi.x + xk * xScale, roi.y + yk * yScale);
        outKpConf[k] = ck;
    }
}
