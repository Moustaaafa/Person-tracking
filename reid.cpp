#include "reid.h"

#include <cmath>
#include <cstring>

ReIDExtractor::ReIDExtractor(Ort::Env& env, const std::string& modelPath, const Ort::SessionOptions& options)
    : session(env, modelPath.c_str(), options) {}

void ReIDExtractor::l2Normalize(std::vector<float>& values) {
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }

    sum = std::sqrt(sum) + 1e-12;
    for (float& value : values) {
        value = static_cast<float>(value / sum);
    }
}

std::vector<float> ReIDExtractor::extract(const cv::Mat& frameBGR, const cv::Rect& box) {
    const cv::Rect roi = box & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
    if (roi.width <= 1 || roi.height <= 1) {
        return {};
    }

    cv::Mat crop = frameBGR(roi).clone();
    cv::resize(crop, crop, cv::Size(inW, inH));
    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    crop.convertTo(crop, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(crop, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;

    std::vector<float> input(3 * inH * inW);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input.data() + c * inH * inW, channels[c].data, static_cast<size_t>(inH * inW) * sizeof(float));
    }

    auto inNameA = session.GetInputNameAllocated(0, allocator);
    auto outNameA = session.GetOutputNameAllocated(0, allocator);
    const char* inName = inNameA.get();
    const char* outName = outNameA.get();

    const std::vector<int64_t> shape = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, input.data(), input.size(), shape.data(), shape.size());

    const char* inputNames[] = {inName};
    const char* outputNames[] = {outName};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    float* embPtr = outputs[0].GetTensorMutableData<float>();
    const auto embShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    int dim = 1;
    for (const auto value : embShape) {
        if (value > 0) {
            dim *= static_cast<int>(value);
        }
    }

    std::vector<float> embedding(embPtr, embPtr + dim);
    l2Normalize(embedding);
    return embedding;
}
