#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class ReIDExtractor {
public:
    ReIDExtractor(Ort::Env& env, const std::string& modelPath, const Ort::SessionOptions& options);

    std::vector<float> extract(const cv::Mat& frameBGR, const cv::Rect& box);

private:
    static void l2Normalize(std::vector<float>& values);

    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    const int inH = 256;
    const int inW = 128;
};
