#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

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
    bool debugShapes = false);
