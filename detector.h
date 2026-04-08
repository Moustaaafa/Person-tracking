#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Det {
    cv::Rect box;
    float conf = 0.0f;
    std::vector<float> emb;
    cv::Mat mask_u8;
    std::vector<cv::Point2f> kps;
    std::vector<float> kp_conf;
};

std::vector<Det> detectPersonsYOLOv8Seg(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const cv::Mat& frameBGR,
    int inW,
    int inH,
    float confThresh,
    float iouThresh,
    bool debugShapes = false);
