#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "detector.h"

void draw_pose(cv::Mat& frame,
               const std::vector<cv::Point2f>& kps,
               const std::vector<float>& kpConf,
               float kpThresh = 0.30f);

void blend_all_masks_once(cv::Mat& frame, const std::vector<Det>& dets, float alpha = 0.30f);
