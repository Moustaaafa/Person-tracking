#include "visualization.h"

namespace {

const std::vector<std::pair<int, int>> kCoco17Edges = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 6},
    {5, 7}, {7, 9},
    {6, 8}, {8, 10},
    {5, 11}, {6, 12},
    {11, 12},
    {11, 13}, {13, 15},
    {12, 14}, {14, 16}
};

}  // namespace

void draw_pose(cv::Mat& frame,
               const std::vector<cv::Point2f>& kps,
               const std::vector<float>& kpConf,
               float kpThresh) {
    if (kps.size() != 17 || kpConf.size() != 17) {
        return;
    }

    for (const auto& [a, b] : kCoco17Edges) {
        if (kpConf[a] < kpThresh || kpConf[b] < kpThresh) {
            continue;
        }
        cv::line(frame, kps[a], kps[b], cv::Scalar(0, 255, 255), 2);
    }

    for (int i = 0; i < 17; ++i) {
        if (kpConf[i] < kpThresh) {
            continue;
        }
        cv::circle(frame, kps[i], 3, cv::Scalar(255, 0, 0), -1);
    }
}

void blend_all_masks_once(cv::Mat& frame, const std::vector<Det>& dets, float alpha) {
    cv::Mat overlay = frame.clone();
    for (const auto& det : dets) {
        if (!det.mask_u8.empty() && det.mask_u8.type() == CV_8U) {
            overlay.setTo(cv::Scalar(0, 0, 255), det.mask_u8);
        }
    }
    cv::addWeighted(overlay, alpha, frame, 1.0f - alpha, 0.0, frame);
}
