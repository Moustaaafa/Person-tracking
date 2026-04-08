#pragma once

#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detector.h"

struct Track {
    int id = -1;
    cv::Rect box;
    std::vector<float> emb;
    int time_since_update = 0;
};

struct GalleryItem {
    std::vector<float> emb;
    int last_seen_frame = 0;
};

class Tracker {
public:
    int next_id = 1;
    std::vector<Track> tracks;
    int max_age = 120;
    float iou_gate = 0.05f;
    float cos_dist_thresh = 0.35f;
    float emb_momentum = 0.9f;
    std::unordered_map<int, GalleryItem> gallery;
    int gallery_max_age = 5000;
    float gallery_cos_thresh = 0.35f;
    float gallery_momentum = 0.9f;

    void step(std::vector<Det>& dets, int frameIdx);
    int get_id_for_box(const cv::Rect& box) const;

private:
    static float iou(const cv::Rect& a, const cv::Rect& b);
    static float cosineDistance(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> smoothEmb(const std::vector<float>& oldEmb,
                                        const std::vector<float>& newEmb,
                                        float momentum);
    void cleanupGallery(int frameIdx);
    int matchGalleryId(const std::vector<float>& emb) const;
    void updateGallery(int id, const std::vector<float>& emb, int frameIdx);
};
