#include "tracker.h"

#include <algorithm>
#include <cmath>

float Tracker::iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    const int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

float Tracker::cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 1.0f;
    }

    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(1.0 - dot);
}

std::vector<float> Tracker::smoothEmb(const std::vector<float>& oldEmb,
                                      const std::vector<float>& newEmb,
                                      float momentum) {
    if (oldEmb.empty() || oldEmb.size() != newEmb.size()) {
        return newEmb;
    }

    std::vector<float> out(oldEmb.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = momentum * oldEmb[i] + (1.0f - momentum) * newEmb[i];
    }

    double sum = 0.0;
    for (float value : out) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    sum = std::sqrt(sum) + 1e-12;
    for (float& value : out) {
        value = static_cast<float>(value / sum);
    }
    return out;
}

void Tracker::cleanupGallery(int frameIdx) {
    for (auto it = gallery.begin(); it != gallery.end();) {
        if (frameIdx - it->second.last_seen_frame > gallery_max_age) {
            it = gallery.erase(it);
        } else {
            ++it;
        }
    }
}

int Tracker::matchGalleryId(const std::vector<float>& emb) const {
    if (emb.empty()) {
        return -1;
    }

    int bestId = -1;
    float bestDistance = 1.0f;
    for (const auto& [id, item] : gallery) {
        const float distance = cosineDistance(item.emb, emb);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestId = id;
        }
    }

    if (bestId != -1 && bestDistance <= gallery_cos_thresh) {
        return bestId;
    }
    return -1;
}

void Tracker::updateGallery(int id, const std::vector<float>& emb, int frameIdx) {
    if (emb.empty()) {
        return;
    }

    auto& item = gallery[id];
    if (item.emb.empty()) {
        item.emb = emb;
    } else {
        item.emb = smoothEmb(item.emb, emb, gallery_momentum);
    }
    item.last_seen_frame = frameIdx;
}

void Tracker::step(std::vector<Det>& dets, int frameIdx) {
    for (auto& track : tracks) {
        track.time_since_update++;
    }

    const int trackCount = static_cast<int>(tracks.size());
    const int detCount = static_cast<int>(dets.size());

    struct Pair {
        int trackIndex;
        int detIndex;
        float cost;
        float iouValue;
        float cosineDistance;
    };

    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(trackCount) * static_cast<size_t>(detCount));

    const float wIou = 0.4f;
    const float wReid = 0.6f;

    for (int ti = 0; ti < trackCount; ++ti) {
        for (int di = 0; di < detCount; ++di) {
            const float iouValue = iou(tracks[ti].box, dets[di].box);
            const float distance = cosineDistance(tracks[ti].emb, dets[di].emb);
            const float cost = wIou * (1.0f - iouValue) + wReid * distance;
            pairs.push_back({ti, di, cost, iouValue, distance});
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        return a.cost < b.cost;
    });

    std::vector<int> detAssigned(detCount, -1);
    std::vector<int> trackAssigned(trackCount, -1);

    for (const auto& pair : pairs) {
        if (trackAssigned[pair.trackIndex] != -1 || detAssigned[pair.detIndex] != -1) {
            continue;
        }

        bool ok = false;
        if (pair.iouValue >= iou_gate && pair.cosineDistance <= (cos_dist_thresh + 0.15f)) {
            ok = true;
        }
        if (pair.iouValue < iou_gate && pair.cosineDistance <= cos_dist_thresh) {
            ok = true;
        }
        if (!ok) {
            continue;
        }

        trackAssigned[pair.trackIndex] = pair.detIndex;
        detAssigned[pair.detIndex] = pair.trackIndex;
    }

    for (int ti = 0; ti < trackCount; ++ti) {
        const int di = trackAssigned[ti];
        if (di == -1) {
            continue;
        }

        tracks[ti].box = dets[di].box;
        tracks[ti].emb = smoothEmb(tracks[ti].emb, dets[di].emb, emb_momentum);
        tracks[ti].time_since_update = 0;
        updateGallery(tracks[ti].id, dets[di].emb, frameIdx);
    }

    for (int di = 0; di < detCount; ++di) {
        if (detAssigned[di] != -1) {
            continue;
        }

        const int reuseId = matchGalleryId(dets[di].emb);

        Track track;
        track.id = (reuseId != -1) ? reuseId : next_id++;
        track.box = dets[di].box;
        track.emb = dets[di].emb;
        track.time_since_update = 0;
        tracks.push_back(std::move(track));

        updateGallery(tracks.back().id, dets[di].emb, frameIdx);
    }

    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(), [&](const Track& track) {
            return track.time_since_update > max_age;
        }),
        tracks.end());

    cleanupGallery(frameIdx);
}

int Tracker::get_id_for_box(const cv::Rect& box) const {
    int bestId = -1;
    float bestIou = 0.0f;
    for (const auto& track : tracks) {
        const float currentIou = iou(track.box, box);
        if (currentIou > bestIou) {
            bestIou = currentIou;
            bestId = track.id;
        }
    }
    return bestId;
}
