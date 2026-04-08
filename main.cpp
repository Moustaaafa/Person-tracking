#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "config.h"
#include "detector.h"
#include "pose.h"
#include "reid.h"
#include "tracker.h"
#include "visualization.h"

int main() {
    const AppConfig config = makeDefaultConfig();

    cv::VideoCapture cap(config.videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << config.videoPath << "\n";
        return -1;
    }

    const int frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        fps = 25;
    }

    cv::VideoWriter writer;
    std::string outPath = config.outPath;
    if (!outPath.empty()) {
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(outPath, fourcc, fps, cv::Size(frameW, frameH));
        if (!writer.isOpened()) {
            std::cerr << "Warning: could not open writer, will not save output.\n";
            outPath.clear();
        }
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SEG_POSE_REID_TRACK_GALLERY");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session segSession(env, config.yoloSegPath.c_str(), options);
    Ort::AllocatorWithDefaultOptions segAllocator;

    Ort::Session poseSession(env, config.yoloPosePath.c_str(), options);
    Ort::AllocatorWithDefaultOptions poseAllocator;

    ReIDExtractor reid(env, config.reidPath, options);

    Tracker tracker;
    tracker.max_age = std::max(tracker.max_age, static_cast<int>(std::round(3.5 * fps)));

    cv::Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        ++frameIdx;

        auto dets = detectPersonsYOLOv8Seg(
            segSession,
            segAllocator,
            frame,
            config.segW,
            config.segH,
            config.detConfThresh,
            config.detIouThresh,
            config.debugShapesOnce && frameIdx == 1);

        for (auto& det : dets) {
            det.emb = reid.extract(frame, det.box);

            runPoseOnCropYOLOv8(
                poseSession,
                poseAllocator,
                frame,
                det.box,
                config.poseW,
                config.poseH,
                config.poseDetThresh,
                det.kps,
                det.kp_conf,
                config.debugShapesOnce && frameIdx == 1);
        }

        tracker.step(dets, frameIdx);

        blend_all_masks_once(frame, dets, 0.30f);

        for (const auto& det : dets) {
            const int id = tracker.get_id_for_box(det.box);

            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

            const std::string label = "ID " + std::to_string(id) + " conf " + cv::format("%.2f", det.conf);
            cv::putText(frame,
                        label,
                        cv::Point(det.box.x, std::max(0, det.box.y - 6)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 255, 0),
                        2);

            draw_pose(frame, det.kps, det.kp_conf, config.kpDrawThresh);
        }

        cv::imshow(config.windowTitle, frame);
        const int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }

        if (!outPath.empty()) {
            writer.write(frame);
        }

        if (frameIdx % 50 == 0) {
            std::cout << "Processed frame " << frameIdx << "\n";
        }
    }

    cap.release();
    if (!outPath.empty()) {
        writer.release();
    }
    cv::destroyAllWindows();

    std::cout << "Done.\n";
    if (!outPath.empty()) {
        std::cout << "Saved: " << outPath << "\n";
    }
    return 0;
}
