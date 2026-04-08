#pragma once

#include <string>

struct AppConfig {
    std::string yoloSegPath;
    std::string yoloPosePath;
    std::string reidPath;
    std::string videoPath;
    std::string outPath;

    int segW = 640;
    int segH = 640;
    int poseW = 640;
    int poseH = 640;

    float detConfThresh = 0.35f;
    float detIouThresh = 0.45f;
    float poseDetThresh = 0.25f;
    float kpDrawThresh = 0.30f;

    bool debugShapesOnce = false;
    std::string windowTitle = "YOLOv8-SEG + YOLOv8-POSE + ReID + Gallery Tracker";
};

AppConfig makeDefaultConfig();
