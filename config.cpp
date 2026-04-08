#include "config.h"

AppConfig makeDefaultConfig() {
    AppConfig config;
    config.yoloSegPath = "/home/mosta/Downloads/yolo_env/yolov8n-seg.onnx";
    config.yoloPosePath = "/home/mosta/Downloads/yolo_env/yolov8n-pose.onnx";
    config.reidPath = "/home/mosta/Downloads/yolo_env/osnet_x0_25.onnx";
    config.videoPath = "/home/mosta/Downloads/wensday.mp4";
    config.outPath = "/home/mosta/Downloads/Clips/wensday_output.mp4";
    return config;
}
