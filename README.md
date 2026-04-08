# Person Tracking App

A person tracking application built with **C++** and **Qt Creator**, 
combining state-of-the-art deep learning models for detection, segmentation, 
pose estimation, and re-identification.

## Pipeline

### 1. YOLOv8-SEG — Person Detection + Segmentation
- Detects persons in each frame
- Generates instance segmentation masks
- Runs via ONNX Runtime

### 2. YOLOv8-POSE — Pose Estimation
- Runs on each detected person crop
- Estimates 17 COCO keypoints (nose, shoulders, elbows, knees, etc.)
- Visualizes skeleton overlay on each person

### 3. OSNet — Re-Identification (ReID)
- Extracts a 512-dim appearance embedding for each detected person
- Used to match persons across frames
- Embeddings are L2-normalized and EMA-smoothed over time

### 4. Tracker — IoU + ReID Association
- Matches detections to existing tracks using:
  - **IoU** (spatial overlap) weighted at 0.4
  - **Cosine distance** (appearance similarity) weighted at 0.6
- Greedy matching on sorted cost matrix
- **Long-term gallery memory** — remembers person embeddings for up to 5000 frames, allowing ID reuse when a person reappears after a long absence

## Tech Stack
- **Language:** C++17
- **IDE:** Qt Creator
- **Inference:** ONNX Runtime
- **Vision:** OpenCV 4
- **Models:** YOLOv8n-seg, YOLOv8n-pose, OSNet x0.25

## Build

### Requirements
```bash
sudo apt install libonnxruntime-dev
sudo apt install libopencv-dev
```

### Compile
Open `Person_tracking.pro` in Qt Creator and click **Build**, or:
```bash
qmake Person_tracking.pro
make
```

## Usage
Edit the paths in `main.cpp`:
```cpp
std::string yoloSegPath  = "path/to/yolov8n-seg.onnx";
std::string yoloPosePath = "path/to/yolov8n-pose.onnx";
std::string reidPath     = "path/to/osnet_x0_25.onnx";
std::string videoPath    = "path/to/your/video.mp4";
std::string outPath      = "path/to/output.mp4";
```
Then build and run.

## Output
- Bounding boxes with **person ID** and confidence score
- **Segmentation mask** overlay (red)
- **Pose skeleton** with 17 keypoints
- Saved output video
