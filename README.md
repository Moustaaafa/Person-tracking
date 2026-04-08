Person Tracking System Documentation

YOLOv8-SEG + YOLOv8-POSE + OSNet ReID + IoU/ReID Tracker with Long-Term Gallery

1. Overview

This application is a multi-stage computer vision pipeline for tracking people in video. It combines:

Person detection
Instance segmentation
Pose estimation
Person re-identification
Multi-object tracking
Long-term identity recovery after disappearance

The system reads a video, detects people in each frame, estimates their body keypoints, extracts appearance embeddings, and assigns stable IDs over time. If a person leaves the scene and reappears later, the tracker can reuse the old ID using a gallery-based long-term memory.

2. Main Purpose

The code is designed to:

detect only persons
generate a segmentation mask for each person
estimate COCO-17 body keypoints
extract a ReID embedding for appearance-based identity matching
track people frame to frame using:
IoU
cosine distance between embeddings
preserve identity over long gaps using a gallery memory

This makes it suitable for:

surveillance analytics
sports or activity analysis
human motion analysis
re-identification after occlusion or temporary disappearance
3. Build and Runtime
Build
g++ main.cpp -O2 -std=c++17 `pkg
in md
# Person Tracking System Documentation  
**YOLOv8-SEG + YOLOv8-POSE + OSNet ReID + IoU/ReID Tracker with Long-Term Gallery**

## 1. Overview

This application is a multi-stage computer vision pipeline for tracking people in video. It combines:

- person detection
- instance segmentation
- pose estimation
- person re-identification
- multi-object tracking
- long-term identity recovery after disappearance

The system reads a video, detects people in each frame, estimates body keypoints, extracts appearance embeddings, and assigns stable IDs over time. If a person leaves the scene and reappears later, the tracker can reuse the old ID using a gallery-based long-term memory.

---

## 2. Main Purpose

The code is designed to:

- detect only persons
- generate a segmentation mask for each person
- estimate COCO-17 body keypoints
- extract a ReID embedding for appearance-based identity matching
- track people frame to frame using:
  - IoU
  - cosine distance between embeddings
- preserve identity over long gaps using a gallery memory

This makes it suitable for:

- surveillance analytics
- sports or activity analysis
- human motion analysis
- re-identification after occlusion or temporary disappearance

---

## 3. Build and Runtime

## Build
```bash
g++ main.cpp -O2 -std=c++17 `pkg-config --cflags --libs opencv4` -lonnxruntime -o app
Run

Update the model and video paths inside main() and then run:

./app
4. High-Level Architecture

The pipeline consists of the following major modules:

YOLOv8-SEG
detects persons
produces bounding boxes
produces instance mask coefficients and mask prototypes
YOLOv8-POSE
runs on each detected person crop
predicts body keypoints for that person
OSNet ReID
extracts an appearance embedding from the person crop
embedding is L2-normalized for cosine comparison
Tracker
matches detections to tracks using IoU and ReID similarity
handles missed detections over short gaps
stores long-term appearance memory in a gallery
Visualization
overlays masks
draws bounding boxes
shows track IDs
draws pose skeletons
5. Models Used
5.1 YOLOv8-SEG
Purpose

Used for:

person detection
instance segmentation
Expected Outputs

The code supports common YOLOv8-seg ONNX layouts:

pred: [1, C, N]
proto: [1, nm, mh, mw]

Where:

N = number of candidate detections
C = channels per prediction
nm = number of mask coefficients
mh, mw = prototype mask resolution
Supported Layout Variants

The code handles both:

C = 4 + nc + nm
C = 5 + nc + nm

That means it supports exports:

without objectness
with objectness
Assumptions
COCO dataset format
nc = 80
person class id = 0
Output Usage

For each person detection:

bounding box is decoded
confidence score is computed
mask coefficients are extracted
prototype masks are combined into a full-frame binary mask
5.2 YOLOv8-POSE
Purpose

Used for per-person pose estimation.

Input Strategy

Instead of running pose on the full frame, the code:

crops each detected person from the frame
resizes the crop
runs the pose model only on that crop

This reduces ambiguity and improves per-person keypoint association.

Expected Output

Common format assumed:

[1, C, N]

The code infers the keypoint start index automatically for common layouts such as:

[cx, cy, w, h, score, 17*3]
[cx, cy, w, h, obj, cls?, 17*3]
Keypoints

The system expects COCO-17 keypoints, each represented as:

x
y
confidence
Output Usage

For the best pose candidate in the crop:

keypoints are scaled back into original frame coordinates
confidence values are stored
skeleton lines and joints are drawn later
5.3 OSNet ReID
Purpose

Used for person re-identification.

Input
cropped person image from the bounding box
resized to 128 x 256
converted from BGR to RGB
normalized using ImageNet-like mean/std
Output
feature embedding vector
L2-normalized
Why It Matters

This embedding gives each person an appearance signature. It helps:

distinguish nearby people with overlapping boxes
maintain identity when IoU alone is unreliable
recover the same ID after temporary disappearance
6. Core Data Structures
6.1 Det

Represents one detection in the current frame.

Fields
box: bounding box
conf: detection confidence
emb: ReID embedding
mask_u8: full-frame binary mask
kps: pose keypoints
kp_conf: keypoint confidences

This structure accumulates all information related to one detected person.

6.2 Track

Represents one tracked identity.

Fields
id: persistent person ID
box: latest bounding box
emb: smoothed appearance embedding
time_since_update: number of frames since last matched detection
6.3 GalleryItem

Represents a long-term memory entry for an identity.

Fields
emb: long-term appearance embedding
last_seen_frame: frame index of last update

The gallery allows the system to reuse old IDs when a person reappears.

7. Utility Functions
sigmoidf

Applies sigmoid activation to convert mask logits into probabilities.

iou

Computes intersection-over-union between two boxes.

Used for:

NMS
track association
cosine_distance

Measures appearance difference between two embeddings.

Used for:

track matching
gallery matching
argsort_desc

Returns indices sorted by descending confidence.

nms

Performs non-maximum suppression on detections using IoU threshold.

print_shape

Prints ONNX tensor output shapes for debugging model exports.

8. ReID Extraction Pipeline

Implemented in ReIDExtractor.

Workflow
clip the input box to frame boundaries
crop the image
resize to 128 x 256
convert BGR to RGB
normalize input channels
build tensor in CHW format
run ONNX model
read embedding output
L2-normalize embedding
Output

A normalized embedding vector stored in Det.emb.

9. Person Detection and Segmentation Pipeline

Implemented in detectPersonsYOLOv8Seg().

Workflow
1. Preprocessing
resize frame to model input size
convert BGR to RGB
normalize to [0,1]
rearrange to CHW tensor
2. ONNX Inference

The segmentation model is run with two outputs:

prediction tensor
prototype mask tensor
3. Output Interpretation

The code determines which output is:

prediction tensor
prototype tensor

by checking tensor rank.

4. Layout Inference

The code automatically checks whether the model output includes:

objectness score
or not
5. Detection Decoding

For each prediction:

decode box center and size
compute score for class person
reject low-confidence detections
scale box back to original frame size
clamp box to image boundaries
6. Mask Construction

For each valid person detection:

extract mask coefficients
combine them with the mask prototypes
apply sigmoid
resize the mask to full-frame resolution
threshold it inside the bounding box
store the binary mask in mask_u8
7. NMS

After all candidates are built, non-maximum suppression removes overlapping detections.

Output

A vector of Det objects containing:

person box
confidence
instance mask
10. Pose Estimation Pipeline

Implemented in runPoseOnCropYOLOv8().

Workflow
1. Crop Person Region

The detection box is clipped to image bounds and cropped.

2. Preprocess
resize crop to pose model input size
convert BGR to RGB
normalize to [0,1]
convert to CHW tensor
3. Run ONNX Pose Inference

The crop is passed to the pose network.

4. Infer Output Layout

The function checks the output channels and infers where keypoint values begin.

5. Select Best Detection

Among the pose candidates in the crop, the one with the highest detection score is selected.

6. Decode Keypoints

For each of the 17 keypoints:

read x
read y
read confidence
map coordinates back to original frame coordinates
Output

Two vectors are filled:

out_kps
out_kpconf

These are stored in the detection object.

11. Tracking Pipeline

Implemented in Tracker::step().

The tracking strategy combines:

IoU between boxes
cosine distance between ReID embeddings
11.1 Track Update Preparation

At the beginning of each frame:

time_since_update is increased for all tracks
11.2 Pairwise Matching Cost

For each track and detection pair:

IoU is computed
cosine distance is computed
total cost is computed as:
cost = 0.4 * (1 - IoU) + 0.6 * cosine_distance

This gives more weight to appearance similarity than box overlap.

11.3 Greedy Assignment

All track-detection pairs are sorted by ascending cost.

A match is accepted if it passes gating rules:

if IoU is good enough and appearance is reasonable
or if IoU is weak but appearance is very strong

This helps maintain identity through motion, occlusion, and camera gaps.

11.4 Matched Track Update

When a detection matches an existing track:

box is updated
embedding is smoothed using momentum
time_since_update is reset
gallery memory is updated
11.5 Unmatched Detection Handling

If a detection does not match any active track:

the gallery is searched for a similar historical embedding
if a similar identity exists, that old ID is reused
otherwise a new ID is created
11.6 Dead Track Removal

Tracks that exceed max_age without updates are removed.

11.7 Gallery Cleanup

Old gallery items beyond gallery_max_age are deleted.

12. Long-Term Gallery Memory

The gallery is the key mechanism for ID reuse after long gaps.

Why It Exists

A normal tracker usually loses identity when:

a person disappears for many frames
a blackout occurs
the person exits and later re-enters

The gallery solves that by storing appearance memory for each known ID.

Gallery Matching

When a new unmatched detection appears:

compare its embedding against all gallery embeddings
find the smallest cosine distance
if below threshold, reuse that existing ID
Gallery Update

Whenever a track is matched:

its appearance embedding updates the gallery
momentum smoothing keeps the memory stable over time

This gives the system long-term identity consistency.

13. Drawing and Visualization
draw_pose()

Draws:

skeleton edges in yellow
keypoints in blue

Only keypoints above confidence threshold are drawn.

blend_all_masks_once()

Combines all masks into a single overlay before blending.

This is important because blending masks repeatedly one by one can darken the frame and create the black-frame artifact. By blending once:

overlay stays stable
frame brightness remains correct
Visual Elements Drawn Per Person
red segmentation overlay
green bounding box
green text label with:
ID
confidence
pose skeleton and joints
14. Main Workflow

Implemented in main().

Step-by-step runtime flow
1. Load paths

The code defines:

segmentation model path
pose model path
ReID model path
input video path
output video path
2. Open video

The input video is loaded with cv::VideoCapture.

3. Read metadata

The following are extracted:

frame width
frame height
FPS
4. Initialize video writer

If output saving is enabled, a writer is opened for the processed video.

5. Initialize ONNX Runtime

The following sessions are created:

segmentation session
pose session
ReID extractor session
6. Initialize tracker

The tracker is configured, including:

max_age
gallery thresholds
embedding smoothing
7. Frame processing loop

For each frame:

a. Detect persons and masks

Call detectPersonsYOLOv8Seg().

b. For each detection
extract ReID embedding
run pose estimation on crop
c. Associate detections with tracks

Call tracker.step().

d. Draw masks

Call blend_all_masks_once().

e. Draw boxes, IDs, confidence, and pose

Loop over detections and render results.

f. Display frame

Show the processed frame in an OpenCV window.

g. Save frame

If output writer is enabled, write the frame to disk.

h. Logging

Every 50 frames, print progress.

8. Release resources

At the end:

release video capture
release writer
destroy windows
15. Pipeline Summary
End-to-end workflow
Input video
   ↓
Frame read
   ↓
YOLOv8-SEG
   ↓
Person boxes + masks
   ↓
For each detection:
   ├─ OSNet ReID embedding
   └─ YOLOv8-POSE on crop
   ↓
Tracker association
   ├─ IoU matching
   ├─ ReID similarity
   └─ gallery-based ID reuse
   ↓
Visualization
   ├─ mask overlay
   ├─ bounding box
   ├─ ID label
   └─ pose skeleton
   ↓
Display / save output video
16. Important Parameters
Detection
detConfThresh = 0.35
detIouThresh = 0.45

These control:

minimum person confidence
NMS overlap filtering
Pose
poseDetThresh = 0.25
kpDrawThresh = 0.30

These control:

minimum pose candidate confidence
minimum keypoint confidence for drawing
Tracker
max_age = max(default, 3.5 * fps)
iou_gate = 0.05
cos_dist_thresh = 0.35
emb_momentum = 0.9

These control:

how long a track survives without detection
how strict IoU matching is
how strict appearance matching is
how embeddings are smoothed
Gallery
gallery_max_age = 5000
gallery_cos_thresh = 0.35
gallery_momentum = 0.9

These control:

how long gallery identities are kept
how strict long-term ID reuse is
how stable stored gallery embeddings remain
17. Strengths of This Design
combines geometry and appearance for robust tracking
supports long-term ID recovery
separates pose estimation per person crop
provides segmentation, pose, and tracking in one pipeline
supports ONNX deployment with OpenCV and ONNX Runtime
handles common YOLOv8 export variations
18. Limitations and Assumptions
assumes COCO class layout and person = 0
assumes common YOLOv8 ONNX output formats
greedy matching is simpler than Hungarian assignment
pose model output parsing may require adjustment for unusual exports
no batching is used, so performance may drop with many people
full-frame segmentation mask resizing may be costly on large frames
19. Possible Improvements
use letterbox preprocessing instead of direct resize
replace greedy assignment with Hungarian matching
batch ReID and pose inference for multiple people
add motion model such as Kalman filtering
use mask-aware IoU for better association
support multi-class detection if needed
make model paths and thresholds configurable from CLI
add FPS benchmarking and profiling
export tracking results to JSON or CSV
20. File-Level Functional Summary
ReIDExtractor

Responsible for:

preprocessing person crops
running OSNet
returning normalized embeddings
detectPersonsYOLOv8Seg

Responsible for:

frame preprocessing
YOLOv8-seg inference
decoding person detections
building masks
applying NMS
runPoseOnCropYOLOv8

Responsible for:

crop preprocessing
pose ONNX inference
decoding the best keypoint result
Tracker

Responsible for:

managing active tracks
matching detections to tracks
smoothing embeddings
maintaining long-term gallery memory
reusing IDs after disappearance
draw_pose

Responsible for drawing the pose skeleton.

blend_all_masks_once

Responsible for safe mask blending without repeated darkening.

main

Responsible for:

configuration
model loading
video loop
orchestration of all modules
visualization
saving output
21. Conclusion

This code implements a complete human analysis and tracking system that integrates detection, segmentation, pose estimation, re-identification, and long-term tracking into a single ONNX-based C++ application.

Its main advantage is that it does not rely only on box overlap. By combining appearance embeddings with a gallery memory, it can preserve identity much more reliably, even through temporary disappearance, occlusion, or dark frames.

It is a strong baseline for real-world person tracking applications where persistent identity matters.
