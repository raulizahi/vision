# Vision — Face Detection, Recognition, Body Pose & Gait Recognition

A lightweight face detection, recognition, body pose estimation, and gait recognition system built on Apple's Vision framework. It uses custom **geometric descriptors** for face and gait recognition, and **`VNDetectHumanBodyPoseRequest`** for body pose detection — no deep learning training required.

## How It Works

### Detection

Faces are detected using `VNDetectFaceLandmarksRequest` (revision 3) from Apple's [Vision framework](https://developer.apple.com/documentation/vision) (`Vision.framework`), which returns bounding boxes and 68+ facial landmark points across ~12 anatomical regions (eyes, eyebrows, nose, lips, face contour, etc.).

#### Multi-Scale Tiled Detection

To detect faces that are small relative to the overall image, detection runs at three scales:

| Pass | Tile Size | Overlap | Effective Magnification |
|------|-----------|---------|------------------------|
| 1 | Full image | — | 1× |
| 2 | W/2 × H/2 | 50% | 2× |
| 3 | W/4 × H/4 | 50% | 4× |

Tile-detected face coordinates are remapped to the original image space, and overlapping detections are deduplicated using an IoU (Intersection over Union) threshold.

#### Vertical-Flip Detection

Detection also runs on a vertically-flipped copy of the image at all three scales. This catches upside-down faces — for example, reflections in mirrors or ceiling-mounted cameras. Flipped detections are mapped back to the original image coordinates and deduplicated against upright detections.

### Recognition Algorithm: Geometric Descriptor

The core of the recognition system is a **40-dimensional geometric descriptor** extracted from facial landmarks. The design makes it invariant to position, scale, and moderately robust to head rotation.

#### Descriptor Composition

| Feature Group | Count | Description |
|---|---|---|
| Pairwise centroid distances | 28 | Distances between centroids of 8 key facial regions: left eye, right eye, nose, nose crest, outer lips, left eyebrow, right eyebrow, face contour. C(8,2) = 28 pairs. |
| Region extents | 12 | Width and height of 6 regions: left eye, right eye, nose, outer lips, left eyebrow, right eyebrow. 6 regions x 2 = 12 features. |
| **Total** | **40** | |

#### Scale Normalization

All distances and extents are **normalized by the inter-ocular distance (IOD)** — the distance between the centroids of the left and right eye. This single normalization step provides:

- **Scale invariance** — the descriptor is the same whether the face is near or far from the camera
- **Rotation robustness** — geometric ratios between facial regions are preserved under moderate head rotation
- **Translation invariance** — centroid-based features are inherently position-independent

#### Matching

Recognition uses **Euclidean distance** in the 40-dimensional descriptor space:

1. Compute the descriptor for the detected face
2. Compare against all descriptors in the training database
3. Find the closest match below the threshold (default: **0.80**)
4. Confidence score: `(1.0 - distance / threshold) * 100%`
5. Detections with negative confidence are discarded
6. If no match falls below the threshold, the face is labeled **unknown**

### Why Geometric Descriptors?

- **Lightweight** — 40 floats per face vs. high-dimensional embeddings from neural networks
- **Fast** — simple arithmetic on landmark coordinates, no GPU needed
- **Interpretable** — every dimension has a clear geometric meaning
- **No training phase** — just store descriptors from labeled images

### Body Pose Estimation

Body poses are detected using `VNDetectHumanBodyPoseRequest`, which returns 19 joint keypoints per person (nose, eyes, ears, neck, shoulders, elbows, wrists, hips, root, knees, ankles). Each keypoint includes x/y coordinates and a confidence score. Joints with confidence below 0.1 are excluded from the visualization.

### Gait Recognition Algorithm

Gait recognition identifies people by their walking pattern using a **24-dimensional gait descriptor** extracted from body pose sequences across video frames.

#### Descriptor Composition

| Feature Group | Count | Description |
|---|---|---|
| Joint angle statistics | 8 | Mean and range of knee, hip, elbow, and shoulder angles (pooled left+right) |
| Bilateral symmetry | 4 | Left/right symmetry ratios for each joint angle |
| Body proportions | 4 | Upper/lower body ratio, shoulder width, hip width, shoulder-to-hip ratio (normalized by body height) |
| Cadence & dynamics | 4 | Step frequency, stride length, vertical bounce amplitude, hip sway (normalized by body height) |
| Posture & regularity | 4 | Arm swing amplitude, forward lean mean/stddev, stride regularity (coefficient of variation) |
| **Total** | **24** | |

#### Invariance

- **Scale**: All spatial measurements normalized by estimated body height (nose to ankle midpoint)
- **Direction**: Left/right joints pooled; symmetry ratios use min/max; horizontal features use absolute values
- **Angle**: Joint angles computed using `acos(dot product)`, inherently direction-free

#### Matching

Same Euclidean distance approach as face recognition, with a separate threshold (default: **0.60**) and a dedicated training database (`gait_training.dat`).

## Building

Requires macOS with Xcode command-line tools installed.

```bash
make
```

This compiles `main.c` (C) and the Objective-C modules (`face_detector.m`, `video_detector.m`, `body_detector.m`, `gait_detector.m`) with ARC, and links against Foundation, Vision, AppKit, CoreGraphics, CoreText, ImageIO, UniformTypeIdentifiers, AVFoundation, CoreMedia, and CoreVideo.

## Usage

### Train

Add labeled face images to the training database. Multiple images per person (different angles) improve accuracy.

```bash
./vision train alice photo1.jpg photo2.jpg
./vision train bob   bob_selfie.png
```

### Detect

Detect and identify faces in an image. Produces two annotated output images:

```bash
./vision detect group_photo.jpg output.png
```

- `output.png` — bounding boxes with identity labels (green = matched, red = unknown) and confidence scores
- `output_landmarks.png` — color-coded landmark points overlaid on the original image

Landmark color coding:
| Region | Color |
|---|---|
| Left/Right eye | Cyan |
| Nose | Yellow |
| Nose crest | Orange |
| Outer/Inner lips | Pink |
| Eyebrows | Light green |
| Face contour | White |
| Median line | Light blue |
| Pupils | Red |

### Detect Video

Detect and identify faces across a video file. Samples frames at a configurable interval, runs multi-scale face detection on each, and produces an annotated output video (MP4/H.264).

```bash
./vision detect-video security.mp4 annotated.mp4
./vision detect-video clip.mov annotated.mp4 --interval 0.5
```

- Supports any input format that macOS can play (`.mp4`, `.mov`, `.m4v`, etc.)
- Output is always MP4 (H.264)
- Default sampling interval is 1 second (configurable with `--interval`)
- Each sampled frame is annotated with bounding boxes and identity labels
- Timestamped detection summary is printed to stdout

### Body Pose Detection

Detect human body poses and overlay stick figures with colored joint markers and white connecting rods.

```bash
./vision body photo.jpg pose.png
```

The output image shows 19 detected joint keypoints connected by a skeleton:

| Joint | Color |
|---|---|
| Nose | Red |
| Eyes | Cyan |
| Ears | Orange |
| Shoulders | Green |
| Neck | Light green |
| Elbows | Blue |
| Wrists | Light blue |
| Hips | Yellow |
| Root (hip center) | Orange |
| Knees | Pink |
| Ankles | Purple |

Joints are connected by white rods forming the skeleton: head, spine, arms, and legs.

### Body Pose Video

Detect body poses across a video file and produce an annotated output video with stick figure overlays.

```bash
./vision body-video dance.mp4 pose.mp4
./vision body-video dance.mp4 pose.mp4 --interval 0.5
```

- Same input format support as `detect-video` (`.mp4`, `.mov`, `.m4v`, etc.)
- Default sampling interval is 1 second (configurable with `--interval`)

### Train Gait

Train gait recognition from a walking video. Samples at 0.1s intervals by default for sufficient temporal resolution.

```bash
./vision train-gait raul walking.mp4
./vision train-gait alice hallway.mov --interval 0.05
```

Requires at least 10 valid pose frames (roughly 1 second of walking). Multiple training videos per person improve accuracy.

### Detect Gait

Identify people by walking pattern in a video. Produces an annotated output video with stick figures and identity labels.

```bash
./vision detect-gait testvideo.mp4 output.mp4
./vision detect-gait testvideo.mp4 output.mp4 --interval 0.1
```

- Default sampling interval is 0.1s (10 fps) for gait analysis
- Output video is rendered at the same sampling rate
- Prints gait match result and confidence to stdout

### Manage Training Data

```bash
./vision list              # Show face training entries
./vision reset             # Clear face training database
./vision gait-list         # Show gait training entries
./vision gait-reset        # Clear gait training database
./vision --db custom.dat train alice photo.jpg        # Custom face database
./vision --gait-db custom.dat train-gait raul walk.mp4  # Custom gait database
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `MATCH_THRESHOLD` | 0.80 | Euclidean distance threshold for recognition. Lower = stricter matching. |
| `MAX_DESCRIPTOR_SIZE` | 80 | Buffer size for descriptor storage (currently 40 floats used). |
| `MAX_TRAINING_ENTRIES` | 1000 | Maximum number of labeled face descriptors. |
| `TILE_OVERLAP` | 0.50 | Overlap ratio between adjacent tiles in multi-scale detection. |
| `DEDUP_IOU_THRESHOLD` | 0.30 | IoU above which two detections are considered duplicates. |
| `GAIT_MATCH_THRESHOLD` | 0.60 | Euclidean distance threshold for gait recognition. |
| `MAX_GAIT_TRAINING` | 200 | Maximum number of gait training entries. |

## Project Structure

```
vision/
  face_detector.h            C interface header (public API)
  face_detector.m            Core implementation (Objective-C)
  face_detector_internal.h   Internal helpers shared across modules
  video_detector.h           Video face detection API header
  video_detector.m           Video face detection (AVFoundation)
  body_detector.h            Body pose detection API header
  body_detector.m            Body pose detection and stick figure drawing
  gait_detector.h            Gait recognition API header
  gait_detector.m            Gait descriptor extraction and matching
  main.c                     Command-line interface (C)
  Makefile                   Build configuration
```

### Not included in the repo (generated locally)

- **`training.dat`** — Binary face training database, auto-generated when you run `./vision train`. Contains serialized label + descriptor pairs.
- **`gait_training.dat`** — Binary gait training database, auto-generated when you run `./vision train-gait`.
- **`training-set/`** — Directory for training images. Create this directory and populate it with labeled face images at various angles (e.g., center, left, right) for best results.

To get started, create a `training-set/` folder and train the model:

```bash
mkdir training-set
# Add your face images, then:
./vision train yourname training-set/photo1.jpg training-set/photo2.jpg
```

## Enhancements & Design Decisions

1. **Fallback regions** — If an optional landmark region (e.g., nose crest) is not detected, the algorithm falls back to nearby regions so the descriptor size remains constant at 40 dimensions.

2. **Dual output images** — Detection produces both a labeled bounding-box image and a separate landmark visualization, useful for debugging and understanding what the detector sees.

3. **Confidence reporting** — Even for unmatched faces, the system reports a confidence score relative to the closest training entry, giving insight into near-misses.

4. **Persistent binary database** — Training descriptors are serialized to a compact binary format (`training.dat`) and automatically loaded on startup.

5. **Multi-pose training** — Training with images from multiple angles (center, left, right) captures the natural variation in geometric ratios under head rotation, improving recognition robustness.

6. **Pure C interface** — The Objective-C implementation is hidden behind a clean C API (`face_detector.h`), making it easy to integrate into C/C++ projects or wrap with FFI bindings.

7. **Multi-scale tiled detection** — Automatically detects small faces by running Vision at multiple tile resolutions (full, half, quarter), with coordinate remapping and IoU-based deduplication.

8. **Vertical-flip detection** — Runs detection on a vertically-flipped copy of the image to catch upside-down faces such as mirror reflections, doubling the effective detection coverage.

9. **Gait recognition** — Identifies people by walking pattern using a 24-dimensional descriptor computed from body pose sequences, invariant to walking direction, scale, and camera distance.
