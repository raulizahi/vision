# Vision — Face Detection & Recognition

A lightweight face detection and recognition system built on Apple's Vision framework. It uses a custom **pose-robust geometric descriptor** for face recognition — no deep learning required.

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

## Building

Requires macOS with Xcode command-line tools installed.

```bash
make
```

This compiles `main.c` (C) and `face_detector.m` (Objective-C with ARC) and links against Foundation, Vision, AppKit, CoreGraphics, CoreText, ImageIO, and UniformTypeIdentifiers.

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

### Manage Training Data

```bash
./vision list              # Show all training entries
./vision reset             # Clear the database
./vision --db custom.dat train alice photo.jpg   # Use a custom database file
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `MATCH_THRESHOLD` | 0.80 | Euclidean distance threshold for recognition. Lower = stricter matching. |
| `MAX_DESCRIPTOR_SIZE` | 80 | Buffer size for descriptor storage (currently 40 floats used). |
| `MAX_TRAINING_ENTRIES` | 1000 | Maximum number of labeled face descriptors. |
| `TILE_OVERLAP` | 0.50 | Overlap ratio between adjacent tiles in multi-scale detection. |
| `DEDUP_IOU_THRESHOLD` | 0.30 | IoU above which two detections are considered duplicates. |

## Project Structure

```
vision/
  face_detector.h     C interface header (public API)
  face_detector.m     Core implementation (Objective-C)
  main.c              Command-line interface (C)
  Makefile            Build configuration
```

### Not included in the repo (generated locally)

- **`training.dat`** — Binary training database, auto-generated when you run `./vision train`. Contains serialized label + descriptor pairs.
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
