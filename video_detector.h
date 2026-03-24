/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#ifndef VIDEO_DETECTOR_H
#define VIDEO_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Detect and identify faces in a video file.
 * Samples frames at `sample_interval` seconds apart, runs multi-scale
 * face detection on each, and writes an annotated output video (MP4).
 * Returns total number of face detections across all frames, or -1 on error.
 */
int face_detect_video(const char *input_path,
                      const char *output_path,
                      double sample_interval);

#ifdef __cplusplus
}
#endif

#endif /* VIDEO_DETECTOR_H */
