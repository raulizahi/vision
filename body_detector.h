/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#ifndef BODY_DETECTOR_H
#define BODY_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Detect human body poses in image_path and draw stick figure
 * overlays with colored joint markers and white connecting rods.
 * Saves the annotated image to output_path (PNG format).
 * Returns the number of bodies found, or -1 on error.
 */
int body_detect(const char *image_path, const char *output_path);

/*
 * Detect body poses in a video file and draw stick figure overlays.
 * Samples frames at `sample_interval` seconds apart.
 * Returns total body detections across all frames, or -1 on error.
 */
int body_detect_video(const char *input_path,
                      const char *output_path,
                      double sample_interval);

#ifdef __cplusplus
}
#endif

#endif /* BODY_DETECTOR_H */
