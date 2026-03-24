/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#ifndef GAIT_DETECTOR_H
#define GAIT_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the gait detector and load any existing training data.
 * Returns 0 on success, -1 on failure.
 */
int gait_init(const char *training_db_path);

/*
 * Train on a walking video. Extracts body poses at `sample_interval`
 * seconds apart, computes a gait descriptor, and stores it under `label`.
 * Returns 1 on success, 0 if no gait could be extracted, -1 on error.
 */
int gait_train(const char *video_path, const char *label,
               double sample_interval);

/*
 * Detect and identify people by gait in a video file.
 * Produces an annotated output video with stick figures and identity labels.
 * Returns total identifications, or -1 on error.
 */
int gait_detect_video(const char *input_path, const char *output_path,
                      double sample_interval);

/*
 * Print the current gait training database entries to stdout.
 */
void gait_list_training(void);

/*
 * Clear all gait training data and delete the database file.
 * Returns 0 on success.
 */
int gait_reset(void);

/*
 * Save gait training data and release resources.
 */
void gait_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* GAIT_DETECTOR_H */
