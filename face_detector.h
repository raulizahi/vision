#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the face detector and load any existing training data.
 * Returns 0 on success, -1 on failure.
 */
int face_init(const char *training_db_path);

/*
 * Add a training image. Detects all faces in the image and stores their
 * landmark descriptors under the given label (person name).
 * Returns the number of faces found and stored, or -1 on error.
 */
int face_train(const char *image_path, const char *label);

/*
 * Detect faces in image_path, match them against the training set,
 * draw labeled rectangles around each face, and save the annotated
 * image to output_path (PNG format).
 * Returns the number of faces found, or -1 on error.
 */
int face_detect(const char *image_path, const char *output_path);

/*
 * Print the current training database entries to stdout.
 */
void face_list_training(void);

/*
 * Clear all training data and delete the database file.
 * Returns 0 on success.
 */
int face_reset(void);

/*
 * Save training data and release resources.
 */
void face_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* FACE_DETECTOR_H */
