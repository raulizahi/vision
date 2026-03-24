/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 *
 * Internal helpers shared between face_detector.m and video_detector.m.
 * Not part of the public C API.
 */

#ifndef FACE_DETECTOR_INTERNAL_H
#define FACE_DETECTOR_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreGraphics/CoreGraphics.h>

#define MAX_DESCRIPTOR_SIZE 80
#define MAX_DETECTED_FACES  256

typedef struct {
    VNFaceObservation * __unsafe_unretained face;
    CGRect  fullBB;  /* bounding box in full-image normalised coords */
} DetectedFace;

/* Multi-scale tiled detection (original + vertical flip) */
int detect_faces_multiscale(CGImageRef cgImage,
                            DetectedFace *out, int capacity,
                            NSMutableArray *retainPool);

/* Descriptor extraction and matching */
int  extract_descriptor(VNFaceObservation *face, float *descriptor);
const char *find_best_match(const float *descriptor, int size,
                            float *out_confidence);

/* Drawing helpers */
void draw_rect(CGContextRef ctx, CGRect rect,
               CGFloat r, CGFloat g, CGFloat b);
void draw_label(CGContextRef ctx, const char *text,
                CGRect faceRect, CGFloat r, CGFloat g, CGFloat b);
void draw_landmarks(CGContextRef ctx, VNFaceObservation *face,
                    CGRect bb, size_t imgW, size_t imgH);

#endif /* FACE_DETECTOR_INTERNAL_H */
