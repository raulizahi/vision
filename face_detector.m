/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include "face_detector.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define MAX_DESCRIPTOR_SIZE 80    /* geometric ratio features          */
#define MAX_TRAINING_ENTRIES 1000
#define MATCH_THRESHOLD      1.20f /* Euclidean distance threshold     */
#define RECT_LINE_WIDTH      3.0
#define LABEL_FONT_SIZE     18.0
#define LABEL_PADDING        4.0
#define TILE_OVERLAP         0.5  /* 50 % overlap between adjacent tiles  */
#define DEDUP_IOU_THRESHOLD  0.3  /* IoU above which two boxes are dupes  */

/* ------------------------------------------------------------------ */
/*  Training entry                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char   label[256];
    float  descriptor[MAX_DESCRIPTOR_SIZE];
    int    descriptor_size;          /* actual number of floats stored */
} TrainingEntry;

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static TrainingEntry g_training[MAX_TRAINING_ENTRIES];
static int           g_num_training = 0;
static char          g_db_path[1024] = "training.dat";

/* ------------------------------------------------------------------ */
/*  Training database persistence                                      */
/* ------------------------------------------------------------------ */

static int save_training_db(void)
{
    FILE *fp = fopen(g_db_path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot write %s\n", g_db_path);
        return -1;
    }
    uint32_t n = (uint32_t)g_num_training;
    fwrite(&n, sizeof(n), 1, fp);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t label_len = (uint32_t)strlen(g_training[i].label);
        fwrite(&label_len, sizeof(label_len), 1, fp);
        fwrite(g_training[i].label, 1, label_len, fp);
        uint32_t desc_size = (uint32_t)g_training[i].descriptor_size;
        fwrite(&desc_size, sizeof(desc_size), 1, fp);
        fwrite(g_training[i].descriptor, sizeof(float), desc_size, fp);
    }
    fclose(fp);
    return 0;
}

static int load_training_db(void)
{
    FILE *fp = fopen(g_db_path, "rb");
    if (!fp) return 0;   /* no file yet – not an error */

    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 || n > MAX_TRAINING_ENTRIES) {
        fclose(fp);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t label_len = 0;
        if (fread(&label_len, sizeof(label_len), 1, fp) != 1 ||
            label_len >= sizeof(g_training[i].label)) {
            fclose(fp);
            return -1;
        }
        if (fread(g_training[i].label, 1, label_len, fp) != label_len) {
            fclose(fp);
            return -1;
        }
        g_training[i].label[label_len] = '\0';

        uint32_t desc_size = 0;
        if (fread(&desc_size, sizeof(desc_size), 1, fp) != 1 ||
            desc_size > MAX_DESCRIPTOR_SIZE) {
            fclose(fp);
            return -1;
        }
        if (fread(g_training[i].descriptor, sizeof(float), desc_size, fp) != desc_size) {
            fclose(fp);
            return -1;
        }
        g_training[i].descriptor_size = (int)desc_size;
    }
    g_num_training = (int)n;
    fclose(fp);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Vision helpers                                                     */
/* ------------------------------------------------------------------ */

static CGImageRef load_cgimage(const char *path)
{
    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    NSImage *nsImage = [[NSImage alloc] initWithContentsOfURL:url];
    if (!nsImage) {
        fprintf(stderr, "error: cannot load image %s\n", path);
        return NULL;
    }
    CGImageRef cgImage = [nsImage CGImageForProposedRect:NULL
                                                context:nil
                                                  hints:nil];
    if (cgImage) CGImageRetain(cgImage);
    return cgImage;
}

static NSArray<VNFaceObservation *> *detect_faces(CGImageRef cgImage)
{
    VNDetectFaceLandmarksRequest *request =
        [[VNDetectFaceLandmarksRequest alloc] init];
    request.revision = VNDetectFaceLandmarksRequestRevision3;

    VNImageRequestHandler *handler =
        [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];

    NSError *error = nil;
    [handler performRequests:@[request] error:&error];
    if (error) {
        fprintf(stderr, "error: Vision request failed: %s\n",
                error.localizedDescription.UTF8String);
        return @[];
    }
    return request.results ?: @[];
}

/* ------------------------------------------------------------------ */
/*  Multi-scale (tiled) detection                                      */
/* ------------------------------------------------------------------ */

/*
 * A DetectedFace stores a face observation together with its bounding
 * box remapped to full-image normalized coordinates (0-1).
 */
#define MAX_DETECTED_FACES 256

typedef struct {
    VNFaceObservation * __unsafe_unretained face;
    CGRect  fullBB;          /* bounding box in full-image normalised coords */
} DetectedFace;

static CGFloat iou(CGRect a, CGRect b)
{
    CGFloat x0 = fmax(CGRectGetMinX(a), CGRectGetMinX(b));
    CGFloat y0 = fmax(CGRectGetMinY(a), CGRectGetMinY(b));
    CGFloat x1 = fmin(CGRectGetMaxX(a), CGRectGetMaxX(b));
    CGFloat y1 = fmin(CGRectGetMaxY(a), CGRectGetMaxY(b));
    if (x1 <= x0 || y1 <= y0) return 0.0;
    CGFloat inter = (x1 - x0) * (y1 - y0);
    CGFloat areaA = a.size.width * a.size.height;
    CGFloat areaB = b.size.width * b.size.height;
    return inter / (areaA + areaB - inter);
}

/*
 * Run face detection on the full image and on overlapping half-size
 * tiles (pass 2), collecting all unique faces into `out`.
 * `retainPool` keeps ARC-managed face observations alive across tile
 * iterations — caller must keep the pool alive while using `out`.
 * Returns the number of unique faces found.
 */
static int detect_faces_multiscale(CGImageRef cgImage,
                                   DetectedFace *out, int capacity,
                                   NSMutableArray *retainPool)
{
    int n = 0;
    size_t imgW = CGImageGetWidth(cgImage);
    size_t imgH = CGImageGetHeight(cgImage);

    /* --- Pass 1: full image ---------------------------------------- */
    NSArray<VNFaceObservation *> *fullFaces = detect_faces(cgImage);
    for (VNFaceObservation *face in fullFaces) {
        if (n >= capacity) break;
        [retainPool addObject:face];
        out[n].face   = face;
        out[n].fullBB = face.boundingBox;   /* already in full-image space */
        n++;
    }

    /* --- Pass 2: 2×2 overlapping tiles ----------------------------- */
    CGFloat tileW = imgW / 2.0;
    CGFloat tileH = imgH / 2.0;
    CGFloat stepX = tileW * (1.0 - TILE_OVERLAP);
    CGFloat stepY = tileH * (1.0 - TILE_OVERLAP);

    for (CGFloat ty = 0; ty + tileH <= imgH + 1; ty += stepY) {
        for (CGFloat tx = 0; tx + tileW <= imgW + 1; tx += stepX) {
            CGRect tileRect = CGRectMake(
                (int)tx, (int)ty,
                (int)fmin(tileW, imgW - tx),
                (int)fmin(tileH, imgH - ty));

            if (tileRect.size.width < 64 || tileRect.size.height < 64)
                continue;

            CGImageRef tile = CGImageCreateWithImageInRect(cgImage, tileRect);
            if (!tile) continue;

            NSArray<VNFaceObservation *> *tileFaces = detect_faces(tile);

            for (VNFaceObservation *face in tileFaces) {
                CGRect bb = face.boundingBox;

                /* Remap normalised tile coords → normalised full-image coords */
                CGRect mapped = CGRectMake(
                    (tileRect.origin.x + bb.origin.x * tileRect.size.width)  / imgW,
                    (tileRect.origin.y + bb.origin.y * tileRect.size.height) / imgH,
                    (bb.size.width  * tileRect.size.width)  / imgW,
                    (bb.size.height * tileRect.size.height) / imgH);

                /* Deduplicate: skip if IoU with an existing detection is high */
                int dup = 0;
                for (int i = 0; i < n; i++) {
                    if (iou(out[i].fullBB, mapped) > DEDUP_IOU_THRESHOLD) {
                        dup = 1;
                        break;
                    }
                }
                if (dup) continue;

                if (n >= capacity) break;
                [retainPool addObject:face];
                out[n].face   = face;
                out[n].fullBB = mapped;
                n++;
            }

            CGImageRelease(tile);
        }
    }

    return n;
}

/* ------------------------------------------------------------------ */
/*  Descriptor extraction and matching                                 */
/* ------------------------------------------------------------------ */

/*
 * Helpers for geometric descriptor extraction.
 */
static CGPoint region_centroid(VNFaceLandmarkRegion2D *region)
{
    if (!region || region.pointCount == 0)
        return CGPointMake(0.0, 0.0);
    const CGPoint *pts = region.normalizedPoints;
    CGFloat sx = 0, sy = 0;
    for (NSUInteger i = 0; i < region.pointCount; i++) {
        sx += pts[i].x;
        sy += pts[i].y;
    }
    return CGPointMake(sx / region.pointCount, sy / region.pointCount);
}

static void region_extent(VNFaceLandmarkRegion2D *region,
                          CGFloat *out_w, CGFloat *out_h)
{
    *out_w = 0; *out_h = 0;
    if (!region || region.pointCount < 2) return;
    const CGPoint *pts = region.normalizedPoints;
    CGFloat minX = pts[0].x, maxX = pts[0].x;
    CGFloat minY = pts[0].y, maxY = pts[0].y;
    for (NSUInteger i = 1; i < region.pointCount; i++) {
        if (pts[i].x < minX) minX = pts[i].x;
        if (pts[i].x > maxX) maxX = pts[i].x;
        if (pts[i].y < minY) minY = pts[i].y;
        if (pts[i].y > maxY) maxY = pts[i].y;
    }
    *out_w = maxX - minX;
    *out_h = maxY - minY;
}

static CGFloat pt_dist(CGPoint a, CGPoint b)
{
    CGFloat dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

/*
 * Extract a pose-robust geometric descriptor from facial landmarks.
 * Instead of raw landmark coordinates, we compute:
 *   - Pairwise distances between region centroids, normalized by
 *     inter-ocular distance (IOD)
 *   - Region extents (width/height) normalized by IOD
 * This makes the descriptor largely invariant to face position,
 * scale, and moderately robust to head rotation.
 * Returns number of floats written, or 0 if no landmarks available.
 */
static int extract_descriptor(VNFaceObservation *face, float *descriptor)
{
    VNFaceLandmarks2D *lm = face.landmarks;
    if (!lm) return 0;

    /* Require the two eyes at minimum */
    if (!lm.leftEye || !lm.rightEye) return 0;

    /* Collect centroids of 8 key regions.
     * Use fallbacks for optional regions so descriptor size is constant. */
    CGPoint c[8];
    c[0] = region_centroid(lm.leftEye);
    c[1] = region_centroid(lm.rightEye);
    c[2] = region_centroid(lm.nose       ?: lm.leftEye);
    c[3] = region_centroid(lm.noseCrest   ?: lm.nose ?: lm.leftEye);
    c[4] = region_centroid(lm.outerLips   ?: lm.nose ?: lm.leftEye);
    c[5] = region_centroid(lm.leftEyebrow ?: lm.leftEye);
    c[6] = region_centroid(lm.rightEyebrow?: lm.rightEye);
    c[7] = region_centroid(lm.faceContour ?: lm.nose ?: lm.leftEye);
    int nc = 8;

    /* Inter-ocular distance as the scale-normalizing reference */
    CGFloat iod = pt_dist(c[0], c[1]);
    if (iod < 0.001) return 0;

    int idx = 0;

    /* Pairwise centroid distances: C(8,2) = 28 features */
    for (int i = 0; i < nc; i++)
        for (int j = i + 1; j < nc; j++)
            descriptor[idx++] = (float)(pt_dist(c[i], c[j]) / iod);

    /* Region extents (width, height) normalized by IOD: 6 × 2 = 12 features */
    VNFaceLandmarkRegion2D *ext[] = {
        lm.leftEye, lm.rightEye, lm.nose,
        lm.outerLips, lm.leftEyebrow, lm.rightEyebrow
    };
    for (int i = 0; i < 6; i++) {
        CGFloat w = 0, h = 0;
        if (ext[i]) region_extent(ext[i], &w, &h);
        descriptor[idx++] = (float)(w / iod);
        descriptor[idx++] = (float)(h / iod);
    }

    /* Total: 40 features */
    return idx;
}

static float compute_distance(const float *a, const float *b, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

/*
 * Find the training label with the smallest distance to the given
 * descriptor.  Returns NULL if no match is within MATCH_THRESHOLD.
 */
static const char *find_best_match(const float *descriptor, int size,
                                   float *out_confidence)
{
    float       best_dist  = MATCH_THRESHOLD;
    const char *best_label = NULL;

    for (int i = 0; i < g_num_training; i++) {
        if (g_training[i].descriptor_size != size) continue;
        float dist = compute_distance(descriptor,
                                      g_training[i].descriptor, size);
        if (dist < best_dist) {
            best_dist  = dist;
            best_label = g_training[i].label;
        }
    }
    if (out_confidence) {
        if (best_label) {
            /* Matched: confidence based on how far below threshold */
            *out_confidence = (1.0f - best_dist / MATCH_THRESHOLD) * 100.0f;
        } else {
            /* No match: find closest entry anyway to report confidence */
            float closest_dist = -1.0f;
            for (int i = 0; i < g_num_training; i++) {
                if (g_training[i].descriptor_size != size) continue;
                float dist = compute_distance(descriptor,
                                              g_training[i].descriptor, size);
                if (closest_dist < 0.0f || dist < closest_dist)
                    closest_dist = dist;
            }
            if (closest_dist >= 0.0f)
                *out_confidence = (1.0f - closest_dist / MATCH_THRESHOLD) * 100.0f;
            else
                *out_confidence = 0.0f;
        }
    }
    return best_label;
}

/* ------------------------------------------------------------------ */
/*  Drawing helpers                                                    */
/* ------------------------------------------------------------------ */

static void draw_rect(CGContextRef ctx, CGRect rect,
                       CGFloat r, CGFloat g, CGFloat b)
{
    CGContextSetRGBStrokeColor(ctx, r, g, b, 1.0);
    CGContextSetLineWidth(ctx, RECT_LINE_WIDTH);
    CGContextStrokeRect(ctx, rect);
}

static void draw_label(CGContextRef ctx, const char *text,
                        CGRect faceRect, CGFloat r, CGFloat g, CGFloat b)
{
    /* Create attributed string */
    NSString *nsText = [NSString stringWithUTF8String:text];
    NSDictionary *attrs = @{
        (id)kCTFontAttributeName :
            (__bridge id)CTFontCreateWithName(CFSTR("Helvetica-Bold"),
                                              LABEL_FONT_SIZE, NULL),
        (id)kCTForegroundColorAttributeName :
            (__bridge id)[[NSColor whiteColor] CGColor]
    };
    NSAttributedString *attrStr =
        [[NSAttributedString alloc] initWithString:nsText attributes:attrs];
    CTLineRef line = CTLineCreateWithAttributedString(
        (__bridge CFAttributedStringRef)attrStr);

    /* Measure text */
    CGRect textBounds = CTLineGetBoundsWithOptions(line, 0);

    /* Background rectangle above the face box */
    CGFloat bgX = faceRect.origin.x;
    CGFloat bgY = faceRect.origin.y + faceRect.size.height + 2.0;
    CGFloat bgW = textBounds.size.width + LABEL_PADDING * 2;
    CGFloat bgH = textBounds.size.height + LABEL_PADDING * 2;

    CGContextSetRGBFillColor(ctx, r, g, b, 0.8);
    CGContextFillRect(ctx, CGRectMake(bgX, bgY, bgW, bgH));

    /* Draw text */
    CGContextSetTextPosition(ctx,
        bgX + LABEL_PADDING,
        bgY + LABEL_PADDING - textBounds.origin.y);
    CTLineDraw(line, ctx);
    CFRelease(line);
}

static void draw_landmarks(CGContextRef ctx, VNFaceObservation *face,
                            CGRect bb, size_t imgW, size_t imgH)
{
    VNFaceLandmarks2D *landmarks = face.landmarks;
    if (!landmarks) return;

    /* Draw points for each landmark region with different colors */
    struct { VNFaceLandmarkRegion2D *region; CGFloat r, g, b; } groups[] = {
        { landmarks.leftEye,        0.0, 1.0, 1.0 },  /* cyan    */
        { landmarks.rightEye,       0.0, 1.0, 1.0 },  /* cyan    */
        { landmarks.nose,           1.0, 1.0, 0.0 },  /* yellow  */
        { landmarks.noseCrest,      1.0, 0.8, 0.0 },  /* orange  */
        { landmarks.outerLips,      1.0, 0.4, 0.7 },  /* pink    */
        { landmarks.innerLips,      1.0, 0.4, 0.7 },  /* pink    */
        { landmarks.leftEyebrow,    0.4, 1.0, 0.4 },  /* l-green */
        { landmarks.rightEyebrow,   0.4, 1.0, 0.4 },  /* l-green */
        { landmarks.faceContour,    1.0, 1.0, 1.0 },  /* white   */
        { landmarks.medianLine,     0.6, 0.6, 1.0 },  /* l-blue  */
        { landmarks.leftPupil,      1.0, 0.0, 0.0 },  /* red     */
        { landmarks.rightPupil,     1.0, 0.0, 0.0 },  /* red     */
    };

    int ngroups = sizeof(groups) / sizeof(groups[0]);
    CGFloat dotRadius = 2.5;

    for (int g = 0; g < ngroups; g++) {
        VNFaceLandmarkRegion2D *region = groups[g].region;
        if (!region) continue;

        CGContextSetRGBFillColor(ctx, groups[g].r, groups[g].g, groups[g].b, 1.0);

        const CGPoint *pts = region.normalizedPoints;
        NSUInteger count = region.pointCount;

        for (NSUInteger i = 0; i < count; i++) {
            /* Landmark points are normalized within the face bounding box */
            CGFloat px = (bb.origin.x + pts[i].x * bb.size.width)  * imgW;
            CGFloat py = (bb.origin.y + pts[i].y * bb.size.height) * imgH;

            CGContextFillEllipseInRect(ctx,
                CGRectMake(px - dotRadius, py - dotRadius,
                           dotRadius * 2, dotRadius * 2));
        }
    }
}

static int save_png(CGContextRef ctx, const char *path)
{
    CGImageRef image = CGBitmapContextCreateImage(ctx);
    if (!image) {
        fprintf(stderr, "error: failed to create output image\n");
        return -1;
    }

    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    CFStringRef pngType = (__bridge CFStringRef)UTTypePNG.identifier;
    CGImageDestinationRef dest =
        CGImageDestinationCreateWithURL((__bridge CFURLRef)url,
                                        pngType, 1, NULL);
    if (!dest) {
        fprintf(stderr, "error: cannot create output file %s\n", path);
        CGImageRelease(image);
        return -1;
    }
    CGImageDestinationAddImage(dest, image, NULL);
    bool ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);

    if (!ok) {
        fprintf(stderr, "error: failed to write PNG %s\n", path);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int face_init(const char *training_db_path)
{
    if (training_db_path) {
        strncpy(g_db_path, training_db_path, sizeof(g_db_path) - 1);
        g_db_path[sizeof(g_db_path) - 1] = '\0';
    }
    return load_training_db();
}

int face_train(const char *image_path, const char *label)
{
    @autoreleasepool {
        CGImageRef cgImage = load_cgimage(image_path);
        if (!cgImage) return -1;

        NSMutableArray *retainPool = [NSMutableArray array];
        DetectedFace dfaces[MAX_DETECTED_FACES];
        int nfaces = detect_faces_multiscale(cgImage, dfaces,
                                              MAX_DETECTED_FACES,
                                              retainPool);
        int stored = 0;

        for (int fi = 0; fi < nfaces; fi++) {
            VNFaceObservation *face = dfaces[fi].face;
            if (g_num_training >= MAX_TRAINING_ENTRIES) {
                fprintf(stderr, "warning: training database full\n");
                break;
            }
            TrainingEntry *entry = &g_training[g_num_training];
            int desc_size = extract_descriptor(face, entry->descriptor);
            if (desc_size == 0) continue;

            strncpy(entry->label, label, sizeof(entry->label) - 1);
            entry->label[sizeof(entry->label) - 1] = '\0';
            entry->descriptor_size = desc_size;
            g_num_training++;
            stored++;
        }

        CGImageRelease(cgImage);
        save_training_db();
        return stored;
    }
}

int face_detect(const char *image_path, const char *output_path)
{
    @autoreleasepool {
        CGImageRef cgImage = load_cgimage(image_path);
        if (!cgImage) return -1;

        size_t width  = CGImageGetWidth(cgImage);
        size_t height = CGImageGetHeight(cgImage);

        /* Create bitmap context and draw original image */
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(
            NULL, width, height, 8, width * 4, colorSpace,
            kCGImageAlphaPremultipliedLast);
        CGColorSpaceRelease(colorSpace);

        if (!ctx) {
            fprintf(stderr, "error: cannot create graphics context\n");
            CGImageRelease(cgImage);
            return -1;
        }

        CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), cgImage);

        /* Detect faces using multi-scale tiled approach */
        NSMutableArray *retainPool = [NSMutableArray array];
        DetectedFace detected[MAX_DETECTED_FACES];
        int count = detect_faces_multiscale(cgImage, detected,
                                            MAX_DETECTED_FACES,
                                            retainPool);

        for (int fi = 0; fi < count; fi++) {
            VNFaceObservation *face = detected[fi].face;
            CGRect bb = detected[fi].fullBB;

            /* Convert normalized bounding box to pixel coordinates.
             * Vision: origin at bottom-left, values 0–1.
             * CGContext: origin at bottom-left, values in pixels.  */
            CGRect pixelRect = CGRectMake(
                bb.origin.x * width,
                bb.origin.y * height,
                bb.size.width * width,
                bb.size.height * height);

            /* Try to match against training set */
            float descriptor[MAX_DESCRIPTOR_SIZE];
            int desc_size = extract_descriptor(face, descriptor);
            const char *match = NULL;
            float confidence = 0.0f;
            if (desc_size > 0)
                match = find_best_match(descriptor, desc_size, &confidence);

            /* Draw rectangle and label */
            if (match) {
                char label_buf[320];
                snprintf(label_buf, sizeof(label_buf),
                         "%s (%.0f%%)", match, confidence);
                printf("  Face %d: %s (confidence: %.1f%%)\n",
                       count, match, confidence);
                draw_rect(ctx, pixelRect, 0.0, 1.0, 0.0);   /* green */
                draw_label(ctx, label_buf, pixelRect, 0.0, 0.7, 0.0);
            } else {
                char label_buf[320];
                snprintf(label_buf, sizeof(label_buf),
                         "unknown (%.0f%%)", confidence);
                printf("  Face %d: unknown (confidence: %.1f%%)\n",
                       count, confidence);
                draw_rect(ctx, pixelRect, 1.0, 0.2, 0.2);   /* red   */
                draw_label(ctx, label_buf, pixelRect, 0.8, 0.1, 0.1);
            }

        }

        /* Save annotated image (boxes + labels only) */
        int rc = save_png(ctx, output_path);
        CGContextRelease(ctx);

        /* Create second image with landmark points */
        if (rc == 0 && count > 0) {
            CGColorSpaceRef cs2 = CGColorSpaceCreateDeviceRGB();
            CGContextRef ctx2 = CGBitmapContextCreate(
                NULL, width, height, 8, width * 4, cs2,
                kCGImageAlphaPremultipliedLast);
            CGColorSpaceRelease(cs2);

            if (ctx2) {
                CGContextDrawImage(ctx2,
                    CGRectMake(0, 0, width, height), cgImage);

                for (int fi = 0; fi < count; fi++)
                    draw_landmarks(ctx2, detected[fi].face,
                                   detected[fi].fullBB, width, height);

                /* Build landmarks path: output.png -> output_landmarks.png */
                char lm_path[1024];
                const char *dot = strrchr(output_path, '.');
                if (dot) {
                    size_t base_len = (size_t)(dot - output_path);
                    snprintf(lm_path, sizeof(lm_path), "%.*s_landmarks%s",
                             (int)base_len, output_path, dot);
                } else {
                    snprintf(lm_path, sizeof(lm_path), "%s_landmarks",
                             output_path);
                }

                if (save_png(ctx2, lm_path) == 0)
                    printf("Landmarks image saved to %s\n", lm_path);

                CGContextRelease(ctx2);
            }
        }

        CGImageRelease(cgImage);
        return (rc == 0) ? count : -1;
    }
}

void face_list_training(void)
{
    if (g_num_training == 0) {
        printf("Training database is empty.\n");
        return;
    }
    printf("Training database (%d entries):\n", g_num_training);
    for (int i = 0; i < g_num_training; i++) {
        printf("  [%3d] %-30s  (%d-dim descriptor)\n",
               i, g_training[i].label,
               g_training[i].descriptor_size);
    }
}

int face_reset(void)
{
    g_num_training = 0;
    remove(g_db_path);
    printf("Training database cleared.\n");
    return 0;
}

void face_cleanup(void)
{
    if (g_num_training > 0)
        save_training_db();
}
