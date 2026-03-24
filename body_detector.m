/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#include "body_detector.h"
#include "face_detector_internal.h"
#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Joint / bone definitions (runtime-initialised)                     */
/* ------------------------------------------------------------------ */

#define MAX_JOINTS 19
#define MAX_BONES  18

typedef struct {
    VNHumanBodyPoseObservationJointName __unsafe_unretained jointName;
    CGFloat r, g, b;
    const char *label;
} JointColor;

typedef struct {
    VNHumanBodyPoseObservationJointName __unsafe_unretained from;
    VNHumanBodyPoseObservationJointName __unsafe_unretained to;
} BoneConnection;

static JointColor     joint_colors[MAX_JOINTS];
static int            num_joint_colors = 0;
static BoneConnection bones[MAX_BONES];
static int            num_bones = 0;

static void init_skeleton(void)
{
    static int done = 0;
    if (done) return;
    done = 1;

    /* Joint colors */
    int j = 0;
#define J(name, R, G, B, lbl) do { \
    joint_colors[j].jointName = (name); \
    joint_colors[j].r = (R); joint_colors[j].g = (G); joint_colors[j].b = (B); \
    joint_colors[j].label = (lbl); j++; } while(0)

    J(VNHumanBodyPoseObservationJointNameNose,           1.0, 0.2, 0.2, "nose");
    J(VNHumanBodyPoseObservationJointNameLeftEye,        0.0, 1.0, 1.0, "l-eye");
    J(VNHumanBodyPoseObservationJointNameRightEye,       0.0, 1.0, 1.0, "r-eye");
    J(VNHumanBodyPoseObservationJointNameLeftEar,        1.0, 0.8, 0.0, "l-ear");
    J(VNHumanBodyPoseObservationJointNameRightEar,       1.0, 0.8, 0.0, "r-ear");
    J(VNHumanBodyPoseObservationJointNameLeftShoulder,   0.0, 1.0, 0.0, "l-shoulder");
    J(VNHumanBodyPoseObservationJointNameRightShoulder,  0.0, 1.0, 0.0, "r-shoulder");
    J(VNHumanBodyPoseObservationJointNameNeck,           0.4, 1.0, 0.4, "neck");
    J(VNHumanBodyPoseObservationJointNameLeftElbow,      0.0, 0.6, 1.0, "l-elbow");
    J(VNHumanBodyPoseObservationJointNameRightElbow,     0.0, 0.6, 1.0, "r-elbow");
    J(VNHumanBodyPoseObservationJointNameLeftWrist,      0.4, 0.4, 1.0, "l-wrist");
    J(VNHumanBodyPoseObservationJointNameRightWrist,     0.4, 0.4, 1.0, "r-wrist");
    J(VNHumanBodyPoseObservationJointNameLeftHip,        1.0, 1.0, 0.0, "l-hip");
    J(VNHumanBodyPoseObservationJointNameRightHip,       1.0, 1.0, 0.0, "r-hip");
    J(VNHumanBodyPoseObservationJointNameRoot,           1.0, 0.5, 0.0, "root");
    J(VNHumanBodyPoseObservationJointNameLeftKnee,       1.0, 0.4, 0.7, "l-knee");
    J(VNHumanBodyPoseObservationJointNameRightKnee,      1.0, 0.4, 0.7, "r-knee");
    J(VNHumanBodyPoseObservationJointNameLeftAnkle,      0.6, 0.2, 1.0, "l-ankle");
    J(VNHumanBodyPoseObservationJointNameRightAnkle,     0.6, 0.2, 1.0, "r-ankle");
    num_joint_colors = j;
#undef J

    /* Bone connections */
    int b = 0;
#define B(f, t) do { bones[b].from = (f); bones[b].to = (t); b++; } while(0)

    /* Head */
    B(VNHumanBodyPoseObservationJointNameNose,          VNHumanBodyPoseObservationJointNameLeftEye);
    B(VNHumanBodyPoseObservationJointNameNose,          VNHumanBodyPoseObservationJointNameRightEye);
    B(VNHumanBodyPoseObservationJointNameLeftEye,       VNHumanBodyPoseObservationJointNameLeftEar);
    B(VNHumanBodyPoseObservationJointNameRightEye,      VNHumanBodyPoseObservationJointNameRightEar);
    /* Neck to head and shoulders */
    B(VNHumanBodyPoseObservationJointNameNeck,          VNHumanBodyPoseObservationJointNameNose);
    B(VNHumanBodyPoseObservationJointNameNeck,          VNHumanBodyPoseObservationJointNameLeftShoulder);
    B(VNHumanBodyPoseObservationJointNameNeck,          VNHumanBodyPoseObservationJointNameRightShoulder);
    /* Arms */
    B(VNHumanBodyPoseObservationJointNameLeftShoulder,  VNHumanBodyPoseObservationJointNameLeftElbow);
    B(VNHumanBodyPoseObservationJointNameLeftElbow,     VNHumanBodyPoseObservationJointNameLeftWrist);
    B(VNHumanBodyPoseObservationJointNameRightShoulder, VNHumanBodyPoseObservationJointNameRightElbow);
    B(VNHumanBodyPoseObservationJointNameRightElbow,    VNHumanBodyPoseObservationJointNameRightWrist);
    /* Spine */
    B(VNHumanBodyPoseObservationJointNameNeck,          VNHumanBodyPoseObservationJointNameRoot);
    /* Hips */
    B(VNHumanBodyPoseObservationJointNameRoot,          VNHumanBodyPoseObservationJointNameLeftHip);
    B(VNHumanBodyPoseObservationJointNameRoot,          VNHumanBodyPoseObservationJointNameRightHip);
    /* Legs */
    B(VNHumanBodyPoseObservationJointNameLeftHip,       VNHumanBodyPoseObservationJointNameLeftKnee);
    B(VNHumanBodyPoseObservationJointNameLeftKnee,      VNHumanBodyPoseObservationJointNameLeftAnkle);
    B(VNHumanBodyPoseObservationJointNameRightHip,      VNHumanBodyPoseObservationJointNameRightKnee);
    B(VNHumanBodyPoseObservationJointNameRightKnee,     VNHumanBodyPoseObservationJointNameRightAnkle);
    num_bones = b;
#undef B
}

/* ------------------------------------------------------------------ */
/*  Helper: get joint position in pixel coordinates                    */
/* ------------------------------------------------------------------ */

static int get_joint_pixel(VNHumanBodyPoseObservation *body,
                           VNHumanBodyPoseObservationJointName name,
                           size_t imgW, size_t imgH,
                           CGPoint *out)
{
    NSError *err = nil;
    VNRecognizedPoint *pt =
        [body recognizedPointForJointName:name error:&err];
    if (!pt || pt.confidence < 0.1) return 0;

    /* Vision: normalised, bottom-left origin -> pixel coords */
    out->x = pt.location.x * imgW;
    out->y = pt.location.y * imgH;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Detect & draw bodies on a CGImage into a context                   */
/* ------------------------------------------------------------------ */

static int detect_and_draw_bodies(CGImageRef cgImage, CGContextRef ctx,
                                  size_t width, size_t height,
                                  int verbose)
{
    init_skeleton();

    VNDetectHumanBodyPoseRequest *request =
        [[VNDetectHumanBodyPoseRequest alloc] init];
    VNImageRequestHandler *handler =
        [[VNImageRequestHandler alloc] initWithCGImage:cgImage
                                              options:@{}];
    NSError *error = nil;
    [handler performRequests:@[request] error:&error];
    if (error) {
        fprintf(stderr, "error: body pose detection failed: %s\n",
                error.localizedDescription.UTF8String);
        return -1;
    }

    NSArray<VNHumanBodyPoseObservation *> *bodies = request.results;
    int count = (int)bodies.count;

    CGFloat rodWidth = 3.0;
    CGFloat dotRadius = 12.0;

    for (int bi = 0; bi < count; bi++) {
        VNHumanBodyPoseObservation *body = bodies[bi];

        if (verbose) printf("  Body %d:\n", bi + 1);

        /* Draw white connecting rods first (behind joints) */
        CGContextSetRGBStrokeColor(ctx, 1.0, 1.0, 1.0, 0.9);
        CGContextSetLineWidth(ctx, rodWidth);
        CGContextSetLineCap(ctx, kCGLineCapRound);

        for (int i = 0; i < num_bones; i++) {
            CGPoint from, to;
            if (!get_joint_pixel(body, bones[i].from,
                                 width, height, &from))
                continue;
            if (!get_joint_pixel(body, bones[i].to,
                                 width, height, &to))
                continue;

            CGContextMoveToPoint(ctx, from.x, from.y);
            CGContextAddLineToPoint(ctx, to.x, to.y);
            CGContextStrokePath(ctx);
        }

        /* Draw colored joint dots on top */
        for (int i = 0; i < num_joint_colors; i++) {
            CGPoint pt;
            if (!get_joint_pixel(body, joint_colors[i].jointName,
                                 width, height, &pt))
                continue;

            CGContextSetRGBFillColor(ctx,
                joint_colors[i].r, joint_colors[i].g,
                joint_colors[i].b, 1.0);
            CGContextFillEllipseInRect(ctx,
                CGRectMake(pt.x - dotRadius, pt.y - dotRadius,
                           dotRadius * 2, dotRadius * 2));

            if (verbose)
                printf("    %-12s (%.0f, %.0f)\n",
                       joint_colors[i].label, pt.x, pt.y);
        }
    }

    return count;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

static void format_time(double seconds, char *buf, size_t buflen)
{
    int h = (int)(seconds / 3600);
    int m = (int)(fmod(seconds, 3600) / 60);
    double s = fmod(seconds, 60);
    snprintf(buf, buflen, "%02d:%02d:%06.3f", h, m, s);
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int body_detect(const char *image_path, const char *output_path)
{
    @autoreleasepool {
        CGImageRef cgImage = load_cgimage(image_path);
        if (!cgImage) return -1;

        size_t width  = CGImageGetWidth(cgImage);
        size_t height = CGImageGetHeight(cgImage);

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

        int count = detect_and_draw_bodies(cgImage, ctx, width, height, 1);

        int rc = save_png(ctx, output_path);
        CGContextRelease(ctx);
        CGImageRelease(cgImage);

        return (rc == 0) ? count : -1;
    }
}

int body_detect_video(const char *input_path,
                      const char *output_path,
                      double sample_interval)
{
    @autoreleasepool {
        /* --- Load input video -------------------------------------- */
        NSString *inPath = [NSString stringWithUTF8String:input_path];
        NSURL *inURL = [NSURL fileURLWithPath:inPath];
        AVURLAsset *asset = [AVURLAsset assetWithURL:inURL];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [asset loadValuesAsynchronouslyForKeys:@[@"tracks", @"duration"]
                             completionHandler:^{ dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        NSArray<AVAssetTrack *> *videoTracks =
            [asset tracksWithMediaType:AVMediaTypeVideo];
        if (videoTracks.count == 0) {
            fprintf(stderr, "error: no video track in %s\n", input_path);
            return -1;
        }
        AVAssetTrack *track = videoTracks[0];
        CGSize naturalSize = track.naturalSize;
        CGAffineTransform txf = track.preferredTransform;
        CGSize outputSize = CGSizeApplyAffineTransform(naturalSize, txf);
        size_t width  = (size_t)fabs(outputSize.width);
        size_t height = (size_t)fabs(outputSize.height);

        double durationSecs = CMTimeGetSeconds(asset.duration);
        printf("Video: %zux%zu, %.1f seconds\n", width, height, durationSecs);

        /* --- Configure frame generator ----------------------------- */
        AVAssetImageGenerator *generator =
            [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        generator.appliesPreferredTrackTransform = YES;
        generator.requestedTimeToleranceBefore = kCMTimeZero;
        generator.requestedTimeToleranceAfter  = kCMTimeZero;

        /* --- Configure video writer -------------------------------- */
        NSString *outPath = [NSString stringWithUTF8String:output_path];
        NSURL *outURL = [NSURL fileURLWithPath:outPath];
        [[NSFileManager defaultManager] removeItemAtURL:outURL error:nil];

        NSError *error = nil;
        AVAssetWriter *writer =
            [AVAssetWriter assetWriterWithURL:outURL
                                    fileType:AVFileTypeMPEG4
                                       error:&error];
        if (!writer) {
            fprintf(stderr, "error: cannot create writer: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        NSDictionary *videoSettings = @{
            AVVideoCodecKey  : AVVideoCodecTypeH264,
            AVVideoWidthKey  : @(width),
            AVVideoHeightKey : @(height),
        };
        AVAssetWriterInput *writerInput =
            [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                              outputSettings:videoSettings];
        writerInput.expectsMediaDataInRealTime = NO;

        NSDictionary *pbAttrs = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey :
                @(kCVPixelFormatType_32ARGB),
            (NSString *)kCVPixelBufferWidthKey  : @(width),
            (NSString *)kCVPixelBufferHeightKey : @(height),
        };
        AVAssetWriterInputPixelBufferAdaptor *adaptor =
            [AVAssetWriterInputPixelBufferAdaptor
                assetWriterInputPixelBufferAdaptorWithAssetWriterInput:writerInput
                                          sourcePixelBufferAttributes:pbAttrs];

        [writer addInput:writerInput];
        [writer startWriting];
        [writer startSessionAtSourceTime:kCMTimeZero];

        /* --- Process frames ---------------------------------------- */
        int totalDetections = 0;
        int frameIndex = 0;

        for (double t = 0; t < durationSecs; t += sample_interval) {
            @autoreleasepool {
                CMTime time = CMTimeMakeWithSeconds(t, 600);
                CMTime actual;
                NSError *frameErr = nil;

                CGImageRef frame =
                    [generator copyCGImageAtTime:time
                                      actualTime:&actual
                                           error:&frameErr];
                if (!frame) {
                    fprintf(stderr, "warning: cannot read frame at %.1fs: %s\n",
                            t, frameErr.localizedDescription.UTF8String);
                    continue;
                }

                CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
                CGContextRef ctx = CGBitmapContextCreate(
                    NULL, width, height, 8, width * 4, cs,
                    kCGImageAlphaPremultipliedLast);
                CGColorSpaceRelease(cs);

                if (!ctx) {
                    CGImageRelease(frame);
                    continue;
                }

                CGContextDrawImage(ctx,
                    CGRectMake(0, 0, width, height), frame);

                char timeBuf[32];
                format_time(t, timeBuf, sizeof(timeBuf));

                int count = detect_and_draw_bodies(frame, ctx,
                                                    width, height, 0);
                if (count > 0) {
                    printf("  [%s] %d body/bodies in frame %d\n",
                           timeBuf, count, frameIndex + 1);
                    totalDetections += count;
                }

                /* Write annotated frame to output video */
                CGImageRef annotated = CGBitmapContextCreateImage(ctx);
                CGContextRelease(ctx);

                if (annotated) {
                    while (!writerInput.readyForMoreMediaData)
                        [NSThread sleepForTimeInterval:0.01];

                    CVPixelBufferRef pixBuf = NULL;
                    CVReturn cvRet = CVPixelBufferPoolCreatePixelBuffer(
                        NULL, adaptor.pixelBufferPool, &pixBuf);

                    if (cvRet == kCVReturnSuccess && pixBuf) {
                        CVPixelBufferLockBaseAddress(pixBuf, 0);
                        void *base = CVPixelBufferGetBaseAddress(pixBuf);
                        size_t bpr  = CVPixelBufferGetBytesPerRow(pixBuf);

                        CGColorSpaceRef cs2 = CGColorSpaceCreateDeviceRGB();
                        CGContextRef pbCtx = CGBitmapContextCreate(
                            base, width, height, 8, bpr, cs2,
                            kCGImageAlphaPremultipliedFirst);
                        CGColorSpaceRelease(cs2);

                        if (pbCtx) {
                            CGContextDrawImage(pbCtx,
                                CGRectMake(0, 0, width, height), annotated);
                            CGContextRelease(pbCtx);
                        }

                        CVPixelBufferUnlockBaseAddress(pixBuf, 0);

                        CMTime pts = CMTimeMakeWithSeconds(t, 600);
                        [adaptor appendPixelBuffer:pixBuf
                              withPresentationTime:pts];
                        CVPixelBufferRelease(pixBuf);
                    }

                    CGImageRelease(annotated);
                }

                CGImageRelease(frame);
                frameIndex++;
            }
        }

        /* --- Finalize output --------------------------------------- */
        [writerInput markAsFinished];

        dispatch_semaphore_t doneSem = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{
            dispatch_semaphore_signal(doneSem);
        }];
        dispatch_semaphore_wait(doneSem, DISPATCH_TIME_FOREVER);

        if (writer.status != AVAssetWriterStatusCompleted) {
            fprintf(stderr, "error: video write failed: %s\n",
                    writer.error.localizedDescription.UTF8String);
            return -1;
        }

        printf("Processed %d frames, %d total body detection(s).\n",
               frameIndex, totalDetections);
        printf("Output saved to %s\n", output_path);

        return totalDetections;
    }
}
