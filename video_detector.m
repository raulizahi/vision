/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>
#import <AppKit/AppKit.h>
#include "video_detector.h"
#include "face_detector.h"
#include "face_detector_internal.h"
#include <stdio.h>

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

int face_detect_video(const char *input_path,
                      const char *output_path,
                      double sample_interval)
{
    @autoreleasepool {
        /* --- Load input video -------------------------------------- */
        NSString *inPath = [NSString stringWithUTF8String:input_path];
        NSURL *inURL = [NSURL fileURLWithPath:inPath];
        AVURLAsset *asset = [AVURLAsset assetWithURL:inURL];

        /* Wait for the asset to load its tracks */
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

        /* Apply transform to get actual output dimensions */
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

        /* Remove existing file if present */
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

                /* Create annotation context at output dimensions */
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

                /* Detect faces */
                NSMutableArray *retainPool = [NSMutableArray array];
                DetectedFace detected[MAX_DETECTED_FACES];
                int count = detect_faces_multiscale(frame, detected,
                                                    MAX_DETECTED_FACES,
                                                    retainPool);

                char timeBuf[32];
                format_time(t, timeBuf, sizeof(timeBuf));

                int drawn = 0;
                for (int fi = 0; fi < count; fi++) {
                    VNFaceObservation *face = detected[fi].face;
                    CGRect bb = detected[fi].fullBB;

                    CGRect pixelRect = CGRectMake(
                        bb.origin.x * width,
                        bb.origin.y * height,
                        bb.size.width * width,
                        bb.size.height * height);

                    float descriptor[MAX_DESCRIPTOR_SIZE];
                    int desc_size = extract_descriptor(face, descriptor);
                    const char *match = NULL;
                    float confidence = 0.0f;
                    if (desc_size > 0)
                        match = find_best_match(descriptor, desc_size,
                                                &confidence);

                    if (confidence < 0.0f) continue;

                    drawn++;
                    if (match) {
                        char label_buf[320];
                        if (g_runner_up_label)
                            snprintf(label_buf, sizeof(label_buf),
                                     "%s (%.0f%%) | %s %.0f%%",
                                     match, confidence,
                                     g_runner_up_label, g_runner_up_confidence);
                        else
                            snprintf(label_buf, sizeof(label_buf),
                                     "%s (%.0f%%)", match, confidence);
                        printf("  [%s] Face %d: %s (confidence: %.1f%%)\n",
                               timeBuf, drawn, match, confidence);
                        draw_rect(ctx, pixelRect, 0.0, 1.0, 0.0);
                        draw_label(ctx, label_buf, pixelRect, 0.0, 0.7, 0.0);
                    } else {
                        char label_buf[320];
                        if (g_runner_up_label)
                            snprintf(label_buf, sizeof(label_buf),
                                     "unknown (%.0f%%) | %s %.0f%%",
                                     confidence,
                                     g_runner_up_label, g_runner_up_confidence);
                        else
                            snprintf(label_buf, sizeof(label_buf),
                                     "unknown (%.0f%%)", confidence);
                        printf("  [%s] Face %d: unknown (confidence: %.1f%%)\n",
                               timeBuf, drawn, confidence);
                        draw_rect(ctx, pixelRect, 1.0, 0.2, 0.2);
                        draw_label(ctx, label_buf, pixelRect, 0.8, 0.1, 0.1);
                    }
                }

                totalDetections += drawn;

                /* Write annotated frame to output video */
                CGImageRef annotated = CGBitmapContextCreateImage(ctx);
                CGContextRelease(ctx);

                if (annotated) {
                    /* Wait until the writer input is ready */
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

                if (drawn > 0) {
                    printf("  [%s] %d face(s) in frame %d\n",
                           timeBuf, drawn, frameIndex);
                }
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

        printf("Processed %d frames, %d total face detection(s).\n",
               frameIndex, totalDetections);
        printf("Output saved to %s\n", output_path);

        return totalDetections;
    }
}
