/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "face_detector.h"
#include "video_detector.h"
#include "body_detector.h"
#include "gait_detector.h"

#define DEFAULT_DB "training.dat"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Face detection and recognition using Apple Vision framework.\n\n"
        "Usage:\n"
        "  %s train  <label> <image> [image2 ...]   "
            "Train on face(s) in image(s)\n"
        "  %s detect <image> <output.png>            "
            "Detect & identify faces, save annotated image\n"
        "  %s detect-video <video> <output.mp4>      "
            "Detect faces in video, save annotated video\n"
        "  %s body <image> <output.png>              "
            "Detect body poses, draw stick figures\n"
        "  %s body-video <video> <output.mp4>        "
            "Detect body poses in video, draw stick figures\n"
        "  %s train-gait <label> <video>             "
            "Train gait from walking video\n"
        "  %s detect-gait <video> <output.mp4>       "
            "Identify person by gait in video\n"
        "  %s gait-list                              "
            "List gait training database entries\n"
        "  %s gait-reset                             "
            "Clear gait training database\n"
        "  %s list                                   "
            "List face training database entries\n"
        "  %s reset                                  "
            "Clear face training database\n\n"
        "Options:\n"
        "  --db <path>       Face training database file "
            "(default: %s)\n"
        "  --gait-db <path>  Gait training database file "
            "(default: gait_training.dat)\n"
        "  --interval <sec>  Sampling interval for video commands "
            "(default: 1.0, gait default: 0.1)\n\n"
        "Examples:\n"
        "  %s train alice photo1.jpg photo2.jpg\n"
        "  %s train bob   bob_selfie.png\n"
        "  %s detect group_photo.jpg output.png\n"
        "  %s detect-video security.mp4 annotated.mp4\n"
        "  %s detect-video security.mp4 annotated.mp4 --interval 0.5\n"
        "  %s body photo.jpg pose.png\n"
        "  %s body-video dance.mp4 pose.mp4\n"
        "  %s train-gait raul walking.mp4\n"
        "  %s detect-gait testvideo.mp4 output.mp4\n",
        prog, prog, prog, prog, prog, prog, prog, prog, prog,
        prog, prog, DEFAULT_DB,
        prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    const char *db_path = DEFAULT_DB;
    const char *gait_db_path = "gait_training.dat";
    int arg_start = 1;

    /* Parse options */
    while (arg_start + 1 < argc) {
        if (strcmp(argv[arg_start], "--db") == 0) {
            db_path = argv[arg_start + 1];
            arg_start += 2;
        } else if (strcmp(argv[arg_start], "--gait-db") == 0) {
            gait_db_path = argv[arg_start + 1];
            arg_start += 2;
        } else {
            break;
        }
    }

    if (argc - arg_start < 1) {
        usage(argv[0]);
        return 1;
    }

    const char *command = argv[arg_start];

    if (face_init(db_path) != 0) {
        fprintf(stderr, "error: failed to initialize face detector\n");
        return 1;
    }
    if (gait_init(gait_db_path) != 0) {
        fprintf(stderr, "error: failed to initialize gait detector\n");
        return 1;
    }

    int rc = 0;

    if (strcmp(command, "train") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: train requires a label and at least "
                            "one image path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *label = argv[arg_start + 1];
            int total = 0;
            for (int i = arg_start + 2; i < argc; i++) {
                int n = face_train(argv[i], label);
                if (n < 0) {
                    fprintf(stderr, "warning: failed to process %s\n",
                            argv[i]);
                } else if (n == 0) {
                    fprintf(stderr, "warning: no faces found in %s\n",
                            argv[i]);
                } else {
                    printf("Stored %d face(s) from %s as '%s'\n",
                           n, argv[i], label);
                    total += n;
                }
            }
            printf("Total: %d face descriptor(s) stored for '%s'\n",
                   total, label);
        }

    } else if (strcmp(command, "detect") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: detect requires an image path and "
                            "output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input  = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            int n = face_detect(input, output);
            if (n < 0) {
                fprintf(stderr, "error: detection failed\n");
                rc = 1;
            } else {
                printf("Found %d face(s). Annotated image saved to %s\n",
                       n, output);
            }
        }

    } else if (strcmp(command, "detect-video") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: detect-video requires a video path and "
                            "output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input  = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 1.0;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            int n = face_detect_video(input, output, interval);
            if (n < 0) {
                fprintf(stderr, "error: video detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "body") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: body requires an image path and "
                            "output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input  = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            int n = body_detect(input, output);
            if (n < 0) {
                fprintf(stderr, "error: body detection failed\n");
                rc = 1;
            } else {
                printf("Found %d body/bodies. Annotated image saved to %s\n",
                       n, output);
            }
        }

    } else if (strcmp(command, "body-video") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: body-video requires a video path and "
                            "output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input  = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 1.0;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            int n = body_detect_video(input, output, interval);
            if (n < 0) {
                fprintf(stderr, "error: body video detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "train-gait") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: train-gait requires a label and "
                            "a video path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *label = argv[arg_start + 1];
            const char *video = argv[arg_start + 2];
            double interval = 0.1;  /* denser sampling for gait */
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            int n = gait_train(video, label, interval);
            if (n <= 0) {
                fprintf(stderr, "error: gait training failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "detect-gait") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: detect-gait requires a video path and "
                            "output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input  = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 0.1;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            int n = gait_detect_video(input, output, interval);
            if (n < 0) {
                fprintf(stderr, "error: gait detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "gait-list") == 0) {
        gait_list_training();

    } else if (strcmp(command, "gait-reset") == 0) {
        gait_reset();

    } else if (strcmp(command, "list") == 0) {
        face_list_training();

    } else if (strcmp(command, "reset") == 0) {
        face_reset();

    } else {
        fprintf(stderr, "error: unknown command '%s'\n", command);
        usage(argv[0]);
        rc = 1;
    }

    face_cleanup();
    gait_cleanup();
    return rc;
}
