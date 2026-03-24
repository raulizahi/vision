/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#include <stdio.h>
#include <string.h>
#include "face_detector.h"

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
        "  %s list                                   "
            "List training database entries\n"
        "  %s reset                                  "
            "Clear training database\n\n"
        "Options:\n"
        "  --db <path>   Training database file "
            "(default: %s)\n\n"
        "Examples:\n"
        "  %s train alice photo1.jpg photo2.jpg\n"
        "  %s train bob   bob_selfie.png\n"
        "  %s detect group_photo.jpg output.png\n",
        prog, prog, prog, prog, DEFAULT_DB, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    const char *db_path = DEFAULT_DB;
    int arg_start = 1;

    /* Parse --db option */
    if (argc >= 3 && strcmp(argv[1], "--db") == 0) {
        db_path = argv[2];
        arg_start = 3;
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
    return rc;
}
