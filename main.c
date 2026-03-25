/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "body_detector.h"
#include "build_version.h"
#include "face_detector.h"
#include "gait_detector.h"
#include "video_detector.h"

#define DEFAULT_DB "training.dat"
#define DEFAULT_GAIT_DB "gait_training.dat"
#define MAX_EVAL_LABELS 512
#define MAX_LABEL_LEN 256
#define MAX_PATH_LEN 4096

typedef int (*PredictFn)(const char *path, double interval,
                         char *label_buf, size_t label_buf_size,
                         float *out_confidence, int *out_observations);

static void usage(const char *prog)
{
    fprintf(stderr,
        "Face detection and recognition using Apple Vision framework.\n\n"
        "App version: %s\n"
        "Build version: %s\n\n"
        "Usage:\n"
        "  %s train <label> <image> [image2 ...]      Train on face(s) in image(s)\n"
        "  %s detect <image> <output.png>             Detect & identify faces, save annotated image\n"
        "  %s detect-video <video> <output.mp4>       Detect faces in video, save annotated video\n"
        "  %s body <image> <output.png>               Detect body poses, draw stick figures\n"
        "  %s body-video <video> <output.mp4>         Detect body poses in video, draw stick figures\n"
        "  %s train-gait <label> <video>              Train gait from walking video\n"
        "  %s detect-gait <video> <output.mp4>        Identify person by gait in video\n"
        "  %s eval-face <dataset_dir> <report_dir>    Build face confusion-matrix report\n"
        "  %s eval-gait <dataset_dir> <report_dir>    Build gait confusion-matrix report\n"
        "  %s version                                 Print progressive app version\n"
        "  %s build-version                           Print exact build version\n"
        "  %s gait-list                               List gait training database entries\n"
        "  %s gait-reset                              Clear gait training database\n"
        "  %s list                                    List face training database entries\n"
        "  %s reset                                   Clear face training database\n\n"
        "Options:\n"
        "  --db <path>       Face training database file (default: %s)\n"
        "  --gait-db <path>  Gait training database file (default: %s)\n"
        "  --interval <sec>  Sampling interval for video commands (default: 1.0, gait default: 0.1)\n"
        "  --overlay         Show feature breakdown overlay (detect-gait only)\n\n"
        "Evaluation dataset layout:\n"
        "  dataset_dir/\n"
        "    alice/ sample1.jpg sample2.jpg\n"
        "    bob/   sample1.jpg sample2.jpg\n\n"
        "Examples:\n"
        "  ./vision train alice photo1.jpg photo2.jpg\n"
        "  ./vision detect group_photo.jpg output.png\n"
        "  ./vision train-gait raul walking.mp4\n"
        "  ./vision eval-face test-faces reports\n"
        "  ./vision eval-gait test-gait reports --interval 0.1\n",
        APP_VERSION, BUILD_VERSION,
        prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog, DEFAULT_DB, DEFAULT_GAIT_DB);
}

static int face_predict_wrapper(const char *path, double interval,
                                char *label_buf, size_t label_buf_size,
                                float *out_confidence, int *out_observations)
{
    (void)interval;
    return face_predict(path, label_buf, label_buf_size,
                        out_confidence, out_observations);
}

static int gait_predict_wrapper(const char *path, double interval,
                                char *label_buf, size_t label_buf_size,
                                float *out_confidence, int *out_observations)
{
    return gait_predict(path, interval, label_buf, label_buf_size,
                        out_confidence, out_observations);
}

static int is_directory(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int is_regular_file(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int ensure_directory(const char *path)
{
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return 0;
    fprintf(stderr, "error: cannot create directory %s\n", path);
    return -1;
}

static int has_supported_extension(const char *path, const char *mode)
{
    const char *dot = strrchr(path, '.');
    if (!dot || dot == path) return 0;
    dot++;

    if (strcmp(mode, "face") == 0) {
        return strcasecmp(dot, "jpg") == 0 ||
               strcasecmp(dot, "jpeg") == 0 ||
               strcasecmp(dot, "png") == 0 ||
               strcasecmp(dot, "heic") == 0 ||
               strcasecmp(dot, "bmp") == 0 ||
               strcasecmp(dot, "tif") == 0 ||
               strcasecmp(dot, "tiff") == 0;
    }

    return strcasecmp(dot, "mp4") == 0 ||
           strcasecmp(dot, "mov") == 0 ||
           strcasecmp(dot, "m4v") == 0 ||
           strcasecmp(dot, "avi") == 0 ||
           strcasecmp(dot, "mkv") == 0;
}

static void csv_write_cell(FILE *fp, const char *text)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"')
            fputc('"', fp);
        fputc(*p, fp);
    }
    fputc('"', fp);
}

static void sanitize_filename_component(const char *input,
                                        char *output, size_t output_size)
{
    size_t j = 0;
    if (output_size == 0) return;

    for (size_t i = 0; input[i] != '\0' && j + 1 < output_size; i++) {
        unsigned char c = (unsigned char)input[i];
        output[j++] = isalnum(c) ? (char)c : '_';
    }
    output[j] = '\0';
}

static void current_utc_timestamp(char *buf, size_t buf_size)
{
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

static int ensure_label_index(const char *label,
                              char labels[][MAX_LABEL_LEN],
                              int *label_count)
{
    for (int i = 0; i < *label_count; i++) {
        if (strcmp(labels[i], label) == 0)
            return i;
    }
    if (*label_count >= MAX_EVAL_LABELS) {
        fprintf(stderr, "error: too many labels in evaluation set\n");
        return -1;
    }
    snprintf(labels[*label_count], MAX_LABEL_LEN, "%s", label);
    (*label_count)++;
    return *label_count - 1;
}

static int write_confusion_csv(const char *path,
                               char labels[][MAX_LABEL_LEN],
                               int label_count,
                               const int *matrix)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot write %s\n", path);
        return -1;
    }

    fprintf(fp, "\"actual/predicted\"");
    for (int col = 0; col < label_count; col++) {
        fputc(',', fp);
        csv_write_cell(fp, labels[col]);
    }
    fputc('\n', fp);

    for (int row = 0; row < label_count; row++) {
        csv_write_cell(fp, labels[row]);
        for (int col = 0; col < label_count; col++)
            fprintf(fp, ",%d", matrix[row * label_count + col]);
        fputc('\n', fp);
    }

    fclose(fp);
    return 0;
}

static int write_summary_txt(const char *path, const char *mode,
                             const char *dataset_dir, const char *report_dir,
                             const char *db_path, double interval,
                             const char *generated_at,
                             char labels[][MAX_LABEL_LEN], int label_count,
                             const int *matrix, const int *actual_totals,
                             int total_samples, int correct,
                             int no_observation_samples)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot write %s\n", path);
        return -1;
    }

    fprintf(fp, "mode: %s\n", mode);
    fprintf(fp, "app_version: %s\n", APP_VERSION);
    fprintf(fp, "build_version: %s\n", BUILD_VERSION);
    fprintf(fp, "generated_at_utc: %s\n", generated_at);
    fprintf(fp, "dataset_dir: %s\n", dataset_dir);
    fprintf(fp, "report_dir: %s\n", report_dir);
    fprintf(fp, "database_path: %s\n", db_path);
    fprintf(fp, "sample_interval_sec: %.3f\n", interval);
    fprintf(fp, "total_samples: %d\n", total_samples);
    fprintf(fp, "correct_predictions: %d\n", correct);
    fprintf(fp, "overall_accuracy: %.4f\n",
            total_samples > 0 ? (double)correct / total_samples : 0.0);
    fprintf(fp, "samples_with_no_observations: %d\n", no_observation_samples);
    fprintf(fp, "label_count: %d\n\n", label_count);

    fprintf(fp, "per_label_recall:\n");
    for (int row = 0; row < label_count; row++) {
        int tp = matrix[row * label_count + row];
        int total = actual_totals[row];
        if (total == 0) continue;
        fprintf(fp, "  %s: %d/%d = %.4f\n",
                labels[row], tp, total, (double)tp / total);
    }

    fclose(fp);
    return 0;
}

static int evaluate_dataset(const char *mode,
                            const char *dataset_dir,
                            const char *report_dir,
                            const char *db_path,
                            double interval,
                            PredictFn predict)
{
    DIR *root = opendir(dataset_dir);
    if (!root) {
        fprintf(stderr, "error: cannot open dataset directory %s\n",
                dataset_dir);
        return -1;
    }

    if (ensure_directory(report_dir) != 0) {
        closedir(root);
        return -1;
    }

    char labels[MAX_EVAL_LABELS][MAX_LABEL_LEN];
    int label_count = 0;
    int *matrix = calloc(MAX_EVAL_LABELS * MAX_EVAL_LABELS, sizeof(int));
    int *actual_totals = calloc(MAX_EVAL_LABELS, sizeof(int));
    int total_samples = 0;
    int correct = 0;
    int no_observation_samples = 0;
    int dataset_labels = 0;

    if (!matrix || !actual_totals) {
        fprintf(stderr, "error: out of memory\n");
        closedir(root);
        free(matrix);
        free(actual_totals);
        return -1;
    }

    int unknown_idx = ensure_label_index("unknown", labels, &label_count);
    if (unknown_idx < 0) {
        closedir(root);
        free(matrix);
        free(actual_totals);
        return -1;
    }

    char version_tag[256];
    sanitize_filename_component(BUILD_VERSION, version_tag,
                                sizeof(version_tag));

    char generated_at[64];
    current_utc_timestamp(generated_at, sizeof(generated_at));

    char predictions_path[MAX_PATH_LEN];
    char confusion_path[MAX_PATH_LEN];
    char summary_path[MAX_PATH_LEN];
    snprintf(predictions_path, sizeof(predictions_path),
             "%s/%s_predictions_%s.csv", report_dir, mode, version_tag);
    snprintf(confusion_path, sizeof(confusion_path),
             "%s/%s_confusion_%s.csv", report_dir, mode, version_tag);
    snprintf(summary_path, sizeof(summary_path),
             "%s/%s_summary_%s.txt", report_dir, mode, version_tag);

    FILE *pred_fp = fopen(predictions_path, "w");
    if (!pred_fp) {
        fprintf(stderr, "error: cannot write %s\n", predictions_path);
        closedir(root);
        free(matrix);
        free(actual_totals);
        return -1;
    }
    fprintf(pred_fp,
            "\"build_version\",\"mode\",\"actual\",\"predicted\","
            "\"confidence\",\"observations\",\"path\"\n");

    struct dirent *label_entry;
    while ((label_entry = readdir(root)) != NULL) {
        if (label_entry->d_name[0] == '.')
            continue;

        char label_dir[MAX_PATH_LEN];
        snprintf(label_dir, sizeof(label_dir), "%s/%s",
                 dataset_dir, label_entry->d_name);
        if (!is_directory(label_dir))
            continue;

        dataset_labels++;
        int actual_idx = ensure_label_index(label_entry->d_name,
                                            labels, &label_count);
        if (actual_idx < 0) {
            fclose(pred_fp);
            closedir(root);
            free(matrix);
            free(actual_totals);
            return -1;
        }

        DIR *samples = opendir(label_dir);
        if (!samples) {
            fprintf(stderr, "warning: cannot open %s\n", label_dir);
            continue;
        }

        struct dirent *sample_entry;
        while ((sample_entry = readdir(samples)) != NULL) {
            if (sample_entry->d_name[0] == '.')
                continue;

            char sample_path[MAX_PATH_LEN];
            snprintf(sample_path, sizeof(sample_path), "%s/%s",
                     label_dir, sample_entry->d_name);
            if (!is_regular_file(sample_path) ||
                !has_supported_extension(sample_path, mode))
                continue;

            char predicted[MAX_LABEL_LEN];
            float confidence = 0.0f;
            int observations = 0;
            if (predict(sample_path, interval, predicted, sizeof(predicted),
                        &confidence, &observations) != 0) {
                snprintf(predicted, sizeof(predicted), "unknown");
                confidence = 0.0f;
                observations = 0;
            }

            int predicted_idx = ensure_label_index(predicted,
                                                   labels, &label_count);
            if (predicted_idx < 0) {
                closedir(samples);
                fclose(pred_fp);
                closedir(root);
                free(matrix);
                free(actual_totals);
                return -1;
            }

            matrix[actual_idx * MAX_EVAL_LABELS + predicted_idx]++;
            actual_totals[actual_idx]++;
            total_samples++;
            if (actual_idx == predicted_idx)
                correct++;
            if (observations <= 0)
                no_observation_samples++;

            csv_write_cell(pred_fp, BUILD_VERSION);
            fputc(',', pred_fp);
            csv_write_cell(pred_fp, mode);
            fputc(',', pred_fp);
            csv_write_cell(pred_fp, labels[actual_idx]);
            fputc(',', pred_fp);
            csv_write_cell(pred_fp, labels[predicted_idx]);
            fprintf(pred_fp, ",%.2f,%d,", confidence, observations);
            csv_write_cell(pred_fp, sample_path);
            fputc('\n', pred_fp);
        }

        closedir(samples);
    }

    fclose(pred_fp);
    closedir(root);

    if (dataset_labels == 0 || total_samples == 0) {
        fprintf(stderr, "error: no evaluation samples found in %s\n",
                dataset_dir);
        free(matrix);
        free(actual_totals);
        return -1;
    }

    int *trimmed = calloc(label_count * label_count, sizeof(int));
    int *trimmed_totals = calloc(label_count, sizeof(int));
    if (!trimmed || !trimmed_totals) {
        fprintf(stderr, "error: out of memory\n");
        free(matrix);
        free(actual_totals);
        free(trimmed);
        free(trimmed_totals);
        return -1;
    }

    for (int row = 0; row < label_count; row++) {
        trimmed_totals[row] = actual_totals[row];
        for (int col = 0; col < label_count; col++)
            trimmed[row * label_count + col] =
                matrix[row * MAX_EVAL_LABELS + col];
    }

    int rc = 0;
    if (write_confusion_csv(confusion_path, labels, label_count, trimmed) != 0)
        rc = -1;
    if (write_summary_txt(summary_path, mode, dataset_dir, report_dir,
                          db_path, interval, generated_at,
                          labels, label_count, trimmed, trimmed_totals,
                          total_samples, correct,
                          no_observation_samples) != 0)
        rc = -1;

    if (rc == 0) {
        printf("App version: %s\n", APP_VERSION);
        printf("Build version: %s\n", BUILD_VERSION);
        printf("Evaluated %d %s sample(s) across %d label(s)\n",
               total_samples, mode, dataset_labels);
        printf("Accuracy: %.2f%% (%d/%d)\n",
               total_samples > 0 ? (100.0 * correct / total_samples) : 0.0,
               correct, total_samples);
        printf("Samples with no observations: %d\n", no_observation_samples);
        printf("Predictions report: %s\n", predictions_path);
        printf("Confusion matrix: %s\n", confusion_path);
        printf("Summary: %s\n", summary_path);
    }

    free(matrix);
    free(actual_totals);
    free(trimmed);
    free(trimmed_totals);
    return rc;
}

int main(int argc, char *argv[])
{
    const char *db_path = DEFAULT_DB;
    const char *gait_db_path = DEFAULT_GAIT_DB;
    int arg_start = 1;

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
        face_cleanup();
        return 1;
    }

    int rc = 0;

    if (strcmp(command, "train") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: train requires a label and at least one image path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *label = argv[arg_start + 1];
            int total = 0;
            for (int i = arg_start + 2; i < argc; i++) {
                int n = face_train(argv[i], label);
                if (n < 0) {
                    fprintf(stderr, "warning: failed to process %s\n", argv[i]);
                } else if (n == 0) {
                    fprintf(stderr, "warning: no faces found in %s\n", argv[i]);
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
            fprintf(stderr, "error: detect requires an image path and output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input = argv[arg_start + 1];
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
            fprintf(stderr, "error: detect-video requires a video path and output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 1.0;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            if (face_detect_video(input, output, interval) < 0) {
                fprintf(stderr, "error: video detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "body") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: body requires an image path and output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input = argv[arg_start + 1];
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
            fprintf(stderr, "error: body-video requires a video path and output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 1.0;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            if (body_detect_video(input, output, interval) < 0) {
                fprintf(stderr, "error: body video detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "train-gait") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: train-gait requires a label and a video path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *label = argv[arg_start + 1];
            const char *video = argv[arg_start + 2];
            double interval = 0.1;
            if (argc - arg_start >= 5 &&
                strcmp(argv[arg_start + 3], "--interval") == 0)
                interval = atof(argv[arg_start + 4]);
            if (gait_train(video, label, interval) <= 0) {
                fprintf(stderr, "error: gait training failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "detect-gait") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: detect-gait requires a video path and output path\n");
            usage(argv[0]);
            rc = 1;
        } else {
            const char *input = argv[arg_start + 1];
            const char *output = argv[arg_start + 2];
            double interval = 0.1;
            int overlay = 0;
            for (int oi = arg_start + 3; oi < argc; oi++) {
                if (strcmp(argv[oi], "--interval") == 0 && oi + 1 < argc) {
                    interval = atof(argv[++oi]);
                } else if (strcmp(argv[oi], "--overlay") == 0) {
                    overlay = 1;
                }
            }
            if (gait_detect_video(input, output, interval, overlay) < 0) {
                fprintf(stderr, "error: gait detection failed\n");
                rc = 1;
            }
        }

    } else if (strcmp(command, "eval-face") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: eval-face requires a dataset directory and report directory\n");
            usage(argv[0]);
            rc = 1;
        } else {
            rc = evaluate_dataset("face",
                                  argv[arg_start + 1],
                                  argv[arg_start + 2],
                                  db_path,
                                  0.0,
                                  face_predict_wrapper) != 0;
        }

    } else if (strcmp(command, "eval-gait") == 0) {
        if (argc - arg_start < 3) {
            fprintf(stderr, "error: eval-gait requires a dataset directory and report directory\n");
            usage(argv[0]);
            rc = 1;
        } else {
            double interval = 0.1;
            for (int oi = arg_start + 3; oi < argc; oi++) {
                if (strcmp(argv[oi], "--interval") == 0 && oi + 1 < argc)
                    interval = atof(argv[++oi]);
            }
            rc = evaluate_dataset("gait",
                                  argv[arg_start + 1],
                                  argv[arg_start + 2],
                                  gait_db_path,
                                  interval,
                                  gait_predict_wrapper) != 0;
        }

    } else if (strcmp(command, "version") == 0) {
        printf("%s\n", APP_VERSION);

    } else if (strcmp(command, "build-version") == 0) {
        printf("%s\n", BUILD_VERSION);

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
