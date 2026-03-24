/*
 * Copyright (c) 2026 Raul I. Lopez
 * SPDX-License-Identifier: MIT
 * See LICENSE file for details.
 */

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>
#include "gait_detector.h"
#include "face_detector_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define MAX_GAIT_DESCRIPTOR_SIZE 32
#define MAX_GAIT_TRAINING        200
#define GAIT_MATCH_THRESHOLD     0.60f
#define MAX_GAIT_FRAMES          500
#define MIN_GAIT_FRAMES          10
#define MIN_VALID_JOINTS         12
#define JOINT_CONFIDENCE_MIN     0.1f

/* ------------------------------------------------------------------ */
/*  Joint indices (for our internal arrays)                            */
/* ------------------------------------------------------------------ */

enum {
    JI_NOSE = 0, JI_LEFT_EYE, JI_RIGHT_EYE,
    JI_LEFT_EAR, JI_RIGHT_EAR,
    JI_NECK,
    JI_LEFT_SHOULDER, JI_RIGHT_SHOULDER,
    JI_LEFT_ELBOW, JI_RIGHT_ELBOW,
    JI_LEFT_WRIST, JI_RIGHT_WRIST,
    JI_ROOT,
    JI_LEFT_HIP, JI_RIGHT_HIP,
    JI_LEFT_KNEE, JI_RIGHT_KNEE,
    JI_LEFT_ANKLE, JI_RIGHT_ANKLE,
    JI_COUNT  /* = 19 */
};

/* ------------------------------------------------------------------ */
/*  Training entry                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char  label[256];
    float descriptor[MAX_GAIT_DESCRIPTOR_SIZE];
    int   descriptor_size;
} GaitEntry;

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static GaitEntry g_gait[MAX_GAIT_TRAINING];
static int       g_num_gait = 0;
static char      g_gait_db_path[1024] = "gait_training.dat";

/* ------------------------------------------------------------------ */
/*  Per-frame pose data                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    CGPoint joints[JI_COUNT];
    int     valid[JI_COUNT];   /* 1 if confidence >= threshold */
    int     num_valid;
} GaitFrame;

/* ------------------------------------------------------------------ */
/*  Database persistence                                               */
/* ------------------------------------------------------------------ */

static int save_gait_db(void)
{
    FILE *fp = fopen(g_gait_db_path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot write %s\n", g_gait_db_path);
        return -1;
    }
    uint32_t n = (uint32_t)g_num_gait;
    fwrite(&n, sizeof(n), 1, fp);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t label_len = (uint32_t)strlen(g_gait[i].label);
        fwrite(&label_len, sizeof(label_len), 1, fp);
        fwrite(g_gait[i].label, 1, label_len, fp);
        uint32_t ds = (uint32_t)g_gait[i].descriptor_size;
        fwrite(&ds, sizeof(ds), 1, fp);
        fwrite(g_gait[i].descriptor, sizeof(float), ds, fp);
    }
    fclose(fp);
    return 0;
}

static int load_gait_db(void)
{
    FILE *fp = fopen(g_gait_db_path, "rb");
    if (!fp) return 0;

    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 || n > MAX_GAIT_TRAINING) {
        fclose(fp); return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t label_len = 0;
        if (fread(&label_len, sizeof(label_len), 1, fp) != 1 ||
            label_len >= sizeof(g_gait[i].label)) {
            fclose(fp); return -1;
        }
        if (fread(g_gait[i].label, 1, label_len, fp) != label_len) {
            fclose(fp); return -1;
        }
        g_gait[i].label[label_len] = '\0';

        uint32_t ds = 0;
        if (fread(&ds, sizeof(ds), 1, fp) != 1 ||
            ds > MAX_GAIT_DESCRIPTOR_SIZE) {
            fclose(fp); return -1;
        }
        if (fread(g_gait[i].descriptor, sizeof(float), ds, fp) != ds) {
            fclose(fp); return -1;
        }
        g_gait[i].descriptor_size = (int)ds;
    }
    g_num_gait = (int)n;
    fclose(fp);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Geometry helpers                                                   */
/* ------------------------------------------------------------------ */

static float pt_distance(CGPoint a, CGPoint b)
{
    float dx = (float)(a.x - b.x), dy = (float)(a.y - b.y);
    return sqrtf(dx * dx + dy * dy);
}

/* Angle at vertex B formed by rays BA and BC, in radians [0, pi] */
static float angle_at_vertex(CGPoint a, CGPoint b, CGPoint c)
{
    float bax = (float)(a.x - b.x), bay = (float)(a.y - b.y);
    float bcx = (float)(c.x - b.x), bcy = (float)(c.y - b.y);
    float dot  = bax * bcx + bay * bcy;
    float magA = sqrtf(bax * bax + bay * bay);
    float magC = sqrtf(bcx * bcx + bcy * bcy);
    if (magA < 1e-6f || magC < 1e-6f) return 0.0f;
    float cosine = dot / (magA * magC);
    if (cosine > 1.0f) cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;
    return acosf(cosine);
}

static int count_zero_crossings(float *signal, int len)
{
    int crossings = 0;
    for (int i = 1; i < len; i++) {
        if ((signal[i - 1] > 0 && signal[i] <= 0) ||
            (signal[i - 1] <= 0 && signal[i] > 0))
            crossings++;
    }
    return crossings;
}

/* ------------------------------------------------------------------ */
/*  Joint name mapping                                                 */
/* ------------------------------------------------------------------ */

static VNHumanBodyPoseObservationJointName joint_name_for_index(int idx)
{
    switch (idx) {
        case JI_NOSE:           return VNHumanBodyPoseObservationJointNameNose;
        case JI_LEFT_EYE:       return VNHumanBodyPoseObservationJointNameLeftEye;
        case JI_RIGHT_EYE:      return VNHumanBodyPoseObservationJointNameRightEye;
        case JI_LEFT_EAR:       return VNHumanBodyPoseObservationJointNameLeftEar;
        case JI_RIGHT_EAR:      return VNHumanBodyPoseObservationJointNameRightEar;
        case JI_NECK:           return VNHumanBodyPoseObservationJointNameNeck;
        case JI_LEFT_SHOULDER:  return VNHumanBodyPoseObservationJointNameLeftShoulder;
        case JI_RIGHT_SHOULDER: return VNHumanBodyPoseObservationJointNameRightShoulder;
        case JI_LEFT_ELBOW:     return VNHumanBodyPoseObservationJointNameLeftElbow;
        case JI_RIGHT_ELBOW:    return VNHumanBodyPoseObservationJointNameRightElbow;
        case JI_LEFT_WRIST:     return VNHumanBodyPoseObservationJointNameLeftWrist;
        case JI_RIGHT_WRIST:    return VNHumanBodyPoseObservationJointNameRightWrist;
        case JI_ROOT:           return VNHumanBodyPoseObservationJointNameRoot;
        case JI_LEFT_HIP:       return VNHumanBodyPoseObservationJointNameLeftHip;
        case JI_RIGHT_HIP:      return VNHumanBodyPoseObservationJointNameRightHip;
        case JI_LEFT_KNEE:      return VNHumanBodyPoseObservationJointNameLeftKnee;
        case JI_RIGHT_KNEE:     return VNHumanBodyPoseObservationJointNameRightKnee;
        case JI_LEFT_ANKLE:     return VNHumanBodyPoseObservationJointNameLeftAnkle;
        case JI_RIGHT_ANKLE:    return VNHumanBodyPoseObservationJointNameRightAnkle;
        default:                return nil;
    }
}

/* ------------------------------------------------------------------ */
/*  Extract joints from a body pose observation                        */
/* ------------------------------------------------------------------ */

static int extract_gait_frame(VNHumanBodyPoseObservation *body,
                              GaitFrame *frame)
{
    int valid_count = 0;
    for (int i = 0; i < JI_COUNT; i++) {
        frame->valid[i] = 0;
        frame->joints[i] = CGPointZero;
        VNHumanBodyPoseObservationJointName name = joint_name_for_index(i);
        if (!name) continue;
        NSError *err = nil;
        VNRecognizedPoint *pt =
            [body recognizedPointForJointName:name error:&err];
        if (pt && pt.confidence >= JOINT_CONFIDENCE_MIN) {
            frame->joints[i] = pt.location;
            frame->valid[i] = 1;
            valid_count++;
        }
    }
    frame->num_valid = valid_count;
    return valid_count;
}

/* ------------------------------------------------------------------ */
/*  Extract gait descriptor from a frame sequence                      */
/* ------------------------------------------------------------------ */

static int extract_gait_descriptor(GaitFrame *frames, int nframes,
                                   double duration,
                                   float *descriptor)
{
    if (nframes < MIN_GAIT_FRAMES) return 0;

    /* Buffers for angle time series */
    float *knee_l   = calloc(nframes, sizeof(float));
    float *knee_r   = calloc(nframes, sizeof(float));
    float *hip_l    = calloc(nframes, sizeof(float));
    float *hip_r    = calloc(nframes, sizeof(float));
    float *elbow_l  = calloc(nframes, sizeof(float));
    float *elbow_r  = calloc(nframes, sizeof(float));
    float *shld_l   = calloc(nframes, sizeof(float));
    float *shld_r   = calloc(nframes, sizeof(float));
    float *root_y   = calloc(nframes, sizeof(float));
    float *sway     = calloc(nframes, sizeof(float));
    float *lean     = calloc(nframes, sizeof(float));
    float *ankle_osc = calloc(nframes, sizeof(float));
    float *arm_ang  = calloc(nframes * 2, sizeof(float));  /* 2 per frame (L+R) */

    int n_knee_l = 0, n_knee_r = 0;
    int n_hip_l = 0, n_hip_r = 0;
    int n_elbow_l = 0, n_elbow_r = 0;
    int n_shld_l = 0, n_shld_r = 0;
    int n_root = 0, n_sway = 0, n_lean = 0, n_ankle = 0, n_arm = 0;

    /* Body height estimation */
    float height_sum = 0;
    int height_count = 0;

    /* Body proportion accumulators */
    float upper_lower_sum = 0; int n_ul = 0;
    float shld_w_sum = 0;     int n_sw = 0;
    float hip_w_sum = 0;      int n_hw = 0;
    float sh_ratio_sum = 0;   int n_shr = 0;
    float stride_max = 0, stride_min = 1e9f;

    for (int f = 0; f < nframes; f++) {
        GaitFrame *fr = &frames[f];
        if (fr->num_valid < MIN_VALID_JOINTS) continue;

        /* Body height: nose to midpoint of ankles */
        if (fr->valid[JI_NOSE] && fr->valid[JI_LEFT_ANKLE] &&
            fr->valid[JI_RIGHT_ANKLE]) {
            CGPoint mid_ankle = {
                (fr->joints[JI_LEFT_ANKLE].x + fr->joints[JI_RIGHT_ANKLE].x) / 2,
                (fr->joints[JI_LEFT_ANKLE].y + fr->joints[JI_RIGHT_ANKLE].y) / 2
            };
            float h = pt_distance(fr->joints[JI_NOSE], mid_ankle);
            if (h > 0.01f) { height_sum += h; height_count++; }
        }

        /* Knee angles */
        if (fr->valid[JI_LEFT_HIP] && fr->valid[JI_LEFT_KNEE] &&
            fr->valid[JI_LEFT_ANKLE])
            knee_l[n_knee_l++] = angle_at_vertex(
                fr->joints[JI_LEFT_HIP], fr->joints[JI_LEFT_KNEE],
                fr->joints[JI_LEFT_ANKLE]);
        if (fr->valid[JI_RIGHT_HIP] && fr->valid[JI_RIGHT_KNEE] &&
            fr->valid[JI_RIGHT_ANKLE])
            knee_r[n_knee_r++] = angle_at_vertex(
                fr->joints[JI_RIGHT_HIP], fr->joints[JI_RIGHT_KNEE],
                fr->joints[JI_RIGHT_ANKLE]);

        /* Hip angles (leg swing) */
        if (fr->valid[JI_LEFT_SHOULDER] && fr->valid[JI_LEFT_HIP] &&
            fr->valid[JI_LEFT_KNEE])
            hip_l[n_hip_l++] = angle_at_vertex(
                fr->joints[JI_LEFT_SHOULDER], fr->joints[JI_LEFT_HIP],
                fr->joints[JI_LEFT_KNEE]);
        if (fr->valid[JI_RIGHT_SHOULDER] && fr->valid[JI_RIGHT_HIP] &&
            fr->valid[JI_RIGHT_KNEE])
            hip_r[n_hip_r++] = angle_at_vertex(
                fr->joints[JI_RIGHT_SHOULDER], fr->joints[JI_RIGHT_HIP],
                fr->joints[JI_RIGHT_KNEE]);

        /* Elbow angles */
        if (fr->valid[JI_LEFT_SHOULDER] && fr->valid[JI_LEFT_ELBOW] &&
            fr->valid[JI_LEFT_WRIST])
            elbow_l[n_elbow_l++] = angle_at_vertex(
                fr->joints[JI_LEFT_SHOULDER], fr->joints[JI_LEFT_ELBOW],
                fr->joints[JI_LEFT_WRIST]);
        if (fr->valid[JI_RIGHT_SHOULDER] && fr->valid[JI_RIGHT_ELBOW] &&
            fr->valid[JI_RIGHT_WRIST])
            elbow_r[n_elbow_r++] = angle_at_vertex(
                fr->joints[JI_RIGHT_SHOULDER], fr->joints[JI_RIGHT_ELBOW],
                fr->joints[JI_RIGHT_WRIST]);

        /* Shoulder angles (arm swing) */
        if (fr->valid[JI_NECK] && fr->valid[JI_LEFT_SHOULDER] &&
            fr->valid[JI_LEFT_ELBOW])
            shld_l[n_shld_l++] = angle_at_vertex(
                fr->joints[JI_NECK], fr->joints[JI_LEFT_SHOULDER],
                fr->joints[JI_LEFT_ELBOW]);
        if (fr->valid[JI_NECK] && fr->valid[JI_RIGHT_SHOULDER] &&
            fr->valid[JI_RIGHT_ELBOW])
            shld_r[n_shld_r++] = angle_at_vertex(
                fr->joints[JI_NECK], fr->joints[JI_RIGHT_SHOULDER],
                fr->joints[JI_RIGHT_ELBOW]);

        /* Root vertical position */
        if (fr->valid[JI_ROOT])
            root_y[n_root++] = (float)fr->joints[JI_ROOT].y;

        /* Hip sway: root x - shoulder midpoint x */
        if (fr->valid[JI_ROOT] && fr->valid[JI_LEFT_SHOULDER] &&
            fr->valid[JI_RIGHT_SHOULDER]) {
            float mid_shld_x = (float)(fr->joints[JI_LEFT_SHOULDER].x +
                                       fr->joints[JI_RIGHT_SHOULDER].x) / 2;
            sway[n_sway++] = (float)fr->joints[JI_ROOT].x - mid_shld_x;
        }

        /* Forward lean: angle of root->neck from vertical */
        if (fr->valid[JI_ROOT] && fr->valid[JI_NECK]) {
            float dx = (float)(fr->joints[JI_NECK].x - fr->joints[JI_ROOT].x);
            float dy = (float)(fr->joints[JI_NECK].y - fr->joints[JI_ROOT].y);
            lean[n_lean++] = atan2f(fabsf(dx), dy);
        }

        /* Ankle oscillation: relative y of left vs right ankle */
        if (fr->valid[JI_LEFT_ANKLE] && fr->valid[JI_RIGHT_ANKLE])
            ankle_osc[n_ankle++] = (float)(fr->joints[JI_LEFT_ANKLE].y -
                                           fr->joints[JI_RIGHT_ANKLE].y);

        /* Arm swing angle from vertical */
        if (fr->valid[JI_LEFT_SHOULDER] && fr->valid[JI_LEFT_WRIST]) {
            float dx = (float)(fr->joints[JI_LEFT_WRIST].x -
                               fr->joints[JI_LEFT_SHOULDER].x);
            float dy = (float)(fr->joints[JI_LEFT_WRIST].y -
                               fr->joints[JI_LEFT_SHOULDER].y);
            arm_ang[n_arm++] = atan2f(fabsf(dx), dy);
        }
        if (fr->valid[JI_RIGHT_SHOULDER] && fr->valid[JI_RIGHT_WRIST]) {
            float dx = (float)(fr->joints[JI_RIGHT_WRIST].x -
                               fr->joints[JI_RIGHT_SHOULDER].x);
            float dy = (float)(fr->joints[JI_RIGHT_WRIST].y -
                               fr->joints[JI_RIGHT_SHOULDER].y);
            arm_ang[n_arm++] = atan2f(fabsf(dx), dy);
        }

        /* Body proportions */
        if (fr->valid[JI_NECK] && fr->valid[JI_ROOT] &&
            fr->valid[JI_LEFT_ANKLE] && fr->valid[JI_RIGHT_ANKLE]) {
            CGPoint mid_ankle = {
                (fr->joints[JI_LEFT_ANKLE].x + fr->joints[JI_RIGHT_ANKLE].x) / 2,
                (fr->joints[JI_LEFT_ANKLE].y + fr->joints[JI_RIGHT_ANKLE].y) / 2
            };
            float upper = pt_distance(fr->joints[JI_NECK], fr->joints[JI_ROOT]);
            float lower = pt_distance(fr->joints[JI_ROOT], mid_ankle);
            if (lower > 0.001f)
                { upper_lower_sum += upper / lower; n_ul++; }
        }
        if (fr->valid[JI_LEFT_SHOULDER] && fr->valid[JI_RIGHT_SHOULDER]) {
            float sw = pt_distance(fr->joints[JI_LEFT_SHOULDER],
                                   fr->joints[JI_RIGHT_SHOULDER]);
            shld_w_sum += sw; n_sw++;
        }
        if (fr->valid[JI_LEFT_HIP] && fr->valid[JI_RIGHT_HIP]) {
            float hw = pt_distance(fr->joints[JI_LEFT_HIP],
                                   fr->joints[JI_RIGHT_HIP]);
            hip_w_sum += hw; n_hw++;
        }
        if (fr->valid[JI_LEFT_SHOULDER] && fr->valid[JI_RIGHT_SHOULDER] &&
            fr->valid[JI_LEFT_HIP] && fr->valid[JI_RIGHT_HIP]) {
            float sw = pt_distance(fr->joints[JI_LEFT_SHOULDER],
                                   fr->joints[JI_RIGHT_SHOULDER]);
            float hw = pt_distance(fr->joints[JI_LEFT_HIP],
                                   fr->joints[JI_RIGHT_HIP]);
            if (hw > 0.001f)
                { sh_ratio_sum += sw / hw; n_shr++; }
        }

        /* Stride width: horizontal distance between ankles */
        if (fr->valid[JI_LEFT_ANKLE] && fr->valid[JI_RIGHT_ANKLE]) {
            float d = fabsf((float)(fr->joints[JI_LEFT_ANKLE].x -
                                    fr->joints[JI_RIGHT_ANKLE].x));
            if (d > stride_max) stride_max = d;
            if (d < stride_min) stride_min = d;
        }
    }

    float body_height = (height_count > 0) ? height_sum / height_count : 1.0f;
    if (body_height < 0.01f) body_height = 1.0f;

    /* --- Compute statistics --- */
    int idx = 0;

    /* Helper macros */
#define MEAN(arr, n) ({ float s=0; for(int i=0;i<(n);i++) s+=(arr)[i]; (n)>0 ? s/(n) : 0; })
#define RANGE(arr, n) ({ float mn=1e9f,mx=-1e9f; for(int i=0;i<(n);i++){if((arr)[i]<mn)mn=(arr)[i];if((arr)[i]>mx)mx=(arr)[i];} (n)>0?mx-mn:0; })
#define STDDEV(arr, n) ({ float m=MEAN(arr,n), s=0; for(int i=0;i<(n);i++){float d=(arr)[i]-m;s+=d*d;} (n)>1?sqrtf(s/((n)-1)):0; })
#define SYM_RATIO(mean_l, mean_r) ({ float a=fabsf(mean_l),b=fabsf(mean_r); (a>0.001f&&b>0.001f)?fminf(a,b)/fmaxf(a,b):1.0f; })

    /* Pool left+right for mean/range */
    float *knee_pool  = calloc(n_knee_l + n_knee_r, sizeof(float));
    memcpy(knee_pool, knee_l, n_knee_l * sizeof(float));
    memcpy(knee_pool + n_knee_l, knee_r, n_knee_r * sizeof(float));
    int n_knee = n_knee_l + n_knee_r;

    float *hip_pool = calloc(n_hip_l + n_hip_r, sizeof(float));
    memcpy(hip_pool, hip_l, n_hip_l * sizeof(float));
    memcpy(hip_pool + n_hip_l, hip_r, n_hip_r * sizeof(float));
    int n_hip = n_hip_l + n_hip_r;

    float *elbow_pool = calloc(n_elbow_l + n_elbow_r, sizeof(float));
    memcpy(elbow_pool, elbow_l, n_elbow_l * sizeof(float));
    memcpy(elbow_pool + n_elbow_l, elbow_r, n_elbow_r * sizeof(float));
    int n_elbow = n_elbow_l + n_elbow_r;

    float *shld_pool = calloc(n_shld_l + n_shld_r, sizeof(float));
    memcpy(shld_pool, shld_l, n_shld_l * sizeof(float));
    memcpy(shld_pool + n_shld_l, shld_r, n_shld_r * sizeof(float));
    int n_shld = n_shld_l + n_shld_r;

    /* Group A: Joint angle statistics (12 features) */
    descriptor[idx++] = MEAN(knee_pool, n_knee);           /* 0  */
    descriptor[idx++] = RANGE(knee_pool, n_knee);          /* 1  */
    descriptor[idx++] = MEAN(hip_pool, n_hip);             /* 2  */
    descriptor[idx++] = RANGE(hip_pool, n_hip);            /* 3  */
    descriptor[idx++] = MEAN(elbow_pool, n_elbow);         /* 4  */
    descriptor[idx++] = RANGE(elbow_pool, n_elbow);        /* 5  */
    descriptor[idx++] = MEAN(shld_pool, n_shld);           /* 6  */
    descriptor[idx++] = RANGE(shld_pool, n_shld);          /* 7  */

    float knee_l_mean  = MEAN(knee_l, n_knee_l);
    float knee_r_mean  = MEAN(knee_r, n_knee_r);
    float hip_l_mean   = MEAN(hip_l, n_hip_l);
    float hip_r_mean   = MEAN(hip_r, n_hip_r);
    float elbow_l_mean = MEAN(elbow_l, n_elbow_l);
    float elbow_r_mean = MEAN(elbow_r, n_elbow_r);
    float shld_l_mean  = MEAN(shld_l, n_shld_l);
    float shld_r_mean  = MEAN(shld_r, n_shld_r);

    descriptor[idx++] = SYM_RATIO(knee_l_mean, knee_r_mean);   /* 8  */
    descriptor[idx++] = SYM_RATIO(hip_l_mean, hip_r_mean);     /* 9  */
    descriptor[idx++] = SYM_RATIO(elbow_l_mean, elbow_r_mean); /* 10 */
    descriptor[idx++] = SYM_RATIO(shld_l_mean, shld_r_mean);   /* 11 */

    /* Group B: Body proportions (4 features) */
    descriptor[idx++] = n_ul > 0 ? upper_lower_sum / n_ul : 0;          /* 12 */
    descriptor[idx++] = n_sw > 0 ? (shld_w_sum / n_sw) / body_height : 0; /* 13 */
    descriptor[idx++] = n_hw > 0 ? (hip_w_sum / n_hw) / body_height : 0;  /* 14 */
    descriptor[idx++] = n_shr > 0 ? sh_ratio_sum / n_shr : 0;           /* 15 */

    /* Group C: Cadence and dynamics (4 features) */
    int zc = count_zero_crossings(ankle_osc, n_ankle);
    float cadence = (duration > 0 && n_ankle > 0)
        ? (float)(zc / 2.0) / (float)duration : 0;
    descriptor[idx++] = cadence;                                   /* 16 */
    descriptor[idx++] = (stride_max - stride_min) / body_height;   /* 17 */
    descriptor[idx++] = STDDEV(root_y, n_root) / body_height;     /* 18 */
    descriptor[idx++] = STDDEV(sway, n_sway) / body_height;       /* 19 */

    /* Group D: Temporal dynamics and posture (4 features) */
    descriptor[idx++] = RANGE(arm_ang, n_arm) / (float)M_PI;      /* 20 */
    descriptor[idx++] = MEAN(lean, n_lean);                        /* 21 */
    descriptor[idx++] = STDDEV(lean, n_lean);                      /* 22 */

    /* Ankle oscillation regularity (coefficient of variation of intervals) */
    float regularity = 0;
    if (zc >= 2 && n_ankle > 1) {
        /* Find zero-crossing positions */
        int *zc_pos = calloc(zc, sizeof(int));
        int zci = 0;
        for (int i = 1; i < n_ankle && zci < zc; i++) {
            if ((ankle_osc[i - 1] > 0 && ankle_osc[i] <= 0) ||
                (ankle_osc[i - 1] <= 0 && ankle_osc[i] > 0))
                zc_pos[zci++] = i;
        }
        if (zci >= 2) {
            float *intervals = calloc(zci - 1, sizeof(float));
            for (int i = 0; i < zci - 1; i++)
                intervals[i] = (float)(zc_pos[i + 1] - zc_pos[i]);
            float imean = MEAN(intervals, zci - 1);
            float istd  = STDDEV(intervals, zci - 1);
            regularity = (imean > 0) ? istd / imean : 0;
            free(intervals);
        }
        free(zc_pos);
    }
    descriptor[idx++] = regularity;                                /* 23 */

#undef MEAN
#undef RANGE
#undef STDDEV
#undef SYM_RATIO

    /* Cleanup */
    free(knee_l); free(knee_r);
    free(hip_l); free(hip_r);
    free(elbow_l); free(elbow_r);
    free(shld_l); free(shld_r);
    free(root_y); free(sway); free(lean);
    free(ankle_osc); free(arm_ang);
    free(knee_pool); free(hip_pool);
    free(elbow_pool); free(shld_pool);

    return idx;  /* 24 */
}

/* ------------------------------------------------------------------ */
/*  Matching                                                           */
/* ------------------------------------------------------------------ */

static float gait_distance(const float *a, const float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

static const char *gait_best_match(const float *descriptor, int size,
                                   float *out_confidence)
{
    float       best_dist  = GAIT_MATCH_THRESHOLD;
    const char *best_label = NULL;

    for (int i = 0; i < g_num_gait; i++) {
        if (g_gait[i].descriptor_size != size) continue;
        float dist = gait_distance(descriptor, g_gait[i].descriptor, size);
        if (dist < best_dist) {
            best_dist  = dist;
            best_label = g_gait[i].label;
        }
    }
    if (out_confidence) {
        if (best_label)
            *out_confidence = (1.0f - best_dist / GAIT_MATCH_THRESHOLD) * 100.0f;
        else {
            float closest = -1;
            for (int i = 0; i < g_num_gait; i++) {
                if (g_gait[i].descriptor_size != size) continue;
                float dist = gait_distance(descriptor,
                                           g_gait[i].descriptor, size);
                if (closest < 0 || dist < closest) closest = dist;
            }
            *out_confidence = (closest >= 0)
                ? (1.0f - closest / GAIT_MATCH_THRESHOLD) * 100.0f : 0;
        }
    }
    return best_label;
}

/* ------------------------------------------------------------------ */
/*  Collect pose frames from a video                                   */
/* ------------------------------------------------------------------ */

static int collect_gait_frames(const char *video_path,
                               double sample_interval,
                               GaitFrame *frames, int max_frames,
                               double *out_duration)
{
    NSString *path = [NSString stringWithUTF8String:video_path];
    NSURL *url = [NSURL fileURLWithPath:path];
    AVURLAsset *asset = [AVURLAsset assetWithURL:url];

    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [asset loadValuesAsynchronouslyForKeys:@[@"tracks", @"duration"]
                         completionHandler:^{ dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    NSArray<AVAssetTrack *> *tracks =
        [asset tracksWithMediaType:AVMediaTypeVideo];
    if (tracks.count == 0) {
        fprintf(stderr, "error: no video track in %s\n", video_path);
        return -1;
    }

    double duration = CMTimeGetSeconds(asset.duration);
    if (out_duration) *out_duration = duration;

    AVAssetImageGenerator *gen =
        [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
    gen.appliesPreferredTrackTransform = YES;
    gen.requestedTimeToleranceBefore = kCMTimeZero;
    gen.requestedTimeToleranceAfter  = kCMTimeZero;

    int nframes = 0;

    for (double t = 0; t < duration && nframes < max_frames;
         t += sample_interval) {
        @autoreleasepool {
            CMTime time = CMTimeMakeWithSeconds(t, 600);
            CMTime actual;
            NSError *err = nil;
            CGImageRef frame =
                [gen copyCGImageAtTime:time actualTime:&actual error:&err];
            if (!frame) continue;

            VNDetectHumanBodyPoseRequest *request =
                [[VNDetectHumanBodyPoseRequest alloc] init];
            VNImageRequestHandler *handler =
                [[VNImageRequestHandler alloc] initWithCGImage:frame
                                                      options:@{}];
            NSError *vErr = nil;
            [handler performRequests:@[request] error:&vErr];
            CGImageRelease(frame);

            if (vErr || !request.results || request.results.count == 0)
                continue;

            /* Use the first body (or largest if multiple) */
            VNHumanBodyPoseObservation *body = request.results[0];

            int valid = extract_gait_frame(body, &frames[nframes]);
            if (valid >= MIN_VALID_JOINTS)
                nframes++;
        }
    }

    return nframes;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */


/* ------------------------------------------------------------------ */
/*  Skeleton drawing (lightweight copy for gait video output)          */
/* ------------------------------------------------------------------ */

static void draw_body_on_context(VNHumanBodyPoseObservation *body,
                                 CGContextRef ctx,
                                 size_t imgW, size_t imgH,
                                 const char *label)
{
    /* Draw white rods */
    typedef struct { int from, to; } Bone;
    Bone bone_list[] = {
        {JI_NOSE, JI_LEFT_EYE}, {JI_NOSE, JI_RIGHT_EYE},
        {JI_LEFT_EYE, JI_LEFT_EAR}, {JI_RIGHT_EYE, JI_RIGHT_EAR},
        {JI_NECK, JI_NOSE},
        {JI_NECK, JI_LEFT_SHOULDER}, {JI_NECK, JI_RIGHT_SHOULDER},
        {JI_LEFT_SHOULDER, JI_LEFT_ELBOW}, {JI_LEFT_ELBOW, JI_LEFT_WRIST},
        {JI_RIGHT_SHOULDER, JI_RIGHT_ELBOW}, {JI_RIGHT_ELBOW, JI_RIGHT_WRIST},
        {JI_NECK, JI_ROOT},
        {JI_ROOT, JI_LEFT_HIP}, {JI_ROOT, JI_RIGHT_HIP},
        {JI_LEFT_HIP, JI_LEFT_KNEE}, {JI_LEFT_KNEE, JI_LEFT_ANKLE},
        {JI_RIGHT_HIP, JI_RIGHT_KNEE}, {JI_RIGHT_KNEE, JI_RIGHT_ANKLE},
    };
    int nbones = sizeof(bone_list) / sizeof(bone_list[0]);

    GaitFrame gf;
    extract_gait_frame(body, &gf);

    CGContextSetRGBStrokeColor(ctx, 1.0, 1.0, 1.0, 0.9);
    CGContextSetLineWidth(ctx, 3.0);
    CGContextSetLineCap(ctx, kCGLineCapRound);

    for (int i = 0; i < nbones; i++) {
        if (!gf.valid[bone_list[i].from] || !gf.valid[bone_list[i].to])
            continue;
        CGFloat fx = gf.joints[bone_list[i].from].x * imgW;
        CGFloat fy = gf.joints[bone_list[i].from].y * imgH;
        CGFloat tx = gf.joints[bone_list[i].to].x * imgW;
        CGFloat ty = gf.joints[bone_list[i].to].y * imgH;
        CGContextMoveToPoint(ctx, fx, fy);
        CGContextAddLineToPoint(ctx, tx, ty);
        CGContextStrokePath(ctx);
    }

    /* Draw joint dots */
    CGFloat dotR = 8.0;
    for (int i = 0; i < JI_COUNT; i++) {
        if (!gf.valid[i]) continue;
        CGFloat px = gf.joints[i].x * imgW;
        CGFloat py = gf.joints[i].y * imgH;
        CGContextSetRGBFillColor(ctx, 0.0, 1.0, 0.5, 1.0);
        CGContextFillEllipseInRect(ctx,
            CGRectMake(px - dotR, py - dotR, dotR * 2, dotR * 2));
    }

    /* Draw label above the head */
    if (label && gf.valid[JI_NOSE]) {
        CGRect labelRect = CGRectMake(
            gf.joints[JI_NOSE].x * imgW - 40,
            gf.joints[JI_NOSE].y * imgH,
            80, 20);
        draw_label(ctx, label, labelRect, 0.0, 0.7, 0.0);
    }
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int gait_init(const char *training_db_path)
{
    if (training_db_path) {
        strncpy(g_gait_db_path, training_db_path,
                sizeof(g_gait_db_path) - 1);
        g_gait_db_path[sizeof(g_gait_db_path) - 1] = '\0';
    }
    return load_gait_db();
}

int gait_train(const char *video_path, const char *label,
               double sample_interval)
{
    @autoreleasepool {
        GaitFrame *frames = calloc(MAX_GAIT_FRAMES, sizeof(GaitFrame));
        double duration = 0;
        int nframes = collect_gait_frames(video_path, sample_interval,
                                          frames, MAX_GAIT_FRAMES,
                                          &duration);
        if (nframes < MIN_GAIT_FRAMES) {
            fprintf(stderr, "error: insufficient pose frames (%d, need %d)\n",
                    nframes, MIN_GAIT_FRAMES);
            free(frames);
            return 0;
        }

        printf("Collected %d pose frames over %.1f seconds\n",
               nframes, duration);

        if (g_num_gait >= MAX_GAIT_TRAINING) {
            fprintf(stderr, "error: gait training database full\n");
            free(frames);
            return -1;
        }

        GaitEntry *entry = &g_gait[g_num_gait];
        int desc_size = extract_gait_descriptor(frames, nframes, duration,
                                                entry->descriptor);
        free(frames);

        if (desc_size == 0) {
            fprintf(stderr, "error: could not extract gait descriptor\n");
            return 0;
        }

        strncpy(entry->label, label, sizeof(entry->label) - 1);
        entry->label[sizeof(entry->label) - 1] = '\0';
        entry->descriptor_size = desc_size;
        g_num_gait++;

        save_gait_db();

        printf("Stored gait descriptor (%d features) for '%s'\n",
               desc_size, label);
        return 1;
    }
}

int gait_detect_video(const char *input_path, const char *output_path,
                      double sample_interval)
{
    @autoreleasepool {
        /* Collect frames for gait analysis */
        GaitFrame *frames = calloc(MAX_GAIT_FRAMES, sizeof(GaitFrame));
        double duration = 0;
        int nframes = collect_gait_frames(input_path, sample_interval,
                                          frames, MAX_GAIT_FRAMES,
                                          &duration);

        /* Extract gait descriptor and match */
        const char *gait_label = NULL;
        float gait_confidence = 0;
        if (nframes >= MIN_GAIT_FRAMES) {
            float descriptor[MAX_GAIT_DESCRIPTOR_SIZE];
            int desc_size = extract_gait_descriptor(frames, nframes,
                                                    duration, descriptor);
            if (desc_size > 0)
                gait_label = gait_best_match(descriptor, desc_size,
                                             &gait_confidence);
        }
        free(frames);

        if (gait_label)
            printf("Gait match: %s (confidence: %.1f%%)\n",
                   gait_label, gait_confidence);
        else
            printf("Gait: no match (confidence: %.1f%%)\n", gait_confidence);

        /* Now produce the annotated output video */
        NSString *inPath = [NSString stringWithUTF8String:input_path];
        NSURL *inURL = [NSURL fileURLWithPath:inPath];
        AVURLAsset *asset = [AVURLAsset assetWithURL:inURL];

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [asset loadValuesAsynchronouslyForKeys:@[@"tracks", @"duration"]
                             completionHandler:^{ dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        NSArray<AVAssetTrack *> *videoTracks =
            [asset tracksWithMediaType:AVMediaTypeVideo];
        if (videoTracks.count == 0) return -1;

        AVAssetTrack *track = videoTracks[0];
        CGSize outputSize = CGSizeApplyAffineTransform(
            track.naturalSize, track.preferredTransform);
        size_t width  = (size_t)fabs(outputSize.width);
        size_t height = (size_t)fabs(outputSize.height);

        AVAssetImageGenerator *gen =
            [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        gen.appliesPreferredTrackTransform = YES;
        gen.requestedTimeToleranceBefore = kCMTimeZero;
        gen.requestedTimeToleranceAfter  = kCMTimeZero;

        /* Writer setup */
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

        /* Render frames with stick figures and gait label */
        double vidDuration = CMTimeGetSeconds(asset.duration);
        double render_interval = sample_interval;
        int frameIndex = 0;

        char label_buf[320];
        if (gait_label)
            snprintf(label_buf, sizeof(label_buf),
                     "%s (gait %.0f%%)", gait_label, gait_confidence);
        else
            snprintf(label_buf, sizeof(label_buf),
                     "unknown (gait %.0f%%)", gait_confidence);

        for (double t = 0; t < vidDuration; t += render_interval) {
            @autoreleasepool {
                CMTime time = CMTimeMakeWithSeconds(t, 600);
                CMTime actual;
                NSError *frameErr = nil;
                CGImageRef frame =
                    [gen copyCGImageAtTime:time actualTime:&actual
                                    error:&frameErr];
                if (!frame) continue;

                CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
                CGContextRef ctx = CGBitmapContextCreate(
                    NULL, width, height, 8, width * 4, cs,
                    kCGImageAlphaPremultipliedLast);
                CGColorSpaceRelease(cs);

                if (!ctx) { CGImageRelease(frame); continue; }

                CGContextDrawImage(ctx,
                    CGRectMake(0, 0, width, height), frame);

                /* Detect body and draw stick figure + label */
                VNDetectHumanBodyPoseRequest *request =
                    [[VNDetectHumanBodyPoseRequest alloc] init];
                VNImageRequestHandler *handler =
                    [[VNImageRequestHandler alloc] initWithCGImage:frame
                                                          options:@{}];
                NSError *vErr = nil;
                [handler performRequests:@[request] error:&vErr];

                if (!vErr && request.results.count > 0) {
                    for (VNHumanBodyPoseObservation *body in request.results)
                        draw_body_on_context(body, ctx, width, height,
                                             label_buf);
                }

                /* Write frame */
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

        /* Finalize */
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

        printf("Processed %d frames. Output saved to %s\n",
               frameIndex, output_path);
        return gait_label ? 1 : 0;
    }
}

void gait_list_training(void)
{
    if (g_num_gait == 0) {
        printf("Gait training database is empty.\n");
        return;
    }
    printf("Gait training database (%d entries):\n", g_num_gait);
    for (int i = 0; i < g_num_gait; i++) {
        printf("  [%3d] %-30s  (%d-dim descriptor)\n",
               i, g_gait[i].label, g_gait[i].descriptor_size);
    }
}

int gait_reset(void)
{
    g_num_gait = 0;
    remove(g_gait_db_path);
    printf("Gait training database cleared.\n");
    return 0;
}

void gait_cleanup(void)
{
    if (g_num_gait > 0)
        save_gait_db();
}
