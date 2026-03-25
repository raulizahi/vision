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

/* Per-feature weights for gait distance.
 * Speed-dependent features (ranges, stride) are downweighted because
 * they vary with walking speed and context, not identity.
 * Stable identity features (means, proportions, symmetry) get full weight. */
static const float gait_weights[24] = {
    1.0f, 0.3f,   /*  0: knee mean (stable), 1: knee range (speed-dep) */
    1.0f, 0.3f,   /*  2: hip mean,           3: hip range              */
    1.0f, 0.3f,   /*  4: elbow mean,         5: elbow range            */
    1.0f, 0.3f,   /*  6: shoulder mean,      7: shoulder range         */
    1.0f, 1.0f,   /*  8: knee sym,           9: hip sym                */
    1.0f, 1.0f,   /* 10: elbow sym,         11: shoulder sym           */
    1.0f, 1.0f,   /* 12: upper/lower,       13: shld width             */
    1.0f, 1.0f,   /* 14: hip width,         15: shld/hip ratio         */
    0.5f, 0.2f,   /* 16: cadence (mod),     17: stride (very speed-dep)*/
    1.0f, 1.0f,   /* 18: bounce,            19: sway                   */
    0.5f, 1.0f,   /* 20: arm swing (mod),   21: lean mean              */
    1.0f, 0.5f,   /* 22: lean stddev,       23: regularity (mod)       */
};

static float gait_distance(const float *a, const float *b, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float w = (i < 24) ? gait_weights[i] : 1.0f;
        float d = a[i] - b[i];
        sum += w * d * d;
    }
    return sqrtf(sum);
}

/* Body-region group definitions for per-group match breakdown.
 * Each group maps descriptor feature indices to a body region. */
#define NUM_GAIT_GROUPS 4
typedef struct {
    const char *name;
    float       confidence;  /* per-group confidence % */
} GaitGroupScore;

static void compute_group_contributions(const float *detected,
                                         const float *matched,
                                         int size,
                                         float overall_confidence,
                                         GaitGroupScore scores[NUM_GAIT_GROUPS])
{
    /* Feature-to-group mapping:
     *   Legs:      0,1 (knee angles), 8 (knee sym), 16,17 (cadence,stride), 23 (regularity)
     *   Hips:      2,3 (hip angles), 9 (hip sym), 14 (hip width), 18,19 (bounce,sway)
     *   Arms:      4,5 (elbow angles), 10 (elbow sym), 20 (arm swing)
     *   Shoulders: 6,7 (shld angles), 11 (shld sym), 12,13 (proportions), 15 (sh ratio), 21,22 (lean)
     */
    static const int legs_feats[]      = {0,1,8,16,17,23};
    static const int hips_feats[]      = {2,3,9,14,18,19};
    static const int arms_feats[]      = {4,5,10,20};
    static const int shoulders_feats[] = {6,7,11,12,13,15,21,22};

    const int *groups[]  = {legs_feats, hips_feats, arms_feats, shoulders_feats};
    const int  counts[]  = {6, 6, 4, 8};
    const char *names[]  = {"Legs", "Hips", "Arms", "Shoulders"};

    /* Compute per-group squared distances and total */
    float group_sq[NUM_GAIT_GROUPS];
    float total_sq = 0;

    for (int g = 0; g < NUM_GAIT_GROUPS; g++) {
        float sum = 0;
        for (int i = 0; i < counts[g]; i++) {
            int fi = groups[g][i];
            if (fi < size) {
                float d = detected[fi] - matched[fi];
                sum += d * d;
            }
        }
        group_sq[g] = sum;
        total_sq += sum;
    }

    /* Per-group confidence: linearly interpolate from overall confidence.
     * A group contributing 0% of total distance = 100% confidence.
     * A group contributing its expected share (n/N) = overall confidence.
     * A group contributing more than expected = below overall confidence.
     *
     * contribution = group_sq / total_sq  (fraction of total distance²)
     * expected     = n / N               (fair share)
     * ratio        = contribution / expected  (1.0 = average, >1 = worse)
     * confidence   = overall * (2 - ratio), clamped
     */
    for (int g = 0; g < NUM_GAIT_GROUPS; g++) {
        scores[g].name = names[g];
        if (total_sq < 1e-10f) {
            scores[g].confidence = overall_confidence;
        } else {
            float contribution = group_sq[g] / total_sq;
            float expected = (float)counts[g] / (float)size;
            float ratio = contribution / expected;
            /* ratio < 1 → this group matches better than average
             * ratio = 1 → matches same as average
             * ratio > 1 → matches worse than average */
            scores[g].confidence = overall_confidence * (2.0f - ratio);
        }
    }
}

/* gait_best_match_detailed: match descriptor and return per-group breakdown.
 * and per-group contribution scores. */
static const char *gait_best_match_detailed(const float *descriptor, int size,
                                            float *out_confidence,
                                            GaitGroupScore group_scores[NUM_GAIT_GROUPS])
{
    float       best_dist  = GAIT_MATCH_THRESHOLD;
    const char *best_label = NULL;
    int         best_idx   = -1;

    for (int i = 0; i < g_num_gait; i++) {
        if (g_gait[i].descriptor_size != size) continue;
        float dist = gait_distance(descriptor, g_gait[i].descriptor, size);
        if (dist < best_dist) {
            best_dist  = dist;
            best_label = g_gait[i].label;
            best_idx   = i;
        }
    }

    if (out_confidence) {
        if (best_label)
            *out_confidence = (1.0f - best_dist / GAIT_MATCH_THRESHOLD) * 100.0f;
        else {
            float closest = -1;
            int closest_idx = -1;
            for (int i = 0; i < g_num_gait; i++) {
                if (g_gait[i].descriptor_size != size) continue;
                float dist = gait_distance(descriptor,
                                           g_gait[i].descriptor, size);
                if (closest < 0 || dist < closest) {
                    closest = dist;
                    closest_idx = i;
                }
            }
            *out_confidence = (closest >= 0)
                ? (1.0f - closest / GAIT_MATCH_THRESHOLD) * 100.0f : 0;
            best_idx = closest_idx;
        }
    }

    /* Compute per-group breakdown */
    if (group_scores && best_idx >= 0) {
        compute_group_contributions(descriptor,
                                     g_gait[best_idx].descriptor,
                                     size, *out_confidence,
                                     group_scores);
    } else if (group_scores) {
        for (int g = 0; g < NUM_GAIT_GROUPS; g++) {
            const char *names[] = {"Legs", "Hips", "Arms", "Shoulders"};
            group_scores[g].name = names[g];
            group_scores[g].confidence = 0;
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
/*  Overlay label with configurable font size                          */
/* ------------------------------------------------------------------ */

static void draw_overlay_label(CGContextRef ctx, const char *text,
                                CGRect rect, CGFloat r, CGFloat g, CGFloat b,
                                CGFloat fontSize)
{
    NSString *nsText = [NSString stringWithUTF8String:text];
    CTFontRef font = CTFontCreateWithName(CFSTR("Menlo-Bold"), fontSize, NULL);
    NSDictionary *attrs = @{
        (id)kCTFontAttributeName : (__bridge id)font,
        (id)kCTForegroundColorAttributeName :
            (__bridge id)[[NSColor colorWithRed:r green:g blue:b alpha:1.0] CGColor]
    };
    NSAttributedString *attrStr =
        [[NSAttributedString alloc] initWithString:nsText attributes:attrs];
    CTLineRef line = CTLineCreateWithAttributedString(
        (__bridge CFAttributedStringRef)attrStr);

    CGContextSetTextPosition(ctx, rect.origin.x, rect.origin.y);
    CTLineDraw(line, ctx);

    CFRelease(line);
    CFRelease(font);
}

/* ------------------------------------------------------------------ */
/*  Skeleton drawing (lightweight copy for gait video output)          */
/* ------------------------------------------------------------------ */

static void draw_body_on_context(VNHumanBodyPoseObservation *body,
                                 CGContextRef ctx,
                                 size_t imgW, size_t imgH,
                                 const char *label,
                                 const float *matched_desc,
                                 const float *detected_desc,
                                 int desc_size)
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

    /* Joint color map (matches body_detector.m) */
    static const struct { int idx; CGFloat r, g, b; } jcolors[] = {
        { JI_NOSE,            1.0, 0.2, 0.2 },  /* red        */
        { JI_LEFT_EYE,        0.0, 1.0, 1.0 },  /* cyan       */
        { JI_RIGHT_EYE,       0.0, 1.0, 1.0 },  /* cyan       */
        { JI_LEFT_EAR,        1.0, 0.8, 0.0 },  /* orange     */
        { JI_RIGHT_EAR,       1.0, 0.8, 0.0 },  /* orange     */
        { JI_LEFT_SHOULDER,   0.0, 1.0, 0.0 },  /* green      */
        { JI_RIGHT_SHOULDER,  0.0, 1.0, 0.0 },  /* green      */
        { JI_NECK,            0.4, 1.0, 0.4 },  /* l-green    */
        { JI_LEFT_ELBOW,      0.0, 0.6, 1.0 },  /* blue       */
        { JI_RIGHT_ELBOW,     0.0, 0.6, 1.0 },  /* blue       */
        { JI_LEFT_WRIST,      0.4, 0.4, 1.0 },  /* l-blue     */
        { JI_RIGHT_WRIST,     0.4, 0.4, 1.0 },  /* l-blue     */
        { JI_LEFT_HIP,        1.0, 1.0, 0.0 },  /* yellow     */
        { JI_RIGHT_HIP,       1.0, 1.0, 0.0 },  /* yellow     */
        { JI_ROOT,            1.0, 0.5, 0.0 },  /* orange     */
        { JI_LEFT_KNEE,       1.0, 0.4, 0.7 },  /* pink       */
        { JI_RIGHT_KNEE,      1.0, 0.4, 0.7 },  /* pink       */
        { JI_LEFT_ANKLE,      0.6, 0.2, 1.0 },  /* purple     */
        { JI_RIGHT_ANKLE,     0.6, 0.2, 1.0 },  /* purple     */
    };
    int njcolors = sizeof(jcolors) / sizeof(jcolors[0]);

    /* ---- Per-frame group scores from instantaneous joint angles ---- */
    /* Compare this frame's angles against the matched descriptor means:
     *   desc[0] = knee mean,  desc[2] = hip mean,
     *   desc[4] = elbow mean, desc[6] = shoulder mean
     * Also body proportions:
     *   desc[12] = upper/lower, desc[13] = shoulder width / height,
     *   desc[14] = hip width / height, desc[15] = shoulder/hip ratio
     * Deviation from mean → confidence for that group.
     * Threshold: within desc[1]/2 (half the range) = good match. */
    GaitGroupScore frame_scores[NUM_GAIT_GROUPS];
    int have_frame_scores = 0;
    static const char *grp_names[] = {"Legs", "Hips", "Arms", "Shoulders"};

    if (matched_desc && desc_size >= 16) {
        float deviations[NUM_GAIT_GROUPS] = {0};
        int   dev_count[NUM_GAIT_GROUPS]  = {0};

        /* Legs: knee angle */
        if (gf.valid[JI_LEFT_HIP] && gf.valid[JI_LEFT_KNEE] &&
            gf.valid[JI_LEFT_ANKLE]) {
            float a = angle_at_vertex(gf.joints[JI_LEFT_HIP],
                                      gf.joints[JI_LEFT_KNEE],
                                      gf.joints[JI_LEFT_ANKLE]);
            float range = fmaxf(matched_desc[1], 0.1f);
            deviations[0] += fabsf(a - matched_desc[0]) / range;
            dev_count[0]++;
        }
        if (gf.valid[JI_RIGHT_HIP] && gf.valid[JI_RIGHT_KNEE] &&
            gf.valid[JI_RIGHT_ANKLE]) {
            float a = angle_at_vertex(gf.joints[JI_RIGHT_HIP],
                                      gf.joints[JI_RIGHT_KNEE],
                                      gf.joints[JI_RIGHT_ANKLE]);
            float range = fmaxf(matched_desc[1], 0.1f);
            deviations[0] += fabsf(a - matched_desc[0]) / range;
            dev_count[0]++;
        }

        /* Hips: hip angle (leg swing) */
        if (gf.valid[JI_LEFT_SHOULDER] && gf.valid[JI_LEFT_HIP] &&
            gf.valid[JI_LEFT_KNEE]) {
            float a = angle_at_vertex(gf.joints[JI_LEFT_SHOULDER],
                                      gf.joints[JI_LEFT_HIP],
                                      gf.joints[JI_LEFT_KNEE]);
            float range = fmaxf(matched_desc[3], 0.1f);
            deviations[1] += fabsf(a - matched_desc[2]) / range;
            dev_count[1]++;
        }
        if (gf.valid[JI_RIGHT_SHOULDER] && gf.valid[JI_RIGHT_HIP] &&
            gf.valid[JI_RIGHT_KNEE]) {
            float a = angle_at_vertex(gf.joints[JI_RIGHT_SHOULDER],
                                      gf.joints[JI_RIGHT_HIP],
                                      gf.joints[JI_RIGHT_KNEE]);
            float range = fmaxf(matched_desc[3], 0.1f);
            deviations[1] += fabsf(a - matched_desc[2]) / range;
            dev_count[1]++;
        }

        /* Arms: elbow angle */
        if (gf.valid[JI_LEFT_SHOULDER] && gf.valid[JI_LEFT_ELBOW] &&
            gf.valid[JI_LEFT_WRIST]) {
            float a = angle_at_vertex(gf.joints[JI_LEFT_SHOULDER],
                                      gf.joints[JI_LEFT_ELBOW],
                                      gf.joints[JI_LEFT_WRIST]);
            float range = fmaxf(matched_desc[5], 0.1f);
            deviations[2] += fabsf(a - matched_desc[4]) / range;
            dev_count[2]++;
        }
        if (gf.valid[JI_RIGHT_SHOULDER] && gf.valid[JI_RIGHT_ELBOW] &&
            gf.valid[JI_RIGHT_WRIST]) {
            float a = angle_at_vertex(gf.joints[JI_RIGHT_SHOULDER],
                                      gf.joints[JI_RIGHT_ELBOW],
                                      gf.joints[JI_RIGHT_WRIST]);
            float range = fmaxf(matched_desc[5], 0.1f);
            deviations[2] += fabsf(a - matched_desc[4]) / range;
            dev_count[2]++;
        }

        /* Shoulders: shoulder angle (arm swing) */
        if (gf.valid[JI_NECK] && gf.valid[JI_LEFT_SHOULDER] &&
            gf.valid[JI_LEFT_ELBOW]) {
            float a = angle_at_vertex(gf.joints[JI_NECK],
                                      gf.joints[JI_LEFT_SHOULDER],
                                      gf.joints[JI_LEFT_ELBOW]);
            float range = fmaxf(matched_desc[7], 0.1f);
            deviations[3] += fabsf(a - matched_desc[6]) / range;
            dev_count[3]++;
        }
        if (gf.valid[JI_NECK] && gf.valid[JI_RIGHT_SHOULDER] &&
            gf.valid[JI_RIGHT_ELBOW]) {
            float a = angle_at_vertex(gf.joints[JI_NECK],
                                      gf.joints[JI_RIGHT_SHOULDER],
                                      gf.joints[JI_RIGHT_ELBOW]);
            float range = fmaxf(matched_desc[7], 0.1f);
            deviations[3] += fabsf(a - matched_desc[6]) / range;
            dev_count[3]++;
        }

        /* Convert deviations to confidence:
         * deviation/range <= 0.5 → within expected range → 100%
         * deviation/range  = 1.0 → at edge of range → 50%
         * deviation/range  = 2.0 → double the range → 0%
         * Linear: confidence = (1 - avg_dev / 2) * 100, clamped to [0,100] */
        for (int g = 0; g < NUM_GAIT_GROUPS; g++) {
            frame_scores[g].name = grp_names[g];
            if (dev_count[g] > 0) {
                float avg_dev = deviations[g] / dev_count[g];
                float conf = (1.0f - avg_dev / 2.0f) * 100.0f;
                if (conf < 0) conf = 0;
                if (conf > 100) conf = 100;
                frame_scores[g].confidence = conf;
            } else {
                frame_scores[g].confidence = -1;  /* no data */
            }
        }
        have_frame_scores = 1;
    }

    /* Joint-to-group mapping:
     *   0=Legs, 1=Hips, 2=Arms, 3=Shoulders, -1=no group */
    static const int joint_group[JI_COUNT] = {
        -1,  /* NOSE          */
        -1,  /* LEFT_EYE      */
        -1,  /* RIGHT_EYE     */
        -1,  /* LEFT_EAR      */
        -1,  /* RIGHT_EAR     */
         3,  /* NECK          → Shoulders */
         3,  /* LEFT_SHOULDER  → Shoulders */
         3,  /* RIGHT_SHOULDER → Shoulders */
         2,  /* LEFT_ELBOW    → Arms */
         2,  /* RIGHT_ELBOW   → Arms */
         2,  /* LEFT_WRIST    → Arms */
         2,  /* RIGHT_WRIST   → Arms */
         1,  /* ROOT          → Hips */
         1,  /* LEFT_HIP      → Hips */
         1,  /* RIGHT_HIP     → Hips */
         0,  /* LEFT_KNEE     → Legs */
         0,  /* RIGHT_KNEE    → Legs */
         0,  /* LEFT_ANKLE    → Legs */
         0,  /* RIGHT_ANKLE   → Legs */
    };

    /* Draw halos based on per-frame group scores */
    if (have_frame_scores) {
        CGFloat haloR = 20.0;
        CGContextSetLineWidth(ctx, 3.0);
        for (int i = 0; i < njcolors; i++) {
            int ji = jcolors[i].idx;
            if (!gf.valid[ji]) continue;
            int grp = joint_group[ji];
            if (grp < 0) continue;

            float conf = frame_scores[grp].confidence;
            if (conf < 0) continue;  /* no data for this group */

            CGFloat hr, hg, hb, ha;
            if (conf >= 70.0f) {
                hr = 0.0; hg = 1.0; hb = 0.0; ha = 0.8;  /* green */
            } else if (conf >= 40.0f) {
                hr = 1.0; hg = 1.0; hb = 0.0; ha = 0.6;  /* yellow */
            } else {
                hr = 1.0; hg = 0.0; hb = 0.0; ha = 0.5;  /* red */
            }

            CGFloat px = gf.joints[ji].x * imgW;
            CGFloat py = gf.joints[ji].y * imgH;
            CGContextSetRGBStrokeColor(ctx, hr, hg, hb, ha);
            CGContextStrokeEllipseInRect(ctx,
                CGRectMake(px - haloR, py - haloR, haloR * 2, haloR * 2));
        }
    }

    /* Draw colored joint dots */
    CGFloat dotR = 12.0;
    for (int i = 0; i < njcolors; i++) {
        int ji = jcolors[i].idx;
        if (!gf.valid[ji]) continue;
        CGFloat px = gf.joints[ji].x * imgW;
        CGFloat py = gf.joints[ji].y * imgH;
        CGContextSetRGBFillColor(ctx, jcolors[i].r, jcolors[i].g,
                                 jcolors[i].b, 1.0);
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

    /* Draw per-group legend in top-left corner */
    if (have_frame_scores) {
        CGFloat grpFontSz = 36.0;
        CGFloat grpRowH   = 44.0;
        for (int g = 0; g < NUM_GAIT_GROUPS; g++) {
            if (frame_scores[g].confidence < 0) continue;
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: %.0f%%",
                     frame_scores[g].name, frame_scores[g].confidence);

            CGFloat lr, lg, lb;
            if (frame_scores[g].confidence >= 70.0f) {
                lr = 0.0; lg = 1.0; lb = 0.0;
            } else if (frame_scores[g].confidence >= 40.0f) {
                lr = 1.0; lg = 1.0; lb = 0.0;
            } else {
                lr = 1.0; lg = 0.3; lb = 0.3;
            }

            CGRect legendRect = CGRectMake(10, 10 + g * grpRowH, 300, grpRowH);
            draw_overlay_label(ctx, buf, legendRect, lr, lg, lb, grpFontSz);
        }
    }

    /* Draw per-feature breakdown at bottom of image */
    if (matched_desc && detected_desc && desc_size > 0) {
        static const char *feat_short[] = {
            "knee",    "knee rng",   "hip",     "hip rng",
            "elbow",   "elbow rng",  "shld",    "shld rng",
            "knee sy", "hip sy",     "elbow sy","shld sy",
            "up/lo",   "shld w",     "hip w",   "sh/hp",
            "cadence", "stride",     "bounce",  "sway",
            "arm sw",  "lean",       "lean sd", "regular",
        };

        /* Compute total squared distance for percentages */
        float total_sq = 0;
        for (int i = 0; i < desc_size; i++) {
            float w = (i < 24) ? gait_weights[i] : 1.0f;
            float d = detected_desc[i] - matched_desc[i];
            total_sq += w * d * d;
        }

        /* Layout: 2 columns of 12 rows, starting from bottom */
        CGFloat fontSize = 28.0;
        int rows = (desc_size + 1) / 2;
        CGFloat rowH = 34.0;
        CGFloat colW = (CGFloat)imgW / 2.0;
        CGFloat baseY = (CGFloat)imgH - rows * rowH - 8;

        /* Semi-transparent background */
        CGContextSetRGBFillColor(ctx, 0.0, 0.0, 0.0, 0.7);
        CGContextFillRect(ctx,
            CGRectMake(0, baseY - 4, (CGFloat)imgW, rows * rowH + 12));

        for (int i = 0; i < desc_size && i < 24; i++) {
            int col = i / rows;
            int row = i % rows;

            float w = gait_weights[i];
            float d = detected_desc[i] - matched_desc[i];
            float feat_sq = w * d * d;
            float pct = (total_sq > 0) ? (feat_sq / total_sq) * 100.0f : 0;

            char buf[80];
            snprintf(buf, sizeof(buf), "%-9s %5.2f/%5.2f %4.0f%%",
                     feat_short[i], detected_desc[i], matched_desc[i], pct);

            /* Color: green if <5% contribution, yellow 5-15%, red >15% */
            CGFloat lr, lg, lb;
            if (pct > 15.0f) {
                lr = 1.0; lg = 0.3; lb = 0.3;
            } else if (pct > 5.0f) {
                lr = 1.0; lg = 1.0; lb = 0.0;
            } else {
                lr = 0.6; lg = 1.0; lb = 0.6;
            }

            CGRect r = CGRectMake(col * colW + 10, baseY + row * rowH,
                                  colW - 20, rowH);
            draw_overlay_label(ctx, buf, r, lr, lg, lb, fontSize);
        }
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

int gait_predict(const char *video_path, double sample_interval,
                 char *label_buf, size_t label_buf_size,
                 float *out_confidence, int *out_pose_frames)
{
    @autoreleasepool {
        GaitFrame *frames = calloc(MAX_GAIT_FRAMES, sizeof(GaitFrame));
        double duration = 0;
        int nframes = collect_gait_frames(video_path, sample_interval,
                                          frames, MAX_GAIT_FRAMES,
                                          &duration);

        const char *gait_label = NULL;
        float gait_confidence = 0.0f;
        if (nframes >= MIN_GAIT_FRAMES) {
            float detected_desc[MAX_GAIT_DESCRIPTOR_SIZE];
            int detected_desc_size = extract_gait_descriptor(frames, nframes,
                                                             duration,
                                                             detected_desc);
            if (detected_desc_size > 0) {
                gait_label = gait_best_match_detailed(detected_desc,
                                                      detected_desc_size,
                                                      &gait_confidence,
                                                      NULL);
            }
        }

        free(frames);

        if (label_buf && label_buf_size > 0) {
            const char *result = gait_label ? gait_label : "unknown";
            snprintf(label_buf, label_buf_size, "%s", result);
        }
        if (out_confidence)
            *out_confidence = gait_confidence;
        if (out_pose_frames)
            *out_pose_frames = nframes;

        return 0;
    }
}

int gait_detect_video(const char *input_path, const char *output_path,
                      double sample_interval, int show_overlay)
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
        float detected_desc[MAX_GAIT_DESCRIPTOR_SIZE];
        float matched_desc[MAX_GAIT_DESCRIPTOR_SIZE];
        int matched_desc_size = 0;
        int detected_desc_size = 0;
        GaitGroupScore group_scores[NUM_GAIT_GROUPS];
        int has_groups = 0;
        if (nframes >= MIN_GAIT_FRAMES) {
            detected_desc_size = extract_gait_descriptor(frames, nframes,
                                                          duration,
                                                          detected_desc);
            if (detected_desc_size > 0) {
                gait_label = gait_best_match_detailed(detected_desc,
                                                       detected_desc_size,
                                                       &gait_confidence,
                                                       group_scores);
                has_groups = 1;
            }
        }
        free(frames);

        /* Find the matched descriptor from training DB for per-frame overlay */
        if (has_groups) {
            /* Find closest entry (matched or closest if unknown) */
            float best_dist = 1e9f;
            for (int i = 0; i < g_num_gait; i++) {
                if (g_gait[i].descriptor_size != detected_desc_size) continue;
                float dist = gait_distance(detected_desc,
                                           g_gait[i].descriptor,
                                           detected_desc_size);
                if (dist < best_dist) {
                    best_dist = dist;
                    memcpy(matched_desc, g_gait[i].descriptor,
                           g_gait[i].descriptor_size * sizeof(float));
                    matched_desc_size = g_gait[i].descriptor_size;
                }
            }
        }

        /* Feature names for diagnostic output */
        static const char *feat_names[] = {
            "knee mean",     "knee range",       /*  0, 1  */
            "hip mean",      "hip range",        /*  2, 3  */
            "elbow mean",    "elbow range",      /*  4, 5  */
            "shoulder mean", "shoulder range",    /*  6, 7  */
            "knee sym",      "hip sym",          /*  8, 9  */
            "elbow sym",     "shoulder sym",     /* 10,11  */
            "upper/lower",   "shld width",       /* 12,13  */
            "hip width",     "shld/hip ratio",   /* 14,15  */
            "cadence",       "stride",           /* 16,17  */
            "bounce",        "sway",             /* 18,19  */
            "arm swing",     "lean mean",        /* 20,21  */
            "lean stddev",   "regularity",       /* 22,23  */
        };

        if (gait_label) {
            printf("Gait match: %s (confidence: %.1f%%)\n",
                   gait_label, gait_confidence);
        } else {
            printf("Gait: no match (confidence: %.1f%%)\n", gait_confidence);
        }

        if (has_groups && matched_desc_size > 0) {
            float total_sq = 0;
            float feat_sq[MAX_GAIT_DESCRIPTOR_SIZE];
            for (int i = 0; i < detected_desc_size; i++) {
                float d = detected_desc[i] - matched_desc[i];
                feat_sq[i] = d * d;
                total_sq += feat_sq[i];
            }
            float total_dist = sqrtf(total_sq);

            printf("  Per-feature distance breakdown (total: %.3f, threshold: %.2f):\n",
                   total_dist, GAIT_MATCH_THRESHOLD);
            for (int i = 0; i < detected_desc_size; i++) {
                float pct = (total_sq > 0) ? (feat_sq[i] / total_sq) * 100.0f : 0;
                const char *bar = (pct > 20) ? "!!!" : (pct > 10) ? "!! " : (pct > 5) ? "!  " : "   ";
                printf("    [%2d] %-15s  detected=%.3f  trained=%.3f  dist²=%5.1f%% %s\n",
                       i, (i < 24) ? feat_names[i] : "?",
                       detected_desc[i], matched_desc[i], pct, bar);
            }
        }

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
                                             label_buf,
                                             show_overlay && matched_desc_size > 0 ? matched_desc : NULL,
                                             show_overlay && detected_desc_size > 0 ? detected_desc : NULL,
                                             matched_desc_size);
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
