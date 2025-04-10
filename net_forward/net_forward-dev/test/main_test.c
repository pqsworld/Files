#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "resource/para_desc_txt.h"
#include <malloc.h>
#include "resource/bmp_enhance.h"
#include "bmp.h"
#include "resource/patch_test.h"
#include "../net/net_cnn_common.h"

#define MAX_WIDTH 200
#define MAX_HEIGHT 200
#define MALLOC_ERROR -1
#define INH 118
#define INW 32
#define OUTH 124
#define OUTW 36


#define STR(arg) #arg

#define PTIME(x)                                                                                        \
    printf("\033[1;33m Test Time        : %.4fms \033[0m\n", (double)x * 1000);

#define PTEST(x,y)                                                                                      \
    do {                                                                                                \
        if (x >= 0/* condition */) {                                                                    \
            x = 0;                                                                                      \
            printf(" \n\033[1;30;43m Test %s      \033[0m : \033[1;30;42m PASS     \033[0m\n",y);       \
        } else {                                                                                        \
            printf(" \n\033[1;30;43m Test %s      \033[0m : \033[1;30;41m NOT PASS \033[0m\n",y);       \
        }                                                                                               \
    } while (0)

//#define  CHIP6192
#ifdef CHIP6135
#include "../net/6135/net_api.h"
#endif
#ifdef CHIP6159
#include "../net/6159/net_api.h"
#endif
#ifdef CHIP6157
#include "../net/6157/net_api.h"
#endif
#ifdef CHIP6191
#include "../net/6191/net_api.h"
#endif
#ifdef CHIP6192
#include "../net/6192/net_api.h"
#endif
#ifdef CHIP6193
#include "../net/6193/net_api.h"
#endif
// #define PATCH 1
// #define EXP 1
// #define MASK 1
// #define ENHANCE 1
// #define SPOOF 1
// #define MISTOUCH 1

// gcc main.c  -l main.h lib64/libsilfp_algo_net.lib -o main.o

int main()
{
    GetVersion();
#if (defined PATCH && defined SPOOF)
#define TESTAll 1
#endif

#if 1
    int h, w;
    int mask_h = 122;
    int mask_w = 36;

    int exp_ret = -1;
    int ori_ret = -1;
    int enhance_ret = -1;
    int mask_ret = -1;
    int spoof_ret = -1;
    int mistouch_ret = -1;
    int patch_ret = -1;
    int quality_ret = -1;
    int spd_ret = -1;



    clock_t clockstart, clockend;
    double duration;

    unsigned char* img_src = (unsigned char*)malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    unsigned short* img_dst_exp_short = (unsigned short*)malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned short));
    unsigned char* img_dst_exp_char = (unsigned char*)malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    unsigned char* img_dst_mask = (unsigned char*)malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    unsigned char* img_dst_enhance = (unsigned char*)malloc(MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));

    memset(img_src, 0, MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    memset(img_dst_exp_short, 0, MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned short));
    memset(img_dst_exp_char, 0, MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    memset(img_dst_mask, 0, MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));
    memset(img_dst_enhance, 0, MAX_HEIGHT * MAX_WIDTH * sizeof(unsigned char));

    if (img_src == NULL) {
        printf("--- MALLOC_ERROR! --- : img_src");
        return MALLOC_ERROR;
    };

    if (img_dst_exp_short == NULL) {
        printf("--- MALLOC_ERROR! --- : img_dst_exp_short");
        return MALLOC_ERROR;
    };

    if (img_dst_exp_char == NULL) {
        printf("--- MALLOC_ERROR! --- : img_dst_exp_char");
        return MALLOC_ERROR;
    };

    if (img_dst_mask == NULL) {
        printf("--- MALLOC_ERROR! --- : img_dst_mask");
        return MALLOC_ERROR;
    };

    if (img_dst_enhance == NULL) {
        printf("--- MALLOC_ERROR! --- : img_dst_enhance");
        return MALLOC_ERROR;
    };

    char name[260];

    unsigned char img[200 * 200] = {0};

    memset(name, 0, sizeof(name));

    memset(img, 0, 200 * 200);

    sprintf(name, "./resource/test_enhance_118_32.bmp");

#ifdef CHIP6192
    sprintf(name, "./resource/6192/test.bmp");

#endif

    puts(name);

    int load_ret = LoadBmp(name, img, &h, &w);

    if (load_ret < 0) {
        printf("load image fail!\n");
        return MALLOC_ERROR;
    }

    printf("h %d w %d\n", h, w);

#endif

#if PATCH

#if CHIP6192
    unsigned int* descriptors = (unsigned int*)malloc(500 * sizeof(unsigned int));
    float* descriptors_f = (float*)descriptors;
    unsigned int* descriptors_rect = (unsigned int*)malloc(500 * sizeof(unsigned int));
    float* descriptors_f_rect = (float*)descriptors_rect;
    //int descriptor_net_init_patch_short(int patch_size, void** net_memory);
    //int descriptor_net_patch_short(unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int patch_size, int sample_size, uint32_t* desc, void* net_memory);
    //int descriptor_net_deinit_patch(void* net_memory);
    clockstart = clock();
    int i, j, k, sum, sum_rect;
    int checksum = 0;
    int checksum_rect = 0;
    int ret1, ret2, ret3, ret4, ret5, ret6;
    int patchSZ_rect[2] = { 32, 8 };
    int sampSZ_rect[2] = { 40, 10 };

    for (j = 0; j < 1; j++) {
        void* descriptor_net_memory = NULL;
        void* descriptor_net_memory_rect = NULL;
        ret1 = descriptor_net_init_patch_short(16, &descriptor_net_memory);

        if (ret1 < 0) {
            printf("malloc error!\n");
            return 0;
        }

        ret4 = descriptor_net_init_patch_short_rect(patchSZ_rect, &descriptor_net_memory_rect);

        if (ret4 < 0) {
            printf("malloc error!\n");
            return 0;
        }


        sum = 2070;
        sum_rect = -1261;

        for (i = 0; i < 165; i++) {
            //修改了block函数，注意：0823修改
            ret2 = descriptor_net_patch_short(ucImage_92, 186, 52, nX_92[i] + 8, nY_92[i] + 3, ori_92[i], 16, 18, descriptors, descriptor_net_memory);
            ret5 = descriptor_net_patch_short_rect(ucImage_92, 186, 52, nX_92[i] + 8, nY_92[i] + 3, ori_92[i], patchSZ_rect, sampSZ_rect, descriptors_rect, descriptor_net_memory_rect);

            for (k = 0; k < 256; k++) {
                checksum += (int)(100 * descriptors_f[k]);
            }

            for (k = 0; k < 128; k++) {
                checksum_rect += (int)(100 * descriptors_f_rect[k]);
            }
        }

        ret3 = descriptor_net_deinit_patch(descriptor_net_memory);
        ret6 = descriptor_net_deinit_patch(descriptor_net_memory_rect);
    }

    patch_ret = (ret1 && ret2 && ret3 && ret4 && ret5 && ret6);

    free(descriptors);
    free(descriptors_rect);
    clockend = clock();

    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;


    if (checksum != sum || checksum_rect != sum_rect) {
        printf(" \n\033[1;30;43m Test PATCH      \033[0m : \033[1;30;42m WRONG \033[0m checksum %d checksum_rect %d   \n", checksum, checksum_rect);
    } else {
        printf("Desc Test \033[0m : \033[1;30;42m RIGHT     \033[0m\n");
    }

    PTEST(patch_ret, STR(PATCH));
    PTIME(duration);
#endif

#if CHIP6193
    unsigned int* descriptors = (unsigned int*)malloc(500 * sizeof(unsigned int));
    float* descriptors_f = (float*)descriptors;
    unsigned int* descriptors_rect = (unsigned int*)malloc(500 * sizeof(unsigned int));
    float* descriptors_f_rect = (float*)descriptors_rect;


    //int descriptor_net_init_patch_short(int patch_size, void** net_memory);
    //int descriptor_net_patch_short(unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int patch_size, int sample_size, uint32_t* desc, void* net_memory);
    //int descriptor_net_deinit_patch(void* net_memory);
    clockstart = clock();

    int i, j, k, sum, sum_rect;
    int checksum = 0;
    int checksum_rect = 0;
    int ret1, ret2, ret3, ret4, ret5, ret6;
    int patchSZ_rect[2] = { 32, 8 };
    int sampSZ_rect[2] = { 40, 10 };

    for (j = 0; j < 1; j++) {
        void* descriptor_net_memory = NULL;
        void* descriptor_net_memory_rect = NULL;
        ret1 = descriptor_net_init_patch_short(16, &descriptor_net_memory);

        if (ret1 < 0) {
            printf("malloc error!\n");
            return 0;
        }

        ret4 = descriptor_net_init_patch_short_rect(patchSZ_rect, &descriptor_net_memory_rect);

        if (ret4 < 0) {
            printf("malloc error!\n");
            return 0;
        }

        sum = -4367;
        sum_rect = -1918;
        int flag = 0;

        for (i = 0; i < 130; i++) {

            //修改了block函数，注意：0823修改
            ret2 = descriptor_net_patch_short(ucImage, 128, 52, nX[i] + 8, nY[i] + 3, ori[i], 16, 22, descriptors, descriptor_net_memory);
            ret5 = descriptor_net_patch_short_rect(ucImage, 128, 52, nX[i] + 8, nY[i] + 3, ori[i], patchSZ_rect, sampSZ_rect, descriptors_rect, descriptor_net_memory_rect);

            for (k = 0; k < 256; k++) {
                checksum += (int)(100 * descriptors_f[k]);
            }

            for (k = 0; k < 128; k++) {
                checksum_rect += (int)(100 * descriptors_f_rect[k]);

            }
        }

        ret3 = descriptor_net_deinit_patch(descriptor_net_memory);
        ret6 = descriptor_net_deinit_patch(descriptor_net_memory_rect);
    }

    patch_ret = (ret1 && ret2 && ret3 && ret4 && ret5 && ret6);

    free(descriptors);
    free(descriptors_rect);

    clockend = clock();

    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;


    if (checksum != sum || checksum_rect != sum_rect) {
        printf(" \n\033[1;30;43m Test PATCH      \033[0m : \033[1;30;42m WRONG \033[0m checksum %d checksum_rect %d   \n", checksum, checksum_rect);
    } else {
        printf("Desc Test \033[0m : \033[1;30;42m RIGHT     \033[0m\n");
    }

    PTEST(patch_ret, STR(PATCH));
    PTIME(duration);
#endif

#endif

    /*************************************2.扩边测试*****************************************/

#if EXP

    clockstart = clock();

#ifdef CHIP6192
    sprintf(name, "./resource/6192/enhance_test.bmp");
    puts(name);
    load_ret = LoadBmp(name, img, &h, &w);
    printf("h %d w %d\n", h, w);

    if (load_ret < 0) {
        printf("enhance load image fail!\n");
        return MALLOC_ERROR;
    }

    ori_ret = ori_exp(img, h, w, img_dst_exp_char, 3, 8);

#endif
    exp_ret = desc_exp(img, h, w, img_dst_exp_short, 3, 8);

    clockend = clock();

    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    int cnt;

    for (cnt = 0; cnt < (h + 6) * (w + 16); cnt++) {
        img_dst_exp_char[cnt] = img_dst_exp_short[cnt] >> 8;
    }

    WriteBmp("dst_exp.bmp", img_dst_exp_char, h + 6, w + 16);

    PTEST(ori_ret, STR(exp));
    PTEST(exp_ret, STR(exp));
    PTIME(duration);

#endif

#if MASK
    sprintf(name, "./resource/src_mask.bmp");
    puts(name);
    load_ret = LoadBmp(name, img, &h, &w);

    if (load_ret < 0) {
        printf("mask load image fail!\n");
        return MALLOC_ERROR;
    }

    printf("h %d w %d\n", h, w);

    int32_t threshold = 128;

    int score = 65536;

    clockstart = clock();
    mask_ret = scratch_mask(img, h, w, img_dst_mask, mask_h, mask_w, threshold, &score);
    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    // GFP_Write_Bmp_8("gfp_mask.bmp", img_dst_mask, h, w);
    WriteBmp("dst_mask.bmp", img_dst_mask, h, w);

    PTEST(mask_ret, STR(MASK));
    PTIME(duration);

#endif

#if ENHANCE

    clockstart = clock();
#ifdef CHIP6193
    enhance_ret = net_enhance(img, INH, INW, img_dst_enhance, OUTH, OUTW);
#endif
#ifdef CHIP6192
    sprintf(name, "./resource/6192/enhance_test.bmp");
    puts(name);
    load_ret = LoadBmp(name, img, &h, &w);
    printf("h %d w %d\n", h, w);

    if (load_ret < 0) {
        printf("enhance load image fail!\n");
        return MALLOC_ERROR;
    }

    enhance_ret = net_enhance(img, h, w, img_dst_enhance);
#endif

    WriteBmp("dst_enhance.bmp", img_dst_enhance, INH, INW);


    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    PTEST(enhance_ret, STR(ENHANCE));
    PTIME(duration);

#endif

#if SPOOF
    clockstart = clock();
    int32_t spoof_score = 65536;
    static float g_spoof_value[2] = {0};
    spoof_ret = spoof_check(img, w, h, 0, 0, g_spoof_value, &spoof_score);

    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    PTEST(spoof_ret, STR(SPOOF));
    PTIME(duration);

#endif

#if MISTOUCH
    clockstart = clock();
    int32_t anti_finger_score = 65536;
    mistouch_ret = mistouch_check(img, w, h, &anti_finger_score);

    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    PTEST(mistouch_ret, STR(MISTOUCH));
    PTIME(duration);

#endif

#if QUALITY
    clockstart = clock();
    int32_t quality_score = 65536;
    quality_ret = net_quality(img, h, w, &quality_score);

    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    PTEST(quality_ret, STR(QUALITY));
    PTIME(duration);
#endif

#if SPD
#ifdef CHIP6193
    spd_ret = 0;
    printf("\033[1;33m not support test 93spd          \033[0m\n");
#endif
#ifdef CHIP6192
    clockstart = clock();
    float spd_out[2] = {0};
    spd_ret = spd_check((int*)img, h, w, spd_out);

    clockend = clock();
    duration = (double)(clockend - clockstart) / CLOCKS_PER_SEC;

    PTEST(spd_ret, STR(SPD));
    PTIME(duration);
#endif
#endif

#if 1

    printf("exp_ret: %d\n", exp_ret);
    printf("enhance_ret: %d\n", enhance_ret);
    printf("mask_ret: %d\n", mask_ret);
    printf("spoof_ret: %d\n", spoof_ret);
    printf("mistouch_ret: %d\n", mistouch_ret);
    printf("patch_ret: %d\n", patch_ret);
    printf("quality_ret: %d\n", quality_ret);
    printf("spd_ret: %d\n", spd_ret);

    if (img_dst_exp_short) {
        free(img_dst_exp_short);
    }

    if (img_dst_exp_char) {
        free(img_dst_exp_char);
    }

    if (img_dst_mask) {
        free(img_dst_mask);
    }

    if (img_dst_enhance) {
        free(img_dst_enhance);
    }

    if (img_src) {
        free(img_src);
    }

#endif

#if EXP
    return exp_ret;
#elif ENHANCE
    return enhance_ret;
#elif MASK
    return mask_ret;
#elif PATCH
    return patch_ret;
#elif MISTOUCH
    return mistouch_ret;
#elif QUALITY
    return quality_ret;
#elif SPOOF
    return spoof_ret;
#elif SPD
    return spd_ret;
#endif
}
