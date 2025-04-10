#ifndef __NET_CNN_COMMON_H__
#define __NET_CNN_COMMON_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alog.h"
//#include "./net_param/struct_common.h"

#define PUBLIC   /* empty */
#define PRIVATE  static

//非win环境且非linux环境且未定义neonoff：打开armneon开关
//#if __ANDROID__
#ifndef  _WINDOWS
#if !defined(NEON_OFF) && !defined(MAKE_LINUX)
#ifndef NEON
#ifndef __ARM_NEON__
#define __ARM_NEON__
#endif
#endif
#endif
#endif
//win环境关闭neon
#ifdef  _WINDOWS
#undef NEON
#define NEON 0 // 0
#endif

#ifdef __ARM_NEON__
#include "arm_neon.h"
#define NEON 1
#else
#endif

#ifndef asm
#define asm __asm__
#endif

#ifdef NOT_PLD
asm(
    ".macro pld reg1,reg2 \n"
    "nop \n"
    ".endm \n");
#endif


#ifndef NULL
#define NULL (void*)0
#endif

#define MALLOC_ALIGN 16
#define MALLOC_OVERREAD 64

#define PoolMethod_MAX 0
#define PoolMethod_AVE 1

#define MIN(a, b) (((a) <= (b)) ? (a) : (b))
#define MAX(a, b) (((a) >= (b)) ? (a) : (b))

#define newMat(data, w, h, c)                                  \
    {                                                            \
        (data), (w), (h), (c), (alignPtr((w)* (h), MALLOC_ALIGN)) \
    } // cstep 按channel内存对齐
typedef struct {
    float* data;
    int w;
    int h;
    int c;
    int cstep;
} Mat;
typedef struct {
    unsigned char* data;
    int w;
    int h;
    int c;
    int cstep;
} MatImg;

#if RUN_TST
void mat_print(Mat m);
void mat_printp(Mat* m);
void mat_print_c0(Mat m);
void float_print_n(float* f, int n);
#endif


/*---------------------------------------静态函数声明-----------------------------------------*/
PUBLIC void fill(float* src, float _v, int size);
PUBLIC void deconv4x4s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias);
PUBLIC void copy_make_border_image(const Mat src, Mat dst, int top, int left, int type, float v);
PUBLIC void linear_coeffs(int w, int outw, int* xofs, float* alpha);
PUBLIC void get_warp_coeffs(int* buf, float theta, int patch_size, int sample_size, float nX, float nY);
PUBLIC void get_warp_coeffs_rect(int* buf, float theta, int *patch_size, int *sample_size, float nX, float nY);
PUBLIC void resize_bilinear_image(const Mat src, Mat dst, float* alpha, int* xofs, float* beta, int* yofs, int flag);


PUBLIC size_t alignPtr( size_t ptr, int n );

PUBLIC float* channel( Mat mat, int _c );

PUBLIC int total( Mat mat );

PUBLIC float* new_memory( Mat mat );

PUBLIC void flatten( Mat mat, float* data );


PUBLIC void totensor_neon( float* src, float* dst, int size );

// conv1x1s1
PUBLIC void conv1x1s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias );

// conv3x3dws1
PUBLIC void convdw3x3s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias );

// conv3x3dw2s2
PUBLIC void convdw3x3s2_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias );

PUBLIC void conv3x3s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias );

// conv3x3s2
PUBLIC void conv3x3s2_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias );


PUBLIC void conv5x5s1_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias);

// conv5x5s2
PUBLIC void conv5x5s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias);

// relu
PUBLIC void relu_neon( Mat mat );

// hswish
PUBLIC void hswish_neon( Mat bottom_top_blob );

// hsigmoid
PUBLIC void hsigmoid_neon( Mat bottom_top_blob );

// tanh
PUBLIC void tanh_neon(Mat mat);

// global pooling
PUBLIC void pooling_global( const Mat bottom_blob, Mat top_blob, int pooling_type );

PUBLIC void pooling2x2s2_max_neon( const Mat bottom_blob, Mat top_blob );

PUBLIC void groupconv3x3s1_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias, const int group);

PUBLIC void groupconv3x3s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias, const int group);

// deconv4x4s2 stride=2 ->deconv4x4_block();

// padding_copy ->padding();

PUBLIC void padding( const Mat bottom_blob, Mat top_blob, int top, int left, int type, float v );

PUBLIC void padding_normal(Mat ori_blob, Mat pad_blob, const int pad);

PUBLIC void soft_max( const float* src, int channel, float* dst );

// matrix_add
PUBLIC void mat_add_neon_inplace( Mat bottom_top_blob, Mat add_blob );

// matrix_multiply ->bilinear_neon_cnn();
PUBLIC void mat_scale_neon_inplace( Mat bottom_top_blob, Mat scale_blob );

// flag 1:int img

PUBLIC int bilinear_neon_cnn(const Mat bottom_blob, Mat top_blob, int flag);

// deconv3x3s2
PUBLIC void deconv3x3s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias);

PUBLIC void deconvcrop(Mat ori_blob, Mat crop_blob, int pad);

PUBLIC int bilinear_warp_neon(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int patch_size, int sample_size);

PUBLIC int bilinear_warp_neon_rect(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int *patch_size, int *sample_size);

PUBLIC int bilinear_doublewarp_neon(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int patch_size, int sample_size1, int sample_size2);
// get std
PUBLIC float std_neon(float* src, float mean, int size);

// LeakyRelu
PUBLIC void leakyrelu_neon(Mat mat, float slope);

PUBLIC void normalize_neon(float* src, float* dst, const float mean, const float std, int size);

PUBLIC void copy(const Mat src, Mat dst);

PUBLIC int short_to_float(float* param_f, short* param_s, int len, int expansion);

PUBLIC void* sl_aligned_malloc(size_t size, size_t align);

PUBLIC void sl_aligned_free(void* p);


#if 0
PUBLIC void* fast_malloc(size_t size);

PUBLIC void fast_free(void* ptr);

#endif
#endif