#include "net_cnn_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SL_Math.h"



/*---------------------------------------静态函数定义-----------------------------------------*/

PUBLIC void fill( float* src, float _v, int size )
// PUBLIC __inline void fill( float* src, float _v, int size )
{
    float* ptr = src;

#if NEON // NEON
    int nn = size >> 2;
    int remain = size - ( nn << 2 );
#else
    int remain = size;
#endif // NEON

#if NEON // NEON
    float32x4_t _c = vdupq_n_f32( _v );

#if __aarch64__

    if ( nn > 0 ) {
        asm volatile(
            "0:                             \n"
            "subs       %w0, %w0, #1        \n"
            "st1        {%4.4s}, [%1], #16  \n"
            "bne        0b                  \n"
            : "=r"( nn ), // %0
            "=r"( ptr ) // %1
            : "0"( nn ),
            "1"( ptr ),
            "w"( _c ) // %4
            : "cc", "memory" );
    }

#else

    if ( nn > 0 ) {
        asm volatile(
            "0:                             \n"
            "subs       %0, #1              \n"
            "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
            "bne        0b                  \n"
            : "=r"( nn ), // %0
            "=r"( ptr ) // %1
            : "0"( nn ),
            "1"( ptr ),
            "w"( _c ) // %4
            : "cc", "memory" );
    }

#endif // __aarch64__
#endif

    // NEON
    for ( ; remain > 0; remain-- ) {
        *ptr++ = _v;
    }
}
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#if NEON

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
PRIVATE inline float32x4_t exp_ps(float32x4_t x)
{
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    PRIVATE const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
    float32x4_t y = vld1q_dup_f32(cephes_exp_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p + 4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p + 5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x, x);

    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

PRIVATE inline float32x4_t div_ps(float32x4_t a, float32x4_t b)
{
#if __aarch64__
    return vdivq_f32(a, b);
#else
    float32x4_t reciprocal = vrecpeq_f32(b);
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    // reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    return vmulq_f32(a, reciprocal);
#endif
}

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1 0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
PRIVATE inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t mask_l = vcgeq_f32(x2, vdupq_n_f32(c_cephes_tanh_C1));
    uint32x4_t mask_l2 = vcgtq_f32(x2, vdupq_n_f32(c_cephes_HALFMAXLOGF));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _two = vdupq_n_f32(2.f);
    float32x4_t exp_x_x = exp_ps(vaddq_f32(x, x));
#if __aarch64__
    float32x4_t y0 = vsubq_f32(_one, vdivq_f32(_two, vaddq_f32(exp_x_x, _one)));
#else
    float32x4_t y0 = vsubq_f32(_one, div_ps(_two, vaddq_f32(exp_x_x, _one)));
#endif

    // abs(x) < 0.625
    /*
    z = x2 * x2;
    z =
    (((( -5.70498872745E-3 * z
    + 2.06390887954E-2) * z
    - 5.37397155531E-2) * z
    + 1.33314422036E-1) * z
    - 3.33332819422E-1) * z * x
    + x;
    */
    PRIVATE const float cephes_tanh_p[5] = { c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4 };
    float32x4_t y = vld1q_dup_f32(cephes_tanh_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_tanh_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_tanh_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_tanh_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_tanh_p + 4);

    float32x4_t z = vmulq_f32(x, x);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c4);

    y = vmulq_f32(y, z);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint32x4_t mask_pos = vcgtq_f32(x, vdupq_n_f32(0.f));
    float32x4_t y1 = vreinterpretq_f32_u32(vbslq_u32(mask_pos, vreinterpretq_u32_f32(vdupq_n_f32(1.f)), vreinterpretq_u32_f32(vdupq_n_f32(-1.f))));

    y = vreinterpretq_f32_u32(vbslq_u32(mask_l, vreinterpretq_u32_f32(y0), vreinterpretq_u32_f32(y)));
    y = vreinterpretq_f32_u32(vbslq_u32(mask_l2, vreinterpretq_u32_f32(y1), vreinterpretq_u32_f32(y)));
    return y;
}

PRIVATE inline float32x4_t sigmoid_ps(float32x4_t _v)
{
    float32x4_t _one = vdupq_n_f32(1.f);
    _v = vnegq_f32(_v);
    _v = exp_ps(_v);
    _v = vaddq_f32(_v, _one);
    float32x4_t _outp = vrecpeq_f32(_v);
    // _outp = vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
    return vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
}
#endif
PRIVATE float SL_Tanh( float x )
{
    float ret, temp;
    ret = (float)SL_exp(2 * x);
    temp = 2.0f / (ret + 1);
    return ( 1 - temp );
}

/*-----------------------非静态函数-----------------------------*/

#if RUN_TST
void mat_print(Mat m)
{
    int q, y, x;

    for (q = 0; q < m.c; q++) {
        const float* ptr = channel(m, q);

        for (y = 0; y < m.h; y++) {
            for (x = 0; x < m.w; x++) {
                printf("%f ", ptr[x]);
            }

            ptr += m.w;
            printf("\n");
        }

        printf("----------mat .9f end--------------\n");
    }
}
void mat_printp(Mat* m)
{
    int q, y, x;

    for (q = 0; q < m->c; q++) {
        const float* ptr = channel(*m, q);

        for (y = 0; y < m->h; y++) {
            for (x = 0; x < m->w; x++) {
                printf("%f ", ptr[x]);
            }

            ptr += m->w;
            printf("\n");
        }
    }

    printf("----------matp c0 end--------------\n");
}
void mat_print_c0(Mat m)
{
    int y, x;


    const float* ptr = channel(m, 0);

    for (y = 0; y < m.h; y++) {
        for (x = 0; x < m.w; x++) {
            printf("%f ", ptr[x]);
        }

        ptr += m.w;
        printf("\n");
    }

    printf("----------mat c0 .8f end--------------\n");
}
void float_print_n(float* f, int n)
{
    int i = 0;

    for (; i < n; i++) {
        printf("%f ", f[i]);

        if (i % 20 == 19) {
            printf("\n");
        }
    }

    printf("----------float n end--------------\n");
}

#endif


PUBLIC size_t alignPtr( size_t ptr, int n )
{
    return ( ( ptr + n - 1 ) & -n );
}
PUBLIC float* channel( Mat mat, int _c )
{
    // int str = alignPtr((int)(mat.data + mat.cstep * _c), MALLOC_ALIGN);
    // return (float *)str;
    return mat.data + mat.cstep * _c;
}

PUBLIC int total( Mat mat )
{
    return mat.c * mat.cstep;
}

PUBLIC float* new_memory( Mat mat )
{
    return mat.data + mat.c * mat.cstep;
}

PUBLIC void flatten( Mat mat, float* data )
{
    int inch = mat.c;
    int w = mat.w;
    int h = mat.h;
    int size = w * h;
    int i;

    for ( i = 0; i < inch; i++ ) {
        float* img = channel( mat, i );
        memcpy( data + i * size, img, size * sizeof( float ) );
    }
}

PUBLIC void totensor_neon( float* src, float* dst, int size )
{
    int i = 0;
#if NEON
    const float* img_in = src;
    float* img_out = dst;
    float32x4_t _grayscale = vdupq_n_f32( 1 / 255.f );
    float32x4_t sub;

    for ( i = 0; i <= size - 4; i += 4 ) {
        sub = vmulq_f32( vld1q_f32( img_in ), _grayscale );
        vst1q_f32( img_out, sub );
        img_in += 4;
        img_out += 4;
    }

#endif

    for ( ; i < size; i++ ) {
        *(dst + i) = *(src + i) / 255.0f;
    }
}

// conv1x1s1
PUBLIC void conv1x1s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias )
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = ( outch - remain_outch_start ) >> 2;
    int pp = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( pp = 0; pp < nn_outch; pp++ ) {
        int p = remain_outch_start + pp * 4;

        // Mat out0 = top_blob.channel(p);
        // Mat out1 = top_blob.channel(p+1);
        // Mat out2 = top_blob.channel(p+2);
        // Mat out3 = top_blob.channel(p+3);

        float* out0 = channel( top_blob, p );
        float* out1 = channel( top_blob, p + 1 );
        float* out2 = channel( top_blob, p + 2 );
        float* out3 = channel( top_blob, p + 3 );

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;

        // out0.fill(bias0);
        // out1.fill(bias1);
        // out2.fill(bias2);
        // out3.fill(bias3);

        fill( out0, bias0, top_blob.cstep );
        fill( out1, bias1, top_blob.cstep );
        fill( out2, bias2, top_blob.cstep );
        fill( out3, bias3, top_blob.cstep );

        int q = 0;

        for ( ; q + 3 < inch; q += 4 ) {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            // const float* img0 = bottom_blob.channel(q);
            // const float* img1 = bottom_blob.channel(q+1);
            // const float* img2 = bottom_blob.channel(q+2);
            // const float* img3 = bottom_blob.channel(q+3);

            const float* img0 = channel( bottom_blob, q );
            const float* img1 = channel( bottom_blob, q + 1 );
            const float* img2 = channel( bottom_blob, q + 2 );
            const float* img3 = channel( bottom_blob, q + 3 );

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + ( p + 1 ) * inch + q;
            const float* kernel2 = kernel + ( p + 2 ) * inch + q;
            const float* kernel3 = kernel + ( p + 3 ) * inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // NEON

#if NEON
            float32x4_t _k0 = vld1q_f32( kernel0 );
            float32x4_t _k1 = vld1q_f32( kernel1 );
            float32x4_t _k2 = vld1q_f32( kernel2 );
            float32x4_t _k3 = vld1q_f32( kernel3 );

#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v8.4s, v9.4s}, [%1]        \n"

                    "0:                                 \n"

                    "fmla   v8.4s, v6.4s, %18.s[0]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v10.4s, v11.4s}, [%2]      \n"

                    "fmla   v9.4s, v7.4s, %18.s[0]      \n"

                    "fmla   v10.4s, v6.4s, %19.s[0]     \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v12.4s, v13.4s}, [%3]      \n"

                    "fmla   v11.4s, v7.4s, %19.s[0]     \n"

                    "fmla   v12.4s, v6.4s, %20.s[0]     \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v14.4s, v15.4s}, [%4]      \n"

                    "fmla   v13.4s, v7.4s, %20.s[0]     \n"

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%6], #32   \n"

                    "fmla   v14.4s, v6.4s, %21.s[0]     \n"
                    "fmla   v15.4s, v7.4s, %21.s[0]     \n"

                    "fmla   v8.4s, v4.4s, %18.s[1]      \n"
                    "fmla   v9.4s, v5.4s, %18.s[1]      \n"

                    "fmla   v10.4s, v4.4s, %19.s[1]     \n"
                    "fmla   v11.4s, v5.4s, %19.s[1]     \n"

                    "fmla   v12.4s, v4.4s, %20.s[1]     \n"
                    "fmla   v13.4s, v5.4s, %20.s[1]     \n"

                    "prfm   pldl1keep, [%7, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%7], #32   \n"

                    "fmla   v14.4s, v4.4s, %21.s[1]     \n"
                    "fmla   v15.4s, v5.4s, %21.s[1]     \n"

                    "fmla   v8.4s, v6.4s, %18.s[2]      \n"
                    "fmla   v9.4s, v7.4s, %18.s[2]      \n"

                    "fmla   v10.4s, v6.4s, %19.s[2]     \n"
                    "fmla   v11.4s, v7.4s, %19.s[2]     \n"

                    "fmla   v12.4s, v6.4s, %20.s[2]     \n"
                    "fmla   v13.4s, v7.4s, %20.s[2]     \n"

                    "prfm   pldl1keep, [%8, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%8], #32   \n"

                    "fmla   v14.4s, v6.4s, %21.s[2]     \n"
                    "fmla   v15.4s, v7.4s, %21.s[2]     \n"

                    "fmla   v8.4s, v4.4s, %18.s[3]      \n"
                    "fmla   v9.4s, v5.4s, %18.s[3]      \n"

                    "fmla   v10.4s, v4.4s, %19.s[3]     \n"
                    "fmla   v11.4s, v5.4s, %19.s[3]     \n"

                    "st1    {v8.4s, v9.4s}, [%1], #32   \n"

                    "fmla   v12.4s, v4.4s, %20.s[3]     \n"
                    "fmla   v13.4s, v5.4s, %20.s[3]     \n"

                    "st1    {v10.4s, v11.4s}, [%2], #32 \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                    "fmla   v14.4s, v4.4s, %21.s[3]     \n"
                    "fmla   v15.4s, v5.4s, %21.s[3]     \n"

                    "st1    {v12.4s, v13.4s}, [%3], #32 \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v8.4s, v9.4s}, [%1]        \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                    "bne    0b                          \n"
                    "sub    %5, %5, #32                 \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr0 ), // %1
                    "=r"( outptr1 ), // %2
                    "=r"( outptr2 ), // %3
                    "=r"( outptr3 ), // %4
                    "=r"( r0 ),    // %5
                    "=r"( r1 ),    // %6
                    "=r"( r2 ),    // %7
                    "=r"( r3 )     // %8
                    : "0"( nn ),
                    "1"( outptr0 ),
                    "2"( outptr1 ),
                    "3"( outptr2 ),
                    "4"( outptr3 ),
                    "5"( r0 ),
                    "6"( r1 ),
                    "7"( r2 ),
                    "8"( r3 ),
                    "w"( _k0 ), // %18
                    "w"( _k1 ), // %19
                    "w"( _k2 ), // %20
                    "w"( _k3 ) // %21
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"
                    "0:                                 \n"

                    "vmla.f32   q8, q6, %e18[0]         \n"

                    "pld        [%2, #256]              \n"
                    "vld1.f32   {d20-d23}, [%2 :128]    \n"
                    "vmla.f32   q9, q7, %e18[0]         \n"

                    "vmla.f32   q10, q6, %e19[0]        \n"

                    "pld        [%3, #256]              \n"
                    "vld1.f32   {d24-d27}, [%3 :128]    \n"
                    "vmla.f32   q11, q7, %e19[0]        \n"

                    "vmla.f32   q12, q6, %e20[0]        \n"

                    "pld        [%4, #256]              \n"
                    "vld1.f32   {d28-d31}, [%4 :128]    \n"
                    "vmla.f32   q13, q7, %e20[0]        \n"

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d8-d11}, [%6 :128]!    \n"

                    "vmla.f32   q14, q6, %e21[0]        \n"
                    "vmla.f32   q15, q7, %e21[0]        \n"

                    "vmla.f32   q8, q4, %e18[1]         \n"
                    "vmla.f32   q9, q5, %e18[1]         \n"

                    "vmla.f32   q10, q4, %e19[1]        \n"
                    "vmla.f32   q11, q5, %e19[1]        \n"

                    "vmla.f32   q12, q4, %e20[1]        \n"
                    "vmla.f32   q13, q5, %e20[1]        \n"

                    "pld        [%7, #256]              \n"
                    "vld1.f32   {d12-d15}, [%7 :128]!   \n"

                    "vmla.f32   q14, q4, %e21[1]        \n"
                    "vmla.f32   q15, q5, %e21[1]        \n"

                    "vmla.f32   q8, q6, %f18[0]         \n"
                    "vmla.f32   q9, q7, %f18[0]         \n"

                    "vmla.f32   q10, q6, %f19[0]        \n"
                    "vmla.f32   q11, q7, %f19[0]        \n"

                    "vmla.f32   q12, q6, %f20[0]        \n"
                    "vmla.f32   q13, q7, %f20[0]        \n"

                    "pld        [%8, #256]              \n"
                    "vld1.f32   {d8-d11}, [%8 :128]!    \n"

                    "vmla.f32   q14, q6, %f21[0]        \n"
                    "vmla.f32   q15, q7, %f21[0]        \n"

                    "vmla.f32   q8, q4, %f18[1]         \n"
                    "vmla.f32   q9, q5, %f18[1]         \n"

                    "vmla.f32   q10, q4, %f19[1]        \n"
                    "vmla.f32   q11, q5, %f19[1]        \n"

                    "vmla.f32   q12, q4, %f20[1]        \n"
                    "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                    "vmla.f32   q13, q5, %f20[1]        \n"

                    "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                    "vmla.f32   q14, q4, %f21[1]        \n"
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"

                    "vmla.f32   q15, q5, %f21[1]        \n"

                    "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"

                    "subs       %0, #1                  \n"
                    "vst1.f32   {d28-d31}, [%4 :128]!   \n"

                    "bne        0b                      \n"
                    "sub        %5, #32                 \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr0 ), // %1
                    "=r"( outptr1 ), // %2
                    "=r"( outptr2 ), // %3
                    "=r"( outptr3 ), // %4
                    "=r"( r0 ),    // %5
                    "=r"( r1 ),    // %6
                    "=r"( r2 ),    // %7
                    "=r"( r3 )     // %8
                    : "0"( nn ),
                    "1"( outptr0 ),
                    "2"( outptr1 ),
                    "3"( outptr2 ),
                    "4"( outptr3 ),
                    "5"( r0 ),
                    "6"( r1 ),
                    "7"( r2 ),
                    "8"( r3 ),
                    "w"( _k0 ), // %18
                    "w"( _k1 ), // %19
                    "w"( _k2 ), // %20
                    "w"( _k3 ) // %21
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
            }

#endif // __aarch64__
#endif // NEON

            for ( ; remain > 0; remain-- ) {
                // TODO neon optimize
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }

        for ( ; q < inch; q++ ) {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            // const float* img0 = bottom_blob.channel(q);
            const float* img0 = channel( bottom_blob, q );

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + ( p + 1 ) * inch + q;
            const float* kernel2 = kernel + ( p + 2 ) * inch + q;
            const float* kernel3 = kernel + ( p + 3 ) * inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float* r0 = img0;

            int size = outw * outh;

#if NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // NEON

#if NEON
            float32x4_t _k0 = vdupq_n_f32( k0 );
            float32x4_t _k1 = vdupq_n_f32( k1 );
            float32x4_t _k2 = vdupq_n_f32( k2 );
            float32x4_t _k3 = vdupq_n_f32( k3 );
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"
                    "fmla       v8.4s, v6.4s, %12.4s           \n"
                    "fmla       v9.4s, v7.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"
                    "fmla       v10.4s, v6.4s, %13.4s          \n"
                    "fmla       v11.4s, v7.4s, %13.4s          \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"
                    "fmla       v12.4s, v6.4s, %14.4s          \n"
                    "fmla       v13.4s, v7.4s, %14.4s          \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"
                    "fmla       v14.4s, v6.4s, %15.4s          \n"
                    "fmla       v15.4s, v7.4s, %15.4s          \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                    "bne        0b                             \n"
                    "sub        %5, %5, #32                    \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr0 ), // %1
                    "=r"( outptr1 ), // %2
                    "=r"( outptr2 ), // %3
                    "=r"( outptr3 ), // %4
                    "=r"( r0 )     // %5
                    : "0"( nn ),
                    "1"( outptr0 ),
                    "2"( outptr1 ),
                    "3"( outptr2 ),
                    "4"( outptr3 ),
                    "5"( r0 ),
                    "w"( _k0 ), // %12
                    "w"( _k1 ), // %13
                    "w"( _k2 ), // %14
                    "w"( _k3 ) // %15
                    : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "0:                                 \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"
                    "vmla.f32   q8, q6, %q12            \n"
                    "vmla.f32   q9, q7, %q12            \n"

                    "pld        [%2, #256]              \n"
                    "vld1.f32   {d20-d23}, [%2 :128]    \n"
                    "vmla.f32   q10, q6, %q13           \n"
                    "vmla.f32   q11, q7, %q13           \n"

                    "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                    "pld        [%3, #256]              \n"
                    "vld1.f32   {d24-d27}, [%3 :128]    \n"
                    "vmla.f32   q12, q6, %q14           \n"
                    "vmla.f32   q13, q7, %q14           \n"

                    "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                    "pld        [%4, #256]              \n"
                    "vld1.f32   {d28-d31}, [%4 :128]    \n"
                    "vmla.f32   q14, q6, %q15           \n"
                    "vmla.f32   q15, q7, %q15           \n"

                    "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "subs       %0, #1                  \n"
                    "vst1.f32   {d28-d31}, [%4 :128]!   \n"
                    "bne        0b                      \n"
                    "sub        %5, #32                 \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr0 ), // %1
                    "=r"( outptr1 ), // %2
                    "=r"( outptr2 ), // %3
                    "=r"( outptr3 ), // %4
                    "=r"( r0 )     // %5
                    : "0"( nn ),
                    "1"( outptr0 ),
                    "2"( outptr1 ),
                    "3"( outptr2 ),
                    "4"( outptr3 ),
                    "5"( r0 ),
                    "w"( _k0 ), // %12
                    "w"( _k1 ), // %13
                    "w"( _k2 ), // %14
                    "w"( _k3 ) // %15
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
            }

#endif // __aarch64__
#endif // NEON

            for ( ; remain > 0; remain-- ) {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    remain_outch_start += nn_outch << 2;

    // #pragma omp parallel for num_threads(opt.num_threads)
    int p = remain_outch_start;

    for ( p = remain_outch_start; p < outch; p++ ) {
        // Mat out = top_blob.channel(p);
        float* out = channel( top_blob, p );

        float bias0 = bias ? bias[p] : 0.f;

        // out.fill(bias0);
        fill( out, bias0, top_blob.cstep );

        int q = 0;

        for ( ; q + 3 < inch; q += 4 ) {
            float* outptr = out;

            // const float* img0 = bottom_blob.channel(q);
            // const float* img1 = bottom_blob.channel(q+1);
            // const float* img2 = bottom_blob.channel(q+2);
            // const float* img3 = bottom_blob.channel(q+3);

            const float* img0 = channel( bottom_blob, q );
            const float* img1 = channel( bottom_blob, q + 1 );
            const float* img2 = channel( bottom_blob, q + 2 );
            const float* img3 = channel( bottom_blob, q + 3 );

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // NEON

#if NEON
            float32x4_t _k0 = vdupq_n_f32( k0 );
            float32x4_t _k1 = vdupq_n_f32( k1 );
            float32x4_t _k2 = vdupq_n_f32( k2 );
            float32x4_t _k3 = vdupq_n_f32( k3 );
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %12.4s           \n"
                    "fmla       v1.4s, v3.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%3], #32      \n"
                    "fmla       v0.4s, v2.4s, %13.4s           \n"
                    "fmla       v1.4s, v3.4s, %13.4s           \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%4], #32      \n"
                    "fmla       v0.4s, v2.4s, %14.4s           \n"
                    "fmla       v1.4s, v3.4s, %14.4s           \n"

                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%5], #32      \n"
                    "fmla       v0.4s, v2.4s, %15.4s           \n"
                    "fmla       v1.4s, v3.4s, %15.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 ),   // %4
                    "=r"( r3 )    // %5
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "5"( r3 ),
                    "w"( _k0 ), // %12
                    "w"( _k1 ), // %13
                    "w"( _k2 ), // %14
                    "w"( _k3 ) // %15
                    : "cc", "memory", "v0", "v1", "v2", "v3" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]  \n"
                    "vmla.f32   q0, q2, %q12        \n"
                    "vmla.f32   q1, q3, %q12        \n"
                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d4-d7}, [%3 :128]! \n"
                    "vmla.f32   q0, q2, %q13        \n"
                    "vmla.f32   q1, q3, %q13        \n"
                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d4-d7}, [%4 :128]! \n"
                    "vmla.f32   q0, q2, %q14        \n"
                    "vmla.f32   q1, q3, %q14        \n"
                    "pld        [%5, #256]          \n"
                    "vld1.f32   {d4-d7}, [%5 :128]! \n"
                    "vmla.f32   q0, q2, %q15        \n"
                    "vmla.f32   q1, q3, %q15        \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 ),   // %4
                    "=r"( r3 )    // %5
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "5"( r3 ),
                    "w"( _k0 ), // %12
                    "w"( _k1 ), // %13
                    "w"( _k2 ), // %14
                    "w"( _k3 ) // %15
                    : "cc", "memory", "q0", "q1", "q2", "q3" );
            }

#endif // __aarch64__
#endif // NEON

            for ( ; remain > 0; remain-- ) {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }
        }

        for ( ; q < inch; q++ ) {
            float* outptr = out;

            // const float* img0 = bottom_blob.channel(q);

            const float* img0 = channel( bottom_blob, q );

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

#if NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // NEON

#if NEON
            float32x4_t _k0 = vdupq_n_f32( k0 );
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %6.4s            \n"
                    "fmla       v1.4s, v3.4s, %6.4s            \n"
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 )    // %2
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "w"( _k0 ) // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]  \n"
                    "vmla.f32   q0, q2, %q6         \n"
                    "vmla.f32   q1, q3, %q6         \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 )    // %2
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "w"( _k0 ) // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3" );
            }

#endif // __aarch64__
#endif // NEON

            for ( ; remain > 0; remain-- ) {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
        }
    }
}

// conv3x3dws1
PUBLIC void convdw3x3s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias )
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;
    int g = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( g = 0; g < group; g++ ) {
        // Mat out = top_blob.channel(g);
        float* out = channel( top_blob, g );

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr = out;
        float* outptr2 = outptr + outw;

        // const float* img0 = bottom_blob.channel(g);
        const float* img0 = channel( bottom_blob, g );

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;
        const float* r3 = img0 + w * 3;

#if NEON
        float32x4_t _k012x = vld1q_f32( kernel0 );
        float32x4_t _k345x = vld1q_f32( kernel0 + 3 );
        float32x4_t _k678x = vld1q_f32( kernel0 + 6 );

        _k012x = vsetq_lane_f32( 0.f, _k012x, 3 );
        _k345x = vsetq_lane_f32( 0.f, _k345x, 3 );
        _k678x = vsetq_lane_f32( 0.f, _k678x, 3 );

        float32x4_t _bias0 = vdupq_n_f32( bias0 );
#else
        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

        int i = 0;

        for ( ; i + 1 < outh; i += 2 ) {

#if NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n" // r0
                    "add    %3, %3, #32                     \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "0:                                     \n"

                    "and    v4.16b, %17.16b, %17.16b        \n" // v4 = _bias0
                    "and    v5.16b, %17.16b, %17.16b        \n" // v5 = _bias0

                    "prfm   pldl1keep, [%6, #384]           \n"
                    "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n" // r3
                    "add    %6, %6, #32                     \n"

                    "and    v6.16b, %17.16b, %17.16b        \n" // v6 = _bias0
                    "and    v7.16b, %17.16b, %17.16b        \n" // v7 = _bias0

                    "ext    v15.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v8.4s, %14.s[0]          \n"
                    "fmla   v5.4s, v9.4s, %14.s[0]          \n"

                    "ext    v20.16b, v17.16b, v18.16b, #4   \n"

                    "fmla   v6.4s, v16.4s, %16.s[0]         \n"
                    "fmla   v7.4s, v17.4s, %16.s[0]         \n"

                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v4.4s, v11.4s, %14.s[1]         \n"
                    "fmla   v5.4s, v13.4s, %14.s[1]         \n"

                    "ext    v21.16b, v17.16b, v18.16b, #8   \n"

                    "fmla   v6.4s, v15.4s, %16.s[1]         \n"
                    "fmla   v7.4s, v20.4s, %16.s[1]         \n"

                    "prfm   pldl1keep, [%4, #384]           \n"
                    "ld1    {v22.4s, v23.4s, v24.4s}, [%4]  \n" // r1

                    "fmla   v4.4s, v12.4s, %14.s[2]         \n"
                    "fmla   v5.4s, v14.4s, %14.s[2]         \n"

                    "add    %4, %4, #32                     \n"

                    "fmla   v6.4s, v19.4s, %16.s[2]         \n"
                    "fmla   v7.4s, v21.4s, %16.s[2]         \n"

                    "ext    v25.16b, v22.16b, v23.16b, #4   \n"

                    "fmla   v4.4s, v22.4s, %15.s[0]         \n"
                    "fmla   v5.4s, v23.4s, %15.s[0]         \n"

                    "ext    v27.16b, v23.16b, v24.16b, #4   \n"

                    "fmla   v6.4s, v22.4s, %14.s[0]         \n"
                    "fmla   v7.4s, v23.4s, %14.s[0]         \n"

                    "ext    v26.16b, v22.16b, v23.16b, #8   \n"

                    "fmla   v4.4s, v25.4s, %15.s[1]         \n"
                    "fmla   v5.4s, v27.4s, %15.s[1]         \n"

                    "ext    v28.16b, v23.16b, v24.16b, #8   \n"

                    "fmla   v6.4s, v25.4s, %14.s[1]         \n"
                    "fmla   v7.4s, v27.4s, %14.s[1]         \n"

                    "prfm   pldl1keep, [%5, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%5]    \n" // r2

                    "fmla   v4.4s, v26.4s, %15.s[2]         \n"
                    "fmla   v5.4s, v28.4s, %15.s[2]         \n"

                    "add    %5, %5, #32                     \n"

                    "fmla   v6.4s, v26.4s, %14.s[2]         \n"
                    "fmla   v7.4s, v28.4s, %14.s[2]         \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v4.4s, v8.4s, %16.s[0]          \n"
                    "fmla   v5.4s, v9.4s, %16.s[0]          \n"

                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"

                    "fmla   v6.4s, v8.4s, %15.s[0]          \n"
                    "fmla   v7.4s, v9.4s, %15.s[0]          \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v11.4s, %16.s[1]         \n"
                    "fmla   v5.4s, v13.4s, %16.s[1]         \n"

                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "fmla   v6.4s, v11.4s, %15.s[1]         \n"
                    "fmla   v7.4s, v13.4s, %15.s[1]         \n"

                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n" // r0 next loop

                    "fmla   v4.4s, v12.4s, %16.s[2]         \n"
                    "fmla   v5.4s, v14.4s, %16.s[2]         \n"

                    "add    %3, %3, #32                     \n"
                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v6.4s, v12.4s, %15.s[2]         \n"
                    "fmla   v7.4s, v14.4s, %15.s[2]         \n"

                    "ext    v13.16b, v9.16b, v10.16b, #4    \n"
                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "st1    {v4.4s, v5.4s}, [%1], #32       \n"

                    "ext    v14.16b, v9.16b, v10.16b, #8    \n"

                    "subs   %w0, %w0, #1                    \n"

                    "st1    {v6.4s, v7.4s}, [%2], #32       \n"

                    "bne    0b                              \n"
                    "sub    %3, %3, #32                     \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr ), // %1
                    "=r"( outptr2 ), // %2
                    "=r"( r0 ),    // %3
                    "=r"( r1 ),    // %4
                    "=r"( r2 ),    // %5
                    "=r"( r3 )     // %6
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( outptr2 ),
                    "3"( r0 ),
                    "4"( r1 ),
                    "5"( r2 ),
                    "6"( r3 ),
                    "w"( _k012x ), // %14
                    "w"( _k345x ), // %15
                    "w"( _k678x ), // %16
                    "w"( _bias0 ) // %17
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28" );
            }

            if ( remain >= 4 ) {
                remain -= 4;

                asm volatile(
                    "prfm   pldl1keep, [%2, #256]           \n"
                    "ld1    {v8.4s, v9.4s}, [%2]            \n" // r0
                    "add    %2, %2, #16                     \n"

                    "and    v4.16b, %15.16b, %15.16b        \n" // v4 = _bias0
                    "and    v6.16b, %15.16b, %15.16b        \n" // v6 = _bias0

                    "prfm   pldl1keep, [%5, #256]           \n"
                    "ld1    {v16.4s, v17.4s}, [%5]          \n" // r3
                    "add    %5, %5, #16                     \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v15.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v8.4s, %12.s[0]          \n"
                    "fmla   v6.4s, v16.4s, %14.s[0]         \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"
                    "ext    v19.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v4.4s, v11.4s, %12.s[1]         \n"
                    "fmla   v6.4s, v15.4s, %14.s[1]         \n"

                    "prfm   pldl1keep, [%3, #256]           \n"
                    "ld1    {v22.4s, v23.4s}, [%3]          \n" // r1

                    "fmla   v4.4s, v12.4s, %12.s[2]         \n"

                    "add    %3, %3, #16                     \n"

                    "fmla   v6.4s, v19.4s, %14.s[2]         \n"

                    "ext    v25.16b, v22.16b, v23.16b, #4   \n"

                    "fmla   v4.4s, v22.4s, %13.s[0]         \n"
                    "fmla   v6.4s, v22.4s, %12.s[0]         \n"

                    "ext    v26.16b, v22.16b, v23.16b, #8   \n"

                    "fmla   v4.4s, v25.4s, %13.s[1]         \n"
                    "fmla   v6.4s, v25.4s, %12.s[1]         \n"

                    "prfm   pldl1keep, [%4, #256]           \n"
                    "ld1    {v8.4s, v9.4s}, [%4]            \n" // r2

                    "fmla   v4.4s, v26.4s, %13.s[2]         \n"

                    "add    %4, %4, #16                     \n"

                    "fmla   v6.4s, v26.4s, %12.s[2]         \n"

                    "ext    v11.16b, v8.16b, v9.16b, #4     \n"

                    "fmla   v4.4s, v8.4s, %14.s[0]          \n"
                    "fmla   v6.4s, v8.4s, %13.s[0]          \n"

                    "ext    v12.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v11.4s, %14.s[1]         \n"
                    "fmla   v6.4s, v11.4s, %13.s[1]         \n"

                    "fmla   v4.4s, v12.4s, %14.s[2]         \n"
                    "fmla   v6.4s, v12.4s, %13.s[2]         \n"

                    "st1    {v4.4s}, [%0], #16              \n"
                    "st1    {v6.4s}, [%1], #16              \n"

                    : "=r"( outptr ), // %0
                    "=r"( outptr2 ), // %1
                    "=r"( r0 ),    // %2
                    "=r"( r1 ),    // %3
                    "=r"( r2 ),    // %4
                    "=r"( r3 )     // %5
                    : "0"( outptr ),
                    "1"( outptr2 ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "5"( r3 ),
                    "w"( _k012x ), // %12
                    "w"( _k345x ), // %13
                    "w"( _k678x ), // %14
                    "w"( _bias0 ) // %15
                    : "cc", "memory", "v4", "v6", "v8", "v9", "v11", "v12", "v15", "v16", "v17", "v18", "v19", "v22", "v23", "v25", "v26" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n" // r0
                    "add        %3, #16             \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "0:                             \n"

                    "vmul.f32   q7, q9, %e14[0]     \n"

                    "vand       q13, %q17, %q17     \n" // q13 = _bias0
                    "vmul.f32   q6, q11, %e14[1]    \n"
                    "vmla.f32   q13, q12, %f14[0]   \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d18-d20}, [%4]     \n" // r1
                    "add        %4, #16             \n"

                    "vmla.f32   q7, q9, %e15[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q6, q11, %e15[1]    \n"
                    "vmla.f32   q13, q12, %f15[0]   \n"

                    "vmul.f32   q8, q9, %e14[0]     \n"

                    "vand       q15, %q17, %q17     \n" // q15 = _bias0
                    "vmul.f32   q14, q11, %e14[1]   \n"
                    "vmla.f32   q15, q12, %f14[0]   \n"

                    "pld        [%5, #192]          \n"
                    "vld1.f32   {d18-d20}, [%5 :64] \n" // r2
                    "add        %5, #16             \n"

                    "vmla.f32   q7, q9, %e16[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q6, q11, %e16[1]    \n"
                    "vmla.f32   q13, q12, %f16[0]   \n"

                    "vmla.f32   q8, q9, %e15[0]     \n"
                    "vmla.f32   q14, q11, %e15[1]   \n"
                    "vmla.f32   q15, q12, %f15[0]   \n"

                    "pld        [%6, #192]          \n"
                    "vld1.f32   {d18-d20}, [%6]     \n" // r3
                    "add        %6, #16             \n"

                    "vmla.f32   q8, q9, %e16[0]     \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "vmla.f32   q14, q11, %e16[1]   \n"
                    "vmla.f32   q15, q12, %f16[0]   \n"

                    "vadd.f32   q7, q7, q6          \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d18-d20}, [%3 :64] \n" // r0

                    "vadd.f32   q8, q8, q14         \n"
                    "vadd.f32   q7, q7, q13         \n"
                    "vadd.f32   q8, q8, q15         \n"

                    "vext.32    q11, q9, q10, #1    \n"
                    "vext.32    q12, q9, q10, #2    \n"

                    "add        %3, #16             \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"
                    "vst1.f32   {d16-d17}, [%2]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %3, #16             \n"
                    : "=r"( nn ),    // %0
                    "=r"( outptr ), // %1
                    "=r"( outptr2 ), // %2
                    "=r"( r0 ),    // %3
                    "=r"( r1 ),    // %4
                    "=r"( r2 ),    // %5
                    "=r"( r3 )     // %6
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( outptr2 ),
                    "3"( r0 ),
                    "4"( r1 ),
                    "5"( r2 ),
                    "6"( r3 ),
                    "w"( _k012x ), // %14
                    "w"( _k345x ), // %15
                    "w"( _k678x ), // %16
                    "w"( _bias0 ) // %17
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
            }

#endif // __aarch64__
#endif // __ARM_NEON

            for ( ; remain > 0; remain-- ) {
#if NEON
                float32x4_t _r00 = vld1q_f32( r0 );
                float32x4_t _r10 = vld1q_f32( r1 );
                float32x4_t _r20 = vld1q_f32( r2 );
                float32x4_t _r30 = vld1q_f32( r3 );

                float32x4_t _sum = vmulq_f32( _r00, _k012x );
                _sum = vmlaq_f32( _sum, _r10, _k345x );
                _sum = vmlaq_f32( _sum, _r20, _k678x );

                float32x4_t _sum2 = vmulq_f32( _r10, _k012x );
                _sum2 = vmlaq_f32( _sum2, _r20, _k345x );
                _sum2 = vmlaq_f32( _sum2, _r30, _k678x );

                _sum = vsetq_lane_f32( bias0, _sum, 3 );
                _sum2 = vsetq_lane_f32( bias0, _sum2, 3 );
#if __aarch64__
                *outptr = vaddvq_f32( _sum );
                *outptr2 = vaddvq_f32( _sum2 );
#else
                float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                float32x2_t _ss2 = vadd_f32( vget_low_f32( _sum2 ), vget_high_f32( _sum2 ) );

                float32x2_t _sss2 = vpadd_f32( _ss, _ss2 );

                *outptr = vget_lane_f32( _sss2, 0 );
                *outptr2 = vget_lane_f32( _sss2, 1 );
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                float sum2 = bias0;
                sum2 += r1[0] * k0[0];
                sum2 += r1[1] * k0[1];
                sum2 += r1[2] * k0[2];
                sum2 += r2[0] * k1[0];
                sum2 += r2[1] * k1[1];
                sum2 += r2[2] * k1[2];
                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];

                *outptr = sum;
                *outptr2 = sum2;
#endif
                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
                outptr2++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr += outw;
            outptr2 += outw;
        }

        for ( ; i < outh; i++ ) {

#if NEON
#if __aarch64__
            int nn = outw >> 3;
            int remain = outw & 7;
#else
            int nn = outw >> 2;
            int remain = outw & 3;
#endif // __aarch64__
#else
            int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm   pldl1keep, [%2, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n" // r0
                    "add    %2, %2, #32                     \n"

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #4    \n"

                    "0:                                     \n"

                    "fmul   v6.4s, v8.4s, %10.s[0]          \n"

                    "and    v4.16b, %13.16b, %13.16b        \n" // v4 = _bias0

                    "fmul   v7.4s, v9.4s, %10.s[0]          \n"

                    "and    v5.16b, %13.16b, %13.16b        \n" // v5 = _bias0

                    "fmla   v4.4s, v12.4s, %10.s[1]         \n"

                    "ext    v13.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v5.4s, v14.4s, %10.s[1]         \n"

                    "ext    v15.16b, v9.16b, v10.16b, #8    \n"

                    "fmla   v6.4s, v13.4s, %10.s[2]         \n"

                    "prfm   pldl1keep, [%3, #384]           \n"
                    "ld1    {v16.4s, v17.4s, v18.4s}, [%3]  \n" // r1

                    "fmla   v7.4s, v15.4s, %10.s[2]         \n"

                    "add    %3, %3, #32                     \n"

                    "fmla   v4.4s, v16.4s, %11.s[0]         \n"

                    "ext    v20.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v5.4s, v17.4s, %11.s[0]         \n"

                    "ext    v22.16b, v17.16b, v18.16b, #4   \n"

                    "fmla   v6.4s, v20.4s, %11.s[1]         \n"

                    "ext    v21.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v7.4s, v22.4s, %11.s[1]         \n"

                    "ext    v23.16b, v17.16b, v18.16b, #8   \n"

                    "fmla   v4.4s, v21.4s, %11.s[2]         \n"

                    "prfm   pldl1keep, [%4, #384]           \n"
                    "ld1    {v24.4s, v25.4s, v26.4s}, [%4]  \n" // r2

                    "fmla   v5.4s, v23.4s, %11.s[2]         \n"

                    "add    %4, %4, #32                     \n"

                    "fmla   v6.4s, v24.4s, %12.s[0]         \n"

                    "ext    v12.16b, v24.16b, v25.16b, #4   \n"

                    "fmla   v7.4s, v25.4s, %12.s[0]         \n"

                    "ext    v14.16b, v25.16b, v26.16b, #4   \n"

                    "fmla   v4.4s, v12.4s, %12.s[1]         \n"

                    "ext    v13.16b, v24.16b, v25.16b, #8   \n"

                    "fmla   v5.4s, v14.4s, %12.s[1]         \n"

                    "ext    v15.16b, v25.16b, v26.16b, #8   \n"

                    "fmla   v6.4s, v13.4s, %12.s[2]         \n"
                    "fmla   v7.4s, v15.4s, %12.s[2]         \n"

                    "prfm   pldl1keep, [%2, #384]           \n"
                    "ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n" // r0 next loop

                    "fadd   v4.4s, v4.4s, v6.4s             \n"

                    "add    %2, %2, #32                     \n"

                    "fadd   v5.4s, v5.4s, v7.4s             \n"

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"
                    "ext    v14.16b, v9.16b, v10.16b, #4    \n"

                    "subs   %w0, %w0, #1                    \n"

                    "st1    {v4.4s, v5.4s}, [%1], #32       \n"

                    "bne    0b                              \n"
                    "sub    %2, %2, #32                     \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 )    // %4
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "w"( _k012x ), // %10
                    "w"( _k345x ), // %11
                    "w"( _k678x ), // %12
                    "w"( _bias0 ) // %13
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26" );
            }

            if ( remain >= 4 ) {
                remain -= 4;

                asm volatile(
                    "prfm   pldl1keep, [%1, #192]           \n"
                    "ld1    {v8.4s, v9.4s}, [%1]            \n" // r0
                    "add    %1, %1, #16                     \n"

                    "and    v4.16b, %11.16b, %11.16b        \n" // v4 = _bias0

                    "ext    v12.16b, v8.16b, v9.16b, #4     \n"

                    "fmul   v6.4s, v8.4s, %8.s[0]           \n"

                    "ext    v13.16b, v8.16b, v9.16b, #8     \n"

                    "fmla   v4.4s, v12.4s, %8.s[1]          \n"

                    "prfm   pldl1keep, [%2, #192]           \n"
                    "ld1    {v16.4s, v17.4s}, [%2]          \n" // r1
                    "add    %2, %2, #16                     \n"

                    "fmla   v6.4s, v13.4s, %8.s[2]          \n"

                    "ext    v20.16b, v16.16b, v17.16b, #4   \n"

                    "fmla   v4.4s, v16.4s, %9.s[0]          \n"

                    "ext    v21.16b, v16.16b, v17.16b, #8   \n"

                    "fmla   v6.4s, v20.4s, %9.s[1]          \n"

                    "prfm   pldl1keep, [%3, #192]           \n"
                    "ld1    {v24.4s, v25.4s}, [%3]          \n" // r2
                    "add    %3, %3, #16                     \n"

                    "fmla   v4.4s, v21.4s, %9.s[2]          \n"

                    "ext    v12.16b, v24.16b, v25.16b, #4   \n"

                    "fmla   v6.4s, v24.4s, %10.s[0]         \n"

                    "ext    v13.16b, v24.16b, v25.16b, #8   \n"

                    "fmla   v4.4s, v12.4s, %10.s[1]         \n"

                    "fmla   v6.4s, v13.4s, %10.s[2]         \n"

                    "fadd   v4.4s, v4.4s, v6.4s             \n"

                    "st1    {v4.4s}, [%0], #16              \n"

                    : "=r"( outptr ), // %0
                    "=r"( r0 ),   // %1
                    "=r"( r1 ),   // %2
                    "=r"( r2 )    // %3
                    : "0"( outptr ),
                    "1"( r0 ),
                    "2"( r1 ),
                    "3"( r2 ),
                    "w"( _k012x ), // %8
                    "w"( _k345x ), // %9
                    "w"( _k678x ), // %10
                    "w"( _bias0 ) // %11
                    : "cc", "memory", "v4", "v6", "v8", "v9", "v12", "v13", "v16", "v17", "v20", "v21", "v24", "v25" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%2, #192]          \n"
                    "vld1.f32   {d16-d18}, [%2]     \n" // r0
                    "add        %2, #16             \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "0:                             \n"

                    "vmul.f32   q7, q8, %e10[0]     \n"

                    "vand       q14, %q13, %q13     \n" // q14 = _bias0
                    "vmul.f32   q13, q10, %e10[1]   \n"
                    "vmla.f32   q14, q11, %f10[0]   \n"

                    "pld        [%3, #192]          \n"
                    "vld1.f32   {d16-d18}, [%3]     \n" // r1
                    "add        %3, #16             \n"

                    "vmla.f32   q7, q8, %e11[0]     \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q13, q10, %e11[1]   \n"
                    "vmla.f32   q14, q11, %f11[0]   \n"

                    "pld        [%4, #192]          \n"
                    "vld1.f32   {d16-d18}, [%4]     \n" // r2
                    "add        %4, #16             \n"

                    "vmla.f32   q7, q8, %e12[0]     \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vmla.f32   q13, q10, %e12[1]   \n"
                    "vmla.f32   q14, q11, %f12[0]   \n"

                    "pld        [%2, #192]          \n"
                    "vld1.f32   {d16-d18}, [%2]     \n" // r0
                    "add        %2, #16             \n"

                    "vadd.f32   q7, q7, q13         \n"
                    "vadd.f32   q7, q7, q14         \n"

                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %2, #16             \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 )    // %4
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "w"( _k012x ), // %10
                    "w"( _k345x ), // %11
                    "w"( _k678x ), // %12
                    "w"( _bias0 ) // %13
                    : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
            }

#endif // __aarch64__
#endif // __ARM_NEON

            for ( ; remain > 0; remain-- ) {
#if NEON
                float32x4_t _r00 = vld1q_f32( r0 );
                float32x4_t _r10 = vld1q_f32( r1 );
                float32x4_t _r20 = vld1q_f32( r2 );

                float32x4_t _sum = vmulq_f32( _r00, _k012x );
                _sum = vmlaq_f32( _sum, _r10, _k345x );
                _sum = vmlaq_f32( _sum, _r20, _k678x );

                _sum = vsetq_lane_f32( bias0, _sum, 3 );
#if __aarch64__
                *outptr = vaddvq_f32( _sum );
#else
                float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                _ss = vpadd_f32( _ss, _ss );

                *outptr = vget_lane_f32( _ss, 0 );
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;
#endif
                r0++;
                r1++;
                r2++;
                outptr++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

// conv3x3dw2s2
PUBLIC void convdw3x3s2_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias )
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;
    int g = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( g = 0; g < group; g++ ) {
        // Mat out = top_blob.channel(g);
        float* out = channel( top_blob, g );

        const float bias0 = bias ? bias[g] : 0.f;

        const float* kernel0 = kernel + g * 9;

        float* outptr = out;

        // const float* img0 = bottom_blob.channel(g);
        const float* img0 = channel( bottom_blob, g );

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

#if NEON
        float32x4_t _k012x = vld1q_f32( kernel0 );
        float32x4_t _k345x = vld1q_f32( kernel0 + 3 );
        float32x4_t _k678x = vld1q_f32( kernel0 + 6 );

        _k012x = vsetq_lane_f32( 0.f, _k012x, 3 );
        _k345x = vsetq_lane_f32( 0.f, _k345x, 3 );
        _k678x = vsetq_lane_f32( 0.f, _k678x, 3 );

        float32x4_t _bias0 = vdupq_n_f32( bias0 );
#else
        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

        int i = 0;

        for ( ; i < outh; i++ ) {
#if NEON
            int nn = outw >> 2;
            int remain = outw & 3;
#else
            int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                    "0:                                        \n"
                    "fmul       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %10.s[2]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                    "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%3]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                    "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%4]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "fadd       v0.4s, v0.4s, v10.4s           \n"
                    "fadd       v0.4s, v0.4s, v11.4s           \n"

                    "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s}, [%1], #16             \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 )    // %4
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "w"( _k012x ), // %10
                    "w"( _k345x ), // %11
                    "w"( _k678x ), // %12
                    "w"( _bias0 ) // %13
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vand       q11, %q13, %q13     \n"

                    "0:                             \n"
                    "vmul.f32   q0, q2, %e10[0]     \n"
                    "vmul.f32   q10, q3, %e10[1]    \n"

                    "pld        [%2, #128]          \n"
                    "vld2.f32   {d16-d17}, [%2]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f10[0]    \n"

                    "pld        [%3, #256]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"

                    "vmla.f32   q0, q2, %e11[0]     \n"
                    "vmla.f32   q10, q3, %e11[1]    \n"

                    "pld        [%3, #128]          \n"
                    "vld2.f32   {d16-d17}, [%3]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f11[0]    \n"

                    "pld        [%4, #256]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"

                    "vmla.f32   q0, q2, %e12[0]     \n"
                    "vmla.f32   q10, q3, %e12[1]    \n"

                    "pld        [%4, #128]          \n"
                    "vld2.f32   {d16-d17}, [%4]     \n"
                    "vext.32    q1, q2, q8, #1      \n"

                    "vmla.f32   q11, q1, %f12[0]    \n"

                    "pld        [%2, #256]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"

                    "vadd.f32   q0, q0, q10         \n"
                    "vadd.f32   q0, q0, q11         \n"

                    "vand       q11, %q13, %q13     \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"( nn ),   // %0
                    "=r"( outptr ), // %1
                    "=r"( r0 ),   // %2
                    "=r"( r1 ),   // %3
                    "=r"( r2 )    // %4
                    : "0"( nn ),
                    "1"( outptr ),
                    "2"( r0 ),
                    "3"( r1 ),
                    "4"( r2 ),
                    "w"( _k012x ), // %10
                    "w"( _k345x ), // %11
                    "w"( _k678x ), // %12
                    "w"( _bias0 ) // %13
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
            }

#endif // __aarch64__
#endif // __ARM_NEON

            for ( ; remain > 0; remain-- ) {
#if NEON
                float32x4_t _r00 = vld1q_f32( r0 );
                float32x4_t _r10 = vld1q_f32( r1 );
                float32x4_t _r20 = vld1q_f32( r2 );

                float32x4_t _sum = vmulq_f32( _r00, _k012x );
                _sum = vmlaq_f32( _sum, _r10, _k345x );
                _sum = vmlaq_f32( _sum, _r20, _k678x );

                _sum = vsetq_lane_f32( bias0, _sum, 3 );
#if __aarch64__
                *outptr = vaddvq_f32( _sum );
#else
                float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                _ss = vpadd_f32( _ss, _ss );

                *outptr = vget_lane_f32( _ss, 0 );
#endif // __aarch64__
#else
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                *outptr = sum;
#endif // __ARM_NEON

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

PUBLIC void conv3x3s1_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias )
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;
    int pp = 0, p = 0, q = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( pp = 0; pp < nn_outch; pp++ ) {
        int p = pp * 2;

        float* out0 = channel( top_blob, p );
        float* out1 = channel( top_blob, p + 1 );

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;

        fill( out0, bias0, top_blob.cstep );
        fill( out1, bias1, top_blob.cstep );

        const float* k0 = kernel + p * inch * 9;
        const float* k1 = kernel + ( p + 1 ) * inch * 9;

        for ( q = 0; q < inch; q++ ) {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr0n = outptr0 + outw;
            float* outptr1n = outptr1 + outw;

            const float* img0 = channel( bottom_blob, q );

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

#if NEON
            float32x4_t _k00 = vld1q_f32( k0 );
            float32x4_t _k03 = vld1q_f32( k0 + 3 );
            float32x4_t _k06 = vld1q_f32( k0 + 6 );

            float32x4_t _k10 = vld1q_f32( k1 );
            float32x4_t _k13 = vld1q_f32( k1 + 3 );
            float32x4_t _k16 = vld1q_f32( k1 + 6 );
#endif // __ARM_NEON

            int i = 0;

            for ( ; i + 1 < outh; i += 2 ) {
#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // _sum0

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // _sum1

                        "fmla   v6.4s, v8.4s, %18.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %21.s[0]      \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v12.4s}, [%3]              \n" // _sum0n

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v13.4s}, [%4]              \n" // _sum1n

                        "fmla   v12.4s, v14.4s, %20.s[0]    \n"
                        "fmla   v13.4s, v14.4s, %23.s[0]    \n"

                        "ext    v8.16b, v8.16b, v9.16b, #8  \n"
                        "ext    v9.16b, v14.16b, v15.16b, #4 \n"

                        "fmla   v6.4s, v10.4s, %18.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %21.s[1]     \n"
                        "fmla   v12.4s, v11.4s, %20.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %23.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%6]      \n" // r1
                        "add    %6, %6, #16                 \n"

                        "fmla   v6.4s, v8.4s, %18.s[2]      \n"
                        "fmla   v7.4s, v8.4s, %21.s[2]      \n"
                        "fmla   v12.4s, v9.4s, %20.s[1]     \n"
                        "fmla   v13.4s, v9.4s, %23.s[1]     \n"

                        "ext    v10.16b, v14.16b, v15.16b, #4 \n"

                        "fmla   v6.4s, v14.4s, %19.s[0]     \n"
                        "fmla   v7.4s, v14.4s, %22.s[0]     \n"
                        "fmla   v12.4s, v14.4s, %18.s[0]    \n"
                        "fmla   v13.4s, v14.4s, %21.s[0]    \n"

                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %19.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %22.s[1]     \n"
                        "fmla   v12.4s, v10.4s, %18.s[1]    \n"
                        "fmla   v13.4s, v10.4s, %21.s[1]    \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%7]        \n" // r2
                        "add    %7, %7, #16                 \n"

                        "fmla   v6.4s, v11.4s, %19.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %22.s[2]     \n"
                        "fmla   v12.4s, v11.4s, %18.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %21.s[2]    \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "fmla   v6.4s, v8.4s, %20.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %23.s[0]      \n"
                        "fmla   v12.4s, v8.4s, %19.s[0]     \n"
                        "fmla   v13.4s, v8.4s, %22.s[0]     \n"

                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %20.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %23.s[1]     \n"
                        "fmla   v12.4s, v10.4s, %19.s[1]    \n"
                        "fmla   v13.4s, v10.4s, %22.s[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %20.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %23.s[2]     \n"
                        "fmla   v12.4s, v11.4s, %19.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %22.s[2]    \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "st1    {v12.4s}, [%3], #16         \n"
                        "st1    {v13.4s}, [%4], #16         \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %5, %5, #16                 \n"
                        "sub    %8, %8, #16                 \n"
                        : "=r"( nn ),     // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( outptr0n ), // %3
                        "=r"( outptr1n ), // %4
                        "=r"( r0 ),     // %5
                        "=r"( r1 ),     // %6
                        "=r"( r2 ),     // %7
                        "=r"( r3 )      // %8
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( outptr0n ),
                        "4"( outptr1n ),
                        "5"( r0 ),
                        "6"( r1 ),
                        "7"( r2 ),
                        "8"( r3 ),
                        "w"( _k00 ), // %18
                        "w"( _k03 ), // %19
                        "w"( _k06 ), // %20
                        "w"( _k10 ), // %21
                        "w"( _k13 ), // %22
                        "w"( _k16 ) // %23
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                        "add        %5, #16             \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n" // r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q14, q15, #2   \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1 :64] \n" // _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2 :64] \n" // _sum1

                        "vmla.f32   q6, q8, %e18[0]     \n"
                        "vmla.f32   q7, q8, %e21[0]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d24-d25}, [%3]     \n" // _sum0n

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d26-d27}, [%4]     \n" // _sum1n

                        "vmla.f32   q12, q14, %e20[0]   \n"
                        "vmla.f32   q13, q14, %e23[0]   \n"

                        "vext.32    q8, q8, q9, #2      \n"
                        "vext.32    q9, q14, q15, #1    \n"

                        "vmla.f32   q6, q10, %e18[1]    \n"
                        "vmla.f32   q7, q10, %e21[1]    \n"
                        "vmla.f32   q12, q11, %f20[0]   \n"
                        "vmla.f32   q13, q11, %f23[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d28-d30}, [%6]     \n" // r1
                        "add        %6, #16             \n"

                        "vmla.f32   q6, q8, %f18[0]     \n"
                        "vmla.f32   q7, q8, %f21[0]     \n"
                        "vmla.f32   q12, q9, %e20[1]    \n"
                        "vmla.f32   q13, q9, %e23[1]    \n"

                        "vext.32    q10, q14, q15, #1   \n"

                        "vmla.f32   q6, q14, %e19[0]    \n"
                        "vmla.f32   q7, q14, %e22[0]    \n"
                        "vmla.f32   q12, q14, %e18[0]   \n"
                        "vmla.f32   q13, q14, %e21[0]   \n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vmla.f32   q6, q10, %e19[1]    \n"
                        "vmla.f32   q7, q10, %e22[1]    \n"
                        "vmla.f32   q12, q10, %e18[1]   \n"
                        "vmla.f32   q13, q10, %e21[1]   \n"

                        "pld        [%7, #192]          \n"
                        "vld1.f32   {d16-d18}, [%7 :64] \n" // r2
                        "add        %7, #16             \n"

                        "vmla.f32   q6, q11, %f19[0]    \n"
                        "vmla.f32   q7, q11, %f22[0]    \n"
                        "vmla.f32   q12, q11, %f18[0]   \n"
                        "vmla.f32   q13, q11, %f21[0]   \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vmla.f32   q6, q8, %e20[0]     \n"
                        "vmla.f32   q7, q8, %e23[0]     \n"
                        "vmla.f32   q12, q8, %e19[0]    \n"
                        "vmla.f32   q13, q8, %e22[0]    \n"

                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e20[1]    \n"
                        "vmla.f32   q7, q10, %e23[1]    \n"
                        "vmla.f32   q12, q10, %e19[1]   \n"
                        "vmla.f32   q13, q10, %e22[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f20[0]    \n"
                        "vmla.f32   q7, q11, %f23[0]    \n"
                        "vmla.f32   q12, q11, %f19[0]   \n"
                        "vmla.f32   q13, q11, %f22[0]   \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n" // r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vst1.f32   {d12-d13}, [%1 : 64]!\n"
                        "vst1.f32   {d14-d15}, [%2 : 64]!\n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vst1.f32   {d24-d25}, [%3]!    \n"
                        "vst1.f32   {d26-d27}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %5, #16             \n"
                        "sub        %8, #16             \n"
                        : "=r"( nn ),     // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( outptr0n ), // %3
                        "=r"( outptr1n ), // %4
                        "=r"( r0 ),     // %5
                        "=r"( r1 ),     // %6
                        "=r"( r2 ),     // %7
                        "=r"( r3 )      // %8
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( outptr0n ),
                        "4"( outptr1n ),
                        "5"( r0 ),
                        "6"( r1 ),
                        "7"( r2 ),
                        "8"( r3 ),
                        "w"( _k00 ), // %18
                        "w"( _k03 ), // %19
                        "w"( _k06 ), // %20
                        "w"( _k10 ), // %21
                        "w"( _k13 ), // %22
                        "w"( _k16 ) // %23
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // __ARM_NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );
                    float32x4_t _r30 = vld1q_f32( r3 );

                    float32x4_t _sum0 = vmulq_f32( _r00, _k00 );
                    float32x4_t _sum1 = vmulq_f32( _r00, _k10 );
                    _sum0 = vmlaq_f32( _sum0, _r10, _k03 );
                    _sum1 = vmlaq_f32( _sum1, _r10, _k13 );
                    _sum0 = vmlaq_f32( _sum0, _r20, _k06 );
                    _sum1 = vmlaq_f32( _sum1, _r20, _k16 );

                    float32x4_t _sum0n = vmulq_f32( _r10, _k00 );
                    float32x4_t _sum1n = vmulq_f32( _r10, _k10 );
                    _sum0n = vmlaq_f32( _sum0n, _r20, _k03 );
                    _sum1n = vmlaq_f32( _sum1n, _r20, _k13 );
                    _sum0n = vmlaq_f32( _sum0n, _r30, _k06 );
                    _sum1n = vmlaq_f32( _sum1n, _r30, _k16 );

                    _sum0 = vsetq_lane_f32( *outptr0, _sum0, 3 );
                    _sum1 = vsetq_lane_f32( *outptr1, _sum1, 3 );
                    _sum0n = vsetq_lane_f32( *outptr0n, _sum0n, 3 );
                    _sum1n = vsetq_lane_f32( *outptr1n, _sum1n, 3 );
#if __aarch64__
                    *outptr0 = vaddvq_f32( _sum0 );
                    *outptr1 = vaddvq_f32( _sum1 );
                    *outptr0n = vaddvq_f32( _sum0n );
                    *outptr1n = vaddvq_f32( _sum1n );
#else
                    float32x2_t _ss0 = vadd_f32( vget_low_f32( _sum0 ), vget_high_f32( _sum0 ) );
                    float32x2_t _ss1 = vadd_f32( vget_low_f32( _sum1 ), vget_high_f32( _sum1 ) );
                    float32x2_t _ss0n = vadd_f32( vget_low_f32( _sum0n ), vget_high_f32( _sum0n ) );
                    float32x2_t _ss1n = vadd_f32( vget_low_f32( _sum1n ), vget_high_f32( _sum1n ) );

                    float32x2_t _ss01 = vpadd_f32( _ss0, _ss1 );
                    float32x2_t _ss01n = vpadd_f32( _ss0n, _ss1n );

                    *outptr0 = vget_lane_f32( _ss01, 0 );
                    *outptr1 = vget_lane_f32( _ss01, 1 );
                    *outptr0n = vget_lane_f32( _ss01n, 0 );
                    *outptr1n = vget_lane_f32( _ss01n, 1 );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum0n = 0.f;
                    float sum1 = 0.f;
                    float sum1n = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    sum0n += r1[0] * k0[0];
                    sum0n += r1[1] * k0[1];
                    sum0n += r1[2] * k0[2];
                    sum0n += r2[0] * k0[3];
                    sum0n += r2[1] * k0[4];
                    sum0n += r2[2] * k0[5];
                    sum0n += r3[0] * k0[6];
                    sum0n += r3[1] * k0[7];
                    sum0n += r3[2] * k0[8];

                    sum1n += r1[0] * k1[0];
                    sum1n += r1[1] * k1[1];
                    sum1n += r1[2] * k1[2];
                    sum1n += r2[0] * k1[3];
                    sum1n += r2[1] * k1[4];
                    sum1n += r2[2] * k1[5];
                    sum1n += r3[0] * k1[6];
                    sum1n += r3[1] * k1[7];
                    sum1n += r3[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr0n += sum0n;
                    *outptr1n += sum1n;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr0n++;
                    outptr1n++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr1 += outw;
                outptr0n += outw;
                outptr1n += outw;
            }

            for ( ; i < outh; i++ ) {
#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "0:                                 \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r0
                        "add    %3, %3, #16                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // _sum0

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // _sum1

                        "fmul   v14.4s, v8.4s, %12.s[0]     \n"
                        "fmul   v15.4s, v8.4s, %15.s[0]     \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %12.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %15.s[1]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %15.s[2]    \n"

                        "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v14.4s, v10.4s, %13.s[1]    \n"
                        "fmla   v15.4s, v10.4s, %16.s[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %13.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %16.s[2]     \n"

                        "fmla   v14.4s, v8.4s, %14.s[0]     \n"
                        "fmla   v15.4s, v8.4s, %17.s[0]     \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %14.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %17.s[1]     \n"

                        "fmla   v14.4s, v11.4s, %14.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %17.s[2]    \n"

                        "fadd   v6.4s, v6.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v15.4s        \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        : "=r"( nn ),    // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 )     // %5
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "w"( _k00 ), // %12
                        "w"( _k03 ), // %13
                        "w"( _k06 ), // %14
                        "w"( _k10 ), // %15
                        "w"( _k13 ), // %16
                        "w"( _k16 ) // %17
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n" // r0
                        "add        %3, #16             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n" // _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n" // _sum1

                        "vmul.f32   q14, q8, %e12[0]    \n"
                        "vmul.f32   q15, q8, %e15[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e12[1]    \n"
                        "vmla.f32   q7, q10, %e15[1]    \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" // r1
                        "add        %4, #16             \n"

                        "vmla.f32   q14, q11, %f12[0]   \n"
                        "vmla.f32   q15, q11, %f15[0]   \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q14, q10, %e13[1]   \n"
                        "vmla.f32   q15, q10, %e16[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5]     \n" // r2
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        "vmla.f32   q14, q8, %e14[0]    \n"
                        "vmla.f32   q15, q8, %e17[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e14[1]    \n"
                        "vmla.f32   q7, q10, %e17[1]    \n"

                        "vmla.f32   q14, q11, %f14[0]   \n"
                        "vmla.f32   q15, q11, %f17[0]   \n"

                        "vadd.f32   q6, q6, q14         \n"
                        "vadd.f32   q7, q7, q15         \n"

                        "vst1.f32   {d12-d13}, [%1]!    \n"

                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        : "=r"( nn ),    // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 )     // %5
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "w"( _k00 ), // %12
                        "w"( _k03 ), // %13
                        "w"( _k06 ), // %14
                        "w"( _k10 ), // %15
                        "w"( _k13 ), // %16
                        "w"( _k16 ) // %17
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // __ARM_NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );

                    float32x4_t _sum0 = vmulq_f32( _r00, _k00 );
                    float32x4_t _sum1 = vmulq_f32( _r00, _k10 );
                    _sum0 = vmlaq_f32( _sum0, _r10, _k03 );
                    _sum1 = vmlaq_f32( _sum1, _r10, _k13 );
                    _sum0 = vmlaq_f32( _sum0, _r20, _k06 );
                    _sum1 = vmlaq_f32( _sum1, _r20, _k16 );

                    _sum0 = vsetq_lane_f32( *outptr0, _sum0, 3 );
                    _sum1 = vsetq_lane_f32( *outptr1, _sum1, 3 );
#if __aarch64__
                    *outptr0 = vaddvq_f32( _sum0 );
                    *outptr1 = vaddvq_f32( _sum1 );
#else
                    float32x2_t _ss0 = vadd_f32( vget_low_f32( _sum0 ), vget_high_f32( _sum0 ) );
                    float32x2_t _ss1 = vadd_f32( vget_low_f32( _sum1 ), vget_high_f32( _sum1 ) );
                    float32x2_t _ss01 = vpadd_f32( _ss0, _ss1 );

                    *outptr0 = vget_lane_f32( _ss01, 0 );
                    *outptr1 = vget_lane_f32( _ss01, 1 );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                    outptr1++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( p = remain_outch_start; p < outch; p++ ) {
        float* out = channel( top_blob, p );

        const float bias0 = bias ? bias[p] : 0.f;

        fill( out, bias0, top_blob.cstep );
        const float* kernel0 = kernel + p * inch * 9;

        for ( q = 0; q < inch; q++ ) {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = channel( bottom_blob, q );

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

#if NEON
            float32x4_t _k0123 = vld1q_f32( kernel0 );
            float32x4_t _k3456 = vld1q_f32( kernel0 + 3 );
            float32x4_t _k6789 = vld1q_f32( kernel0 + 6 );
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

            int i = 0;

            for ( ; i + 1 < outh; i += 2 ) {

#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0
                        "add    %3, %3, #16                 \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v6.4s, v11.4s, %14.s[1]     \n"
                        "fmul   v13.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%4]       \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v9.4s, %15.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %15.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v8.4s}, [%2]               \n" // _sum2

                        "fmla   v8.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v14.4s, v11.4s, %14.s[1]    \n"
                        "fmul   v15.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%5]       \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v7.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %16.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %16.s[2]    \n"

                        "fmla   v8.4s, v9.4s, %15.s[0]      \n"
                        "fmla   v14.4s, v11.4s, %15.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%6]       \n" // r3
                        "add    %6, %6, #16                 \n"

                        "fmla   v8.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v14.4s, v11.4s, %16.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %16.s[2]    \n"

                        "fadd   v7.4s, v7.4s, v6.4s         \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0

                        "fadd   v8.4s, v8.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v8.4s, v8.4s, v15.4s        \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v7.4s}, [%1], #16          \n"
                        "st1    {v8.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %3, %3, #16                 \n"
                        : "=r"( nn ),    // %0
                        "=r"( outptr ), // %1
                        "=r"( outptr2 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 ),    // %5
                        "=r"( r3 )     // %6
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( outptr2 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "6"( r3 ),
                        "w"( _k0123 ), // %14
                        "w"( _k3456 ), // %15
                        "w"( _k6789 ) // %16
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(
                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n" // r0
                        "add        %3, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1 :64] \n" // _sum

                        "vmla.f32   q7, q9, %e14[0]     \n"
                        "vmul.f32   q6, q11, %e14[1]    \n"
                        "vmul.f32   q13, q12, %f14[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n" // r1
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e15[1]    \n"
                        "vmla.f32   q13, q12, %f15[0]   \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d16-d17}, [%2]     \n" // _sum2

                        "vmla.f32   q8, q9, %e14[0]     \n"
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        "vmul.f32   q15, q12, %f14[0]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5 :64] \n" // r2
                        "add        %5, #16             \n"

                        "vmla.f32   q7, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e16[1]    \n"
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        "vmla.f32   q8, q9, %e15[0]     \n"
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n" // r3
                        "add        %6, #16             \n"

                        "vmla.f32   q8, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q14, q11, %e16[1]   \n"
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n" // r0

                        "vadd.f32   q8, q8, q14         \n"
                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q8, q8, q15         \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "add        %3, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"
                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %3, #16             \n"
                        : "=r"( nn ),    // %0
                        "=r"( outptr ), // %1
                        "=r"( outptr2 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 ),    // %5
                        "=r"( r3 )     // %6
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( outptr2 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "6"( r3 ),
                        "w"( _k0123 ), // %14
                        "w"( _k3456 ), // %15
                        "w"( _k6789 ) // %16
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // __ARM_NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );
                    float32x4_t _r30 = vld1q_f32( r3 );

                    float32x4_t _sum = vmulq_f32( _r00, _k0123 );
                    _sum = vmlaq_f32( _sum, _r10, _k3456 );
                    _sum = vmlaq_f32( _sum, _r20, _k6789 );

                    float32x4_t _sum2 = vmulq_f32( _r10, _k0123 );
                    _sum2 = vmlaq_f32( _sum2, _r20, _k3456 );
                    _sum2 = vmlaq_f32( _sum2, _r30, _k6789 );

                    _sum = vsetq_lane_f32( *outptr, _sum, 3 );
                    _sum2 = vsetq_lane_f32( *outptr2, _sum2, 3 );

#if __aarch64__
                    *outptr = vaddvq_f32( _sum );
                    *outptr2 = vaddvq_f32( _sum2 );
#else
                    float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                    float32x2_t _ss2 = vadd_f32( vget_low_f32( _sum2 ), vget_high_f32( _sum2 ) );

                    float32x2_t _sss2 = vpadd_f32( _ss, _ss2 );

                    *outptr = vget_lane_f32( _sss2, 0 );
                    *outptr2 = vget_lane_f32( _sss2, 1 );
#endif // __aarch64__
#else
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for ( ; i < outh; i++ ) {

#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v8.4s, %10.s[0]      \n"
                        "fmul   v13.4s, v10.4s, %10.s[1]    \n"
                        "fmul   v14.4s, v11.4s, %10.s[2]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r1
                        "add    %3, %3, #16                 \n"

                        "fmla   v7.4s, v8.4s, %11.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %11.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %11.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r2
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v8.4s, %12.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %12.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v7.4s, v7.4s, v14.4s        \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "st1    {v7.4s}, [%1], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %2, %2, #16                 \n"
                        : "=r"( nn ),   // %0
                        "=r"( outptr ), // %1
                        "=r"( r0 ),   // %2
                        "=r"( r1 ),   // %3
                        "=r"( r2 )    // %4
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( r0 ),
                        "3"( r1 ),
                        "4"( r2 ),
                        "w"( _k0123 ), // %10
                        "w"( _k3456 ), // %11
                        "w"( _k6789 ) // %12
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(
                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1]     \n" // _sum

                        "vmla.f32   q7, q8, %e10[0]     \n"
                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmul.f32   q14, q11, %f10[0]   \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n" // r1
                        "add        %3, #16             \n"

                        "vmla.f32   q7, q8, %e11[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" // r2
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                        : "=r"( nn ),   // %0
                        "=r"( outptr ), // %1
                        "=r"( r0 ),   // %2
                        "=r"( r1 ),   // %3
                        "=r"( r2 )    // %4
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( r0 ),
                        "3"( r1 ),
                        "4"( r2 ),
                        "w"( _k0123 ), // %10
                        "w"( _k3456 ), // %11
                        "w"( _k6789 ) // %12
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // __ARM_NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );

                    float32x4_t _sum = vmulq_f32( _r00, _k0123 );
                    _sum = vmlaq_f32( _sum, _r10, _k3456 );
                    _sum = vmlaq_f32( _sum, _r20, _k6789 );

                    _sum = vsetq_lane_f32( *outptr, _sum, 3 );

#if __aarch64__
                    *outptr = vaddvq_f32( _sum );
#else
                    float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                    _ss = vpadd_f32( _ss, _ss );

                    *outptr = vget_lane_f32( _ss, 0 );
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif
                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}

// conv3x3s2
PUBLIC void conv3x3s2_neon( const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias )
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;
    int pp = 0, p = 0, q = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( pp = 0; pp < nn_outch; pp++ ) {
        int p = pp * 2;

        // Mat out0 = top_blob.channel(p);
        // Mat out1 = top_blob.channel(p + 1);
        float* out0 = channel( top_blob, p );
        float* out1 = channel( top_blob, p + 1 );

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;

        // out0.fill(bias0);
        // out1.fill(bias1);
        fill( out0, bias0, top_blob.cstep );
        fill( out1, bias1, top_blob.cstep );

        const float* k0 = kernel + p * inch * 9;
        const float* k1 = kernel + ( p + 1 ) * inch * 9;

        for (q = 0; q < inch; q++) {
            float* outptr0 = out0;
            float* outptr1 = out1;
            // const float *img0 = bottom_blob.channel(q);
            const float* img0 = channel( bottom_blob, q );
            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

#if NEON
            float32x4_t _k00 = vld1q_f32( k0 );
            float32x4_t _k03 = vld1q_f32( k0 + 3 );
            float32x4_t _k06 = vld1q_f32( k0 + 6 );

            float32x4_t _k10 = vld1q_f32( k1 );
            float32x4_t _k13 = vld1q_f32( k1 + 3 );
            float32x4_t _k16 = vld1q_f32( k1 + 6 );
#endif // NEON

            int i = 0;


            for ( ; i < outh; i++ ) {
#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%3], #32   \n" // v8 v9 = r0

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" // v6 = _sum0

                        "fmul   v12.4s, v8.4s, %12.s[0]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n" // v7 = _sum1

                        "fmul   v13.4s, v8.4s, %15.s[0]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%3]      \n" // v10

                        "fmla   v6.4s, v9.4s, %12.s[1]      \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v7.4s, v9.4s, %15.s[1]      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%4], #32   \n" // r1

                        "fmla   v12.4s, v14.4s, %12.s[2]    \n"
                        "fmla   v13.4s, v14.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%4]      \n"

                        "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v12.4s, v9.4s, %13.s[1]     \n"
                        "fmla   v13.4s, v9.4s, %16.s[1]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%5], #32   \n" // r2

                        "fmla   v6.4s, v14.4s, %13.s[2]     \n"
                        "fmla   v7.4s, v14.4s, %16.s[2]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld2    {v10.4s, v11.4s}, [%5]      \n"

                        "fmla   v12.4s, v8.4s, %14.s[0]     \n"
                        "fmla   v13.4s, v8.4s, %17.s[0]     \n"

                        "ext    v14.16b, v8.16b, v10.16b, #4\n"

                        "fmla   v6.4s, v9.4s, %14.s[1]      \n"
                        "fmla   v7.4s, v9.4s, %17.s[1]      \n"

                        "fmla   v12.4s, v14.4s, %14.s[2]    \n"
                        "fmla   v13.4s, v14.4s, %17.s[2]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld2    {v8.4s, v9.4s}, [%3], #32   \n" // v8 v9 = r0

                        "fadd   v6.4s, v6.4s, v12.4s        \n"
                        "fadd   v7.4s, v7.4s, v13.4s        \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "bne    0b                          \n"
                        "sub    %3, %3, #32                 \n"

                        : "=r"( nn ),    // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 )     // %5
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "w"( _k00 ), // %12
                        "w"( _k03 ), // %13
                        "w"( _k06 ), // %14
                        "w"( _k10 ), // %15
                        "w"( _k13 ), // %16
                        "w"( _k16 ) // %17
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(
                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d16-d19}, [%3]!    \n" // q8 q9 = r0

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n" // q6 = _sum0

                        "vmul.f32   q12, q8, %e12[0]    \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n" // q7 = _sum1

                        "vmul.f32   q13, q8, %e15[0]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d20-d21}, [%3]     \n" // q10

                        "vmla.f32   q6, q9, %e12[1]     \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q7, q9, %e15[1]     \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d16-d19}, [%4]!    \n" // r1

                        "vmla.f32   q12, q11, %f12[0]   \n"
                        "vmla.f32   q13, q11, %f15[0]   \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d20-d21}, [%4]     \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q12, q9, %e13[1]    \n"
                        "vmla.f32   q13, q9, %e16[1]    \n"

                        "pld        [%5, #256]          \n"
                        "vld2.f32   {d16-d19}, [%5]!    \n" // r2

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        "pld        [%5, #128]          \n"
                        "vld2.f32   {d20-d21}, [%5]     \n"

                        "vmla.f32   q12, q8, %e14[0]    \n"
                        "vmla.f32   q13, q8, %e17[0]    \n"

                        "vext.32    q11, q8, q10, #1    \n"

                        "vmla.f32   q6, q9, %e14[1]     \n"
                        "vmla.f32   q7, q9, %e17[1]     \n"

                        "vmla.f32   q12, q11, %f14[0]   \n"
                        "vmla.f32   q13, q11, %f17[0]   \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d16-d19}, [%3]!    \n" // q8 q9 = r0

                        "vadd.f32   q6, q6, q12         \n"
                        "vadd.f32   q7, q7, q13         \n"

                        "subs       %0, #1              \n"

                        "vst1.f32   {d12-d13}, [%1]!    \n"
                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "bne        0b                  \n"
                        "sub        %3, #32             \n"

                        : "=r"( nn ),    // %0
                        "=r"( outptr0 ), // %1
                        "=r"( outptr1 ), // %2
                        "=r"( r0 ),    // %3
                        "=r"( r1 ),    // %4
                        "=r"( r2 )     // %5
                        : "0"( nn ),
                        "1"( outptr0 ),
                        "2"( outptr1 ),
                        "3"( r0 ),
                        "4"( r1 ),
                        "5"( r2 ),
                        "w"( _k00 ), // %12
                        "w"( _k03 ), // %13
                        "w"( _k06 ), // %14
                        "w"( _k10 ), // %15
                        "w"( _k13 ), // %16
                        "w"( _k16 ) // %17
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );

                    float32x4_t _sum0 = vmulq_f32( _r00, _k00 );
                    float32x4_t _sum1 = vmulq_f32( _r00, _k10 );
                    _sum0 = vmlaq_f32( _sum0, _r10, _k03 );
                    _sum1 = vmlaq_f32( _sum1, _r10, _k13 );
                    _sum0 = vmlaq_f32( _sum0, _r20, _k06 );
                    _sum1 = vmlaq_f32( _sum1, _r20, _k16 );

                    _sum0 = vsetq_lane_f32( *outptr0, _sum0, 3 );
                    _sum1 = vsetq_lane_f32( *outptr1, _sum1, 3 );
#if __aarch64__
                    *outptr0 = vaddvq_f32( _sum0 );
                    *outptr1 = vaddvq_f32( _sum1 );
#else
                    float32x2_t _ss0 = vadd_f32( vget_low_f32( _sum0 ), vget_high_f32( _sum0 ) );
                    float32x2_t _ss1 = vadd_f32( vget_low_f32( _sum1 ), vget_high_f32( _sum1 ) );
                    float32x2_t _ss01 = vpadd_f32( _ss0, _ss1 );

                    *outptr0 = vget_lane_f32( _ss01, 0 );
                    *outptr1 = vget_lane_f32( _ss01, 1 );
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                    outptr1++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    // #pragma omp parallel for num_threads(opt.num_threads)


    for ( p = remain_outch_start; p < outch; p++ ) {
        // Mat out = top_blob.channel(p);
        float* out = channel( top_blob, p );


        const float bias0 = bias ? bias[p] : 0.f;

        // out.fill(bias0);
        fill( out, bias0, top_blob.cstep );

        const float* kernel0 = kernel + p * inch * 9;



        for ( q = 0; q < inch; q++ ) {
            float* outptr = out;

            // const float *img0 = bottom_blob.channel(q);
            const float* img0 = channel( bottom_blob, q );

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if NEON
            float32x4_t _k0123 = vld1q_f32( k0 );
            float32x4_t _k3456 = vld1q_f32( k1 );
            float32x4_t _k6789 = vld1q_f32( k2 );
#endif // NEON

            int i = 0;

            for ( ; i < outh; i++ ) {
#if NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__

                if ( nn > 0 ) {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "0:                                        \n"

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v0.4s}, [%1]                  \n"

                        "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                        "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%2]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                        "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%3]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                        "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%4]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                        "fadd       v0.4s, v0.4s, v10.4s           \n"
                        "fadd       v0.4s, v0.4s, v11.4s           \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"( nn ),   // %0
                        "=r"( outptr ), // %1
                        "=r"( r0 ),   // %2
                        "=r"( r1 ),   // %3
                        "=r"( r2 )    // %4
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( r0 ),
                        "3"( r1 ),
                        "4"( r2 ),
                        "w"( _k0123 ), // %10
                        "w"( _k3456 ), // %11
                        "w"( _k6789 ) // %12
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15" );
                }

#else

                if ( nn > 0 ) {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]       \n"

                        "vmla.f32   q0, q2, %e10[0]     \n"
                        "vmul.f32   q10, q3, %e10[1]    \n"

                        "pld        [%2, #128]          \n"
                        "vld2.f32   {d16-d17}, [%2]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmul.f32   q11, q1, %f10[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"

                        "vmla.f32   q0, q2, %e11[0]     \n"
                        "vmla.f32   q10, q3, %e11[1]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d16-d17}, [%3]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f11[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"

                        "vmla.f32   q0, q2, %e12[0]     \n"
                        "vmla.f32   q10, q3, %e12[1]    \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d16-d17}, [%4]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f12[0]    \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vadd.f32   q0, q0, q10         \n"
                        "vadd.f32   q0, q0, q11         \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"( nn ),   // %0
                        "=r"( outptr ), // %1
                        "=r"( r0 ),   // %2
                        "=r"( r1 ),   // %3
                        "=r"( r2 )    // %4
                        : "0"( nn ),
                        "1"( outptr ),
                        "2"( r0 ),
                        "3"( r1 ),
                        "4"( r2 ),
                        "w"( _k0123 ), // %10
                        "w"( _k3456 ), // %11
                        "w"( _k6789 ) // %12
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" );
                }

#endif // __aarch64__
#endif // NEON

                for ( ; remain > 0; remain-- ) {
#if NEON
                    float32x4_t _r00 = vld1q_f32( r0 );
                    float32x4_t _r10 = vld1q_f32( r1 );
                    float32x4_t _r20 = vld1q_f32( r2 );

                    float32x4_t _sum = vmulq_f32( _r00, _k0123 );
                    _sum = vmlaq_f32( _sum, _r10, _k3456 );
                    _sum = vmlaq_f32( _sum, _r20, _k6789 );

                    _sum = vsetq_lane_f32( *outptr, _sum, 3 );

#if __aarch64__
                    *outptr = vaddvq_f32( _sum );
#else
                    float32x2_t _ss = vadd_f32( vget_low_f32( _sum ), vget_high_f32( _sum ) );
                    _ss = vpadd_f32( _ss, _ss );

                    *outptr = vget_lane_f32( _ss, 0 );
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif // NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }

}

// relu
PUBLIC void relu_neon( Mat mat )
{
    float* ptr = mat.data;
    int size = total( mat );
#if NEON
    int nn = size >> 2;
    int remain = size - ( nn << 2 );
#else
    int remain = size;
#endif // __ARM_NEON

#if NEON
#if __aarch64__
    float32x4_t _zero = vdupq_n_f32( 0.f );

    for ( ; nn > 0; nn-- ) {
        float32x4_t _p = vld1q_f32( ptr );
        _p = vmaxq_f32( _p, _zero );
        vst1q_f32( ptr, _p );

        ptr += 4;
    }

#else

    if ( nn > 0 ) {
        asm volatile(
            "veor       q1, q0, q0          \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1 :128]  \n"
            "vmax.f32   q0, q0, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d1}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"( nn ), // %0
            "=r"( ptr ) // %1
            : "0"( nn ),
            "1"( ptr )
            : "cc", "memory", "q0", "q1" );
    }

#endif // __aarch64__
#endif // __ARM_NEON

    for ( ; remain > 0; remain-- ) {
        *ptr = MAX( *ptr, 0.f );
        ptr++;
    }
}

// hswish
PUBLIC void hswish_neon( Mat bottom_top_blob )
{
    const float upper = 3.0f, lower = -3.0f, alpha = 1.0f / 6, beta = 0.5f;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int q = 0;

    for ( q = 0; q < channels; q++ ) {
        // float* ptr = bottom_top_blob.channel(q);
        float* ptr = channel( bottom_top_blob, q );

#if NEON
        int nn = size >> 2;
        int remain = size - ( nn << 2 );
#else
        int remain = size;
#endif // __ARM_NEON

#if NEON
        float32x4_t _zero = vdupq_n_f32( 0.f );
        float32x4_t _one = vdupq_n_f32( 1.f );

        while ( nn-- ) {
            float32x4_t _p = vld1q_f32( ptr );
            float32x4_t _ans = vdupq_n_f32( beta );
            _ans = vmlaq_n_f32( _ans, _p, alpha );
            _ans = vmaxq_f32( _ans, _zero );
            _ans = vminq_f32( _ans, _one );
            _ans = vmulq_f32( _ans, _p );
            vst1q_f32( ptr, _ans );

            ptr += 4;
        }

#endif // __ARM_NEON

        for ( ; remain > 0; remain-- ) {
            if ( *ptr < lower ) {
                *ptr = 0.f;
            } else if ( *ptr > upper )
                ;
            else {
                *ptr = *ptr * ( *ptr * alpha + beta );
            }

            ++ptr;
        }
    }
}

// hsigmoid
PUBLIC void hsigmoid_neon( Mat bottom_top_blob )
{
    const float upper = 3.0f, lower =  -3.0f, alpha = 1.0f / 6, beta = 0.5f;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int q = 0;

    for ( q = 0; q < channels; q++ ) {
        // float* ptr = bottom_top_blob.channel(q);
        float* ptr = channel( bottom_top_blob, q );

#if NEON
        int nn = size >> 2;
        int remain = size - ( nn << 2 );
#else
        int remain = size;
#endif // __ARM_NEON

#if NEON
        float32x4_t _zero = vdupq_n_f32( 0.f );
        float32x4_t _one = vdupq_n_f32( 1.f );

        while ( nn-- ) {
            float32x4_t _p = vld1q_f32( ptr );
            float32x4_t _ans = vdupq_n_f32( beta );
            _ans = vmlaq_n_f32( _ans, _p, alpha );
            _ans = vmaxq_f32( _ans, _zero );
            _ans = vminq_f32( _ans, _one );
            vst1q_f32( ptr, _ans );

            ptr += 4;
        }

#endif // __ARM_NEON

        for ( ; remain > 0; remain-- ) {
            if ( *ptr < lower ) {
                *ptr = 0.f;
            } else if ( *ptr > upper ) {
                *ptr = 1.f;
            } else {
                *ptr = *ptr * alpha + beta;
            }

            ++ptr;
        }
    }
}
// tanh

PUBLIC void tanh_neon(Mat mat)
{
    float* ptr = mat.data;
    int size = total(mat);

#if NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // NEON

#if NEON

    for (; nn > 0; nn--) {
        float32x4_t _p = vld1q_f32(ptr);
        _p = tanh_ps(_p);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }

#endif // NEON

    for (; remain > 0; remain--) {
        *ptr = SL_Tanh(*ptr);
        ptr++;
    }
}
// global pooling
PUBLIC void pooling_global( const Mat bottom_blob, Mat top_blob, int pooling_type )
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    // size_t elemsize = bottom_blob.elemsize;
    // top_blob.create(channels, elemsize, opt.blob_allocator);
    int q = 0, i = 0;

    for ( q = 0; q < channels; q++ ) {
        float* out = channel( top_blob, q );
        fill( out, 0, top_blob.cstep );
    }

    int size = w * h;

    if ( pooling_type == PoolMethod_MAX ) {
        for ( q = 0; q < channels; q++ ) {
            // const float *ptr = bottom_blob.channel(q);
            const float* ptr = channel( bottom_blob, q );

            float maxv = ptr[0];

            for ( i = 0; i < size; i++ ) {
                maxv = MAX( maxv, ptr[i] );
            }

            top_blob.data[q * top_blob.cstep] = maxv;
        }
    } else if ( pooling_type == PoolMethod_AVE ) {

        for ( q = 0; q < channels; q++ ) {
            // const float *ptr = bottom_blob.channel(q);
            const float* ptr = channel( bottom_blob, q );

            float sum = 0.f;

            for ( i = 0; i < size; i++ ) {
                sum += ptr[i];
            }

            top_blob.data[q * top_blob.cstep] = sum / size;
        }
    }
}

PUBLIC void pooling2x2s2_max_neon( const Mat bottom_blob, Mat top_blob )
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;
    int q = 0, i = 0;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for ( q = 0; q < inch; q++ ) {
        const float* img0 = channel( bottom_blob, q );
        float* outptr = channel( top_blob, q );

        const float* r0 = img0;
        const float* r1 = img0 + w;

        for ( i = 0; i < outh; i++ ) {
#if NEON
            int nn = outw >> 2;
            int remain = outw - ( nn << 2 );
#else
            int remain = outw;
#endif // __ARM_NEON

#if NEON
#if __aarch64__

            if ( nn > 0 ) {
                asm volatile(
                    "0:                                   \n"
                    "prfm       pldl1keep, [%1, #256]     \n"
                    "prfm       pldl1keep, [%2, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                    "fmax       v0.4s, v0.4s, v2.4s       \n"
                    "fmax       v1.4s, v1.4s, v3.4s       \n"
                    "fmaxp      v2.4s, v0.4s, v1.4s       \n"
                    "subs       %w0, %w0, #1              \n"
                    "st1        {v2.4s}, [%3], #16        \n"
                    "bne        0b                        \n"
                    : "=r"( nn ),  // %0
                    "=r"( r0 ),  // %1
                    "=r"( r1 ),  // %2
                    "=r"( outptr ) // %3
                    : "0"( nn ),
                    "1"( r0 ),
                    "2"( r1 ),
                    "3"( outptr )
                    : "cc", "memory", "v0", "v1", "v2", "v3" );
            }

#else

            if ( nn > 0 ) {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"
                    "vld1.f32   {d4-d7}, [%2]!      \n"
                    "vmax.f32   q0, q0, q2          \n"
                    "vmax.f32   q1, q1, q3          \n"
                    "vpmax.f32  d4, d0, d1          \n"
                    "vpmax.f32  d5, d2, d3          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d4-d5}, [%3]!      \n"
                    "bne        0b                  \n"
                    : "=r"( nn ),  // %0
                    "=r"( r0 ),  // %1
                    "=r"( r1 ),  // %2
                    "=r"( outptr ) // %3
                    : "0"( nn ),
                    "1"( r0 ),
                    "2"( r1 ),
                    "3"( outptr )
                    : "cc", "memory", "q0", "q1", "q2", "q3" );
            }

#endif // __aarch64__
#endif // __ARM_NEON

            for ( ; remain > 0; remain-- ) {
                float max0 = MAX( r0[0], r0[1] );
                float max1 = MAX( r1[0], r1[1] );

                *outptr = MAX( max0, max1 );

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}

PUBLIC void groupconv3x3s1_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias, const int group)
{
    int out_c = top_blob.c / group;
    int inp_c = bottom_blob.c / group;
    int i = 0;
    Mat input_group = bottom_blob;
    input_group.c = inp_c;
    int jump_inp_mem = input_group.cstep * inp_c;
    Mat output_group = top_blob;
    output_group.c = out_c;
    int jump_out_mem = output_group.cstep * out_c;
    const float* weight = _kernel;
    const float* bias = _bias;
    int jump_weight = inp_c * out_c * 3 * 3;
    int jump_bias = out_c;

    for (; i < group; i++) {
        conv3x3s1_neon(input_group, output_group, weight, bias);
        input_group.data += jump_inp_mem;
        output_group.data += jump_out_mem;
        weight += jump_weight;
        bias += jump_bias;
    }
}

PUBLIC void groupconv3x3s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias, const int group)
{
    int out_c = top_blob.c / group;
    int inp_c = bottom_blob.c / group;
    int i = 0;
    Mat input_group = bottom_blob;
    input_group.c = inp_c;
    int jump_inp_mem = input_group.cstep * inp_c;
    Mat output_group = top_blob;
    output_group.c = out_c;
    int jump_out_mem = output_group.cstep * out_c;
    const float* weight = _kernel;
    const float* bias = _bias;
    int jump_weight = inp_c * out_c * 3 * 3;
    int jump_bias = out_c;

    for (; i < group; i++) {
        conv3x3s2_neon(input_group, output_group, weight, bias);
        input_group.data += jump_inp_mem;
        output_group.data += jump_out_mem;
        weight += jump_weight;
        bias += jump_bias;
    }
}

// deconv4x4s2 stride=2 ->deconv4x4_block()
PUBLIC void deconv4x4s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    // #pragma omp parallel for num_threads(opt.num_threads)
    int p, q;

    for (p = 0; p < outch; p++) {
        float* out = channel(top_blob, p); // top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        // out.fill(bias0);
        fill(out, bias0, top_blob.cstep);

        for (q = 0; q < inch; q++) {
            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 16 + q * 16;

            const float* r0 = img0;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;

#if NEON
            float32x4_t _k0 = vld1q_f32(k0);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _k2 = vld1q_f32(k2);
            float32x4_t _k3 = vld1q_f32(k3);
#endif // NEON
            int i;

            for (i = 0; i < h; i++) {
                // float* outptr = out.row(i * 2);
                float* outptr = out + outw * i * 2;
                float* outptr0 = outptr;
                float* outptr1 = outptr0 + outw;
                float* outptr2 = outptr1 + outw;
                float* outptr3 = outptr2 + outw;

                int j = 0;
#if NEON

                for (; j + 3 < w; j += 4) {
                    float32x4_t _v = vld1q_f32(r0);

                    // row 0
                    float32x4x2_t _out0 = vld2q_f32(outptr0);
                    // 0,2,4,6
                    _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_low_f32(_k0), 0);
                    // 1,3,5,7
                    _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_low_f32(_k0), 1);
                    vst2q_f32(outptr0, _out0);

                    _out0 = vld2q_f32(outptr0 + 2);
                    // 2,4,6,8
                    _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_high_f32(_k0), 0);
                    // 3,5,7,9
                    _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_high_f32(_k0), 1);
                    vst2q_f32(outptr0 + 2, _out0);

                    // row 1
                    float32x4x2_t _out1 = vld2q_f32(outptr1);
                    // 0,2,4,6
                    _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_low_f32(_k1), 0);
                    // 1,3,5,7
                    _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_low_f32(_k1), 1);
                    vst2q_f32(outptr1, _out1);

                    _out1 = vld2q_f32(outptr1 + 2);
                    // 2,4,6,8
                    _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_high_f32(_k1), 0);
                    // 3,5,7,9
                    _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_high_f32(_k1), 1);
                    vst2q_f32(outptr1 + 2, _out1);

                    // row 2
                    float32x4x2_t _out2 = vld2q_f32(outptr2);
                    _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_low_f32(_k2), 0);
                    _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_low_f32(_k2), 1);
                    vst2q_f32(outptr2, _out2);

                    _out2 = vld2q_f32(outptr2 + 2);
                    _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_high_f32(_k2), 0);
                    _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_high_f32(_k2), 1);
                    vst2q_f32(outptr2 + 2, _out2);

                    // row 3
                    float32x4x2_t _out3 = vld2q_f32(outptr3);
                    _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_low_f32(_k3), 0);
                    _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_low_f32(_k3), 1);
                    vst2q_f32(outptr3, _out3);

                    _out3 = vld2q_f32(outptr3 + 2);
                    _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_high_f32(_k3), 0);
                    _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_high_f32(_k3), 1);
                    vst2q_f32(outptr3 + 2, _out3);

                    r0 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }

#endif // NEON

                for (; j < w; j++) {
                    float val = r0[0];

                    outptr0[0] += val * k0[0];
                    outptr0[1] += val * k0[1];
                    outptr0[2] += val * k0[2];
                    outptr0[3] += val * k0[3];

                    outptr1[0] += val * k1[0];
                    outptr1[1] += val * k1[1];
                    outptr1[2] += val * k1[2];
                    outptr1[3] += val * k1[3];

                    outptr2[0] += val * k2[0];
                    outptr2[1] += val * k2[1];
                    outptr2[2] += val * k2[2];
                    outptr2[3] += val * k2[3];

                    outptr3[0] += val * k3[0];
                    outptr3[1] += val * k3[1];
                    outptr3[2] += val * k3[2];
                    outptr3[3] += val * k3[3];

                    r0++;
                    outptr0 += 2;
                    outptr1 += 2;
                    outptr2 += 2;
                    outptr3 += 2;
                }
            }
        }
    }
}

// padding_copy ->padding()
PUBLIC void copy_make_border_image( const Mat src, Mat dst, int top, int left, int type, float v )
{
    int w = dst.w;
    int h = dst.h;

    const float* ptr = src.data;
    float* outptr = dst.data;

    if ( type == 0 ) {
        int y = 0;

        // fill top
        for ( ; y < top; y++ ) {
            int x = 0;

            for ( ; x < w; x++ ) {
                outptr[x] = v;
            }

            outptr += w;
        }

        // fill center
        for ( ; y < ( top + src.h ); y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = v;
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = v;
            }

            ptr += src.w;
            outptr += w;
        }

        // fill bottom
        for ( ; y < h; y++ ) {
            int x = 0;

            for ( ; x < w; x++ ) {
                outptr[x] = v;
            }

            outptr += w;
        }
    }

    if ( type == 1 ) {
        int y = 0;

        // fill top
        for ( ; y < top; y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[0];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - 1];
            }

            outptr += w;
        }

        // fill center
        for ( ; y < ( top + src.h ); y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[0];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - 1];
            }

            ptr += src.w;
            outptr += w;
        }

        // fill bottom
        ptr -= src.w;

        for ( ; y < h; y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[0];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - 1];
            }

            outptr += w;
        }
    }

    if ( type == 2 ) {
        int y = 0;
        // fill top
        ptr += top * src.w;

        for ( ; y < top; y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[left - x];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - ( x - left - src.w ) - 2];
            }

            outptr += w;
            ptr -= src.w;
        }

        // fill center
        for ( ; y < ( top + src.h ); y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[left - x];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - ( x - left - src.w ) - 2];
            }

            ptr += src.w;
            outptr += w;
        }

        // fill bottom
        ptr -= 2 * src.w;

        for ( ; y < h; y++ ) {
            int x = 0;

            for ( ; x < left; x++ ) {
                outptr[x] = ptr[left - x];
            }

            if ( src.w < 12 ) {
                for ( ; x < ( left + src.w ); x++ ) {
                    outptr[x] = ptr[x - left];
                }
            } else {
                memcpy( outptr + left, ptr, src.w * sizeof( float ) );
                x += src.w;
            }

            for ( ; x < w; x++ ) {
                outptr[x] = ptr[src.w - ( x - left - src.w ) - 2];
            }

            outptr += w;
            ptr -= src.w;
        }
    }

    return;
}

PUBLIC void padding( const Mat bottom_blob, Mat top_blob, int top, int left, int type, float v )
{
    // int w = bottom_blob.w;
    //    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    // int bottom = top;
    //    int outh = h + top + bottom;
    // top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    int q = 0;

    for ( ; q < channels; q++ ) {
        // const Mat m = bottom_blob.channel(q);
        // Mat borderm = top_blob.channel(q);
        const Mat m = newMat( bottom_blob.data + q * bottom_blob.cstep, bottom_blob.w, bottom_blob.h, 1 );
        const Mat borderm = newMat( top_blob.data + q * top_blob.cstep, top_blob.w, top_blob.h, 1 );

        float pad_value = v;

        copy_make_border_image( m, borderm, top, left, type, pad_value );
    }
}

PUBLIC void padding_normal(Mat ori_blob, Mat pad_blob, const int pad)
{
    int i = 0, j = 0;
    int w = ori_blob.w;
    int inch = ori_blob.c;

    int outw = pad_blob.w;
    int outh = pad_blob.h;

    float* img;
    float* img_out;

    for (; i < inch; i++) {


        img = channel(ori_blob, i);
        img_out = channel(pad_blob, i);

        for (j = pad; j < outh - pad; j++) {
            memcpy(img_out + j * outw + pad, img + (j - pad) * w, w * sizeof(float));
        }
    }
}

PUBLIC void soft_max( const float* src, int channel, float* dst )
{
    int i;
    float exp_sum = 0;
    float mx = *src;

    for ( i = 1; i < channel; i++ ) {
        if ( *( src + i ) > mx ) {
            mx = *( src + i );
        }
    }

    for ( i = 0; i < channel; i++ ) {
        *( dst + i ) = *( src + i ) - mx;
        //*(dst + i) = exp_appro(*(dst + i));
        *(dst + i) = (float)SL_exp(*(dst + i));
    }

    for ( i = 0; i < channel; i++ ) {
        exp_sum += ( *( dst + i ) );
    }

    for ( i = 0; i < channel; i++ ) {
        *( dst + i ) /= ( exp_sum );
    }
}

// matrix_add
PUBLIC void mat_add_neon_inplace( Mat bottom_top_blob, Mat add_blob )
{
    int i = 0;
    int size = bottom_top_blob.cstep * bottom_top_blob.c;
#if NEON

    float* pa = bottom_top_blob.data;
    float* pb = add_blob.data;
    float32x4_t add;

    for ( i = 0; i < size - 4; i += 4 ) {
        add = vaddq_f32( vld1q_f32( pa ), vld1q_f32( pb ) );
        vst1q_f32( pa, add );
        pa += 4;
        pb += 4;
    }

#endif

    for ( ; i < size; i++ ) {
        bottom_top_blob.data[i] += add_blob.data[i];
    }
}

// matrix_multiply ->bilinear_neon_cnn()
PUBLIC void mat_scale_neon_inplace( Mat bottom_top_blob, Mat scale_blob )
{

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int q = 0;
    const float* scale_ptr = scale_blob.data;

    for ( q = 0; q < channels; q++ ) {
        // float *ptr = bottom_top_blob.channel(q);
        float* ptr = channel( bottom_top_blob, q );

        float s = scale_ptr[q * scale_blob.cstep];

#if NEON
        int nn = size >> 2;
        int remain = size - ( nn << 2 );
#else
        int remain = size;
#endif // __ARM_NEON

#if NEON
        float32x4_t _s = vdupq_n_f32( s );

        for ( ; nn > 0; nn-- ) {
            float32x4_t _p = vld1q_f32( ptr );
            _p = vmulq_f32( _p, _s );
            vst1q_f32( ptr, _p );

            ptr += 4;
        }

#endif // __ARM_NEON

        for ( ; remain > 0; remain-- ) {
            *ptr *= s;

            ptr++;
        }
    }
}

PUBLIC void linear_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = (double)w / outw;

    int dx;

    for (dx = 0; dx < outw; dx++) {
        float fx = (float)((dx + 0.5f) * scale - 0.5f);

        int sx = SL_Floor(fx);
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }

        if (sx >= w - 1) {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

// flag 1:int img
PUBLIC void resize_bilinear_image(const Mat src, Mat dst, float* alpha, int* xofs, float* beta, int* yofs, int flag)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    /*Mat rowsbuf0(w);
    Mat rowsbuf1(w);*/
    float* rowsbuf0 = (float*)malloc(w * sizeof(float));
    float* rowsbuf1 = (float*)malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    int dy;

    for (dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 1) {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            /*const float* S1 = src.row(sy + 1);*/
            const float* S1 = src.data + (sy + 1) * src.w;

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
#if NEON

            for (; dx + 1 < w; dx += 2) {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }

#endif // NEON

            for (; dx < w; dx++) {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        } else {
            // hresize two rows
            /*const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);*/
            const float* S0 = src.data + sy * src.w;
            const float* S1 = src.data + (sy + 1) * src.w;

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
#if NEON

            for (; dx + 1 < w; dx += 2) {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }

#endif // NEON

            for (; dx < w; dx++) {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        /*float* Dp = dst.row(dy);*/
        float* Dp = dst.data + dy * w;

#if NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if NEON
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);

        for (; nn > 0; nn--) {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp + 4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

#endif // NEON

        if (flag == 1) {
            for (; remain; --remain) {
                //             D[x] = rows0[x]*b0 + rows1[x]*b1;
                *Dp++ = (float)(int)(*rows0p++ * b0 + *rows1p++ * b1);
            }
        } else {
            for (; remain; --remain) {
                //             D[x] = rows0[x]*b0 + rows1[x]*b1;
                *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
            }
        }

        beta += 2;
    }

    free(rowsbuf0);
    free(rowsbuf1);
}

PUBLIC int bilinear_neon_cnn(const Mat bottom_blob, Mat top_blob, int flag)
{
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    if (outh == 0 || outw == 0) {
        return -1;
    }

    if (outh == h && outw == w) {
        top_blob = bottom_blob;
        return 0;
    }

    /*int* buf = new int[outw + outh + outw * 2 + outh * 2];*/
    int* buf = (int*)malloc((outw + outh + outw * 2 + outh * 2) * sizeof(int));

    int* xofs = buf;        // new int[outw];
    int* yofs = buf + outw; // new int[outh];

    float* alpha = (float*)(buf + outw + outh);        // new float[outw * 2];
    float* beta = (float*)(buf + outw + outh + outw * 2); // new float[outh * 2];

    linear_coeffs(w, outw, xofs, alpha);
    linear_coeffs(h, outh, yofs, beta);

    // #pragma omp parallel for num_threads(opt.num_threads)
    int q;

    for (q = 0; q < channels; q++) {
        const Mat src = newMat(channel(bottom_blob, q), w, h, channels);
        Mat dst = newMat(channel(top_blob, q), outw, outh, outch);
        resize_bilinear_image(src, dst, alpha, xofs, beta, yofs, flag);
    }

    free(buf);

    return 0;
}

// deconv3x3s2
PUBLIC void deconv3x3s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    // #pragma omp parallel for num_threads(opt.num_threads)
    int p;

    for (p = 0; p < outch; p++) {
        float* out = channel(top_blob, p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, bias0, top_blob.cstep);
        int q;

        for (q = 0; q < inch; q++) {
            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 9 + q * 9;

            const float* r0 = img0;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

#if NEON
            float32x4_t _k0 = vld1q_f32(k0);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _k2 = vld1q_f32(k2);
#endif // NEON
            int i;

            for (i = 0; i < h; i++) {
                // float* outptr = out.row(i * 2);
                float* outptr = out + i * 2 * outw;

                float* outptr0 = outptr;
                float* outptr1 = outptr0 + outw;
                float* outptr2 = outptr1 + outw;

                int j = 0;
#if NEON

                for (; j + 3 < w; j += 4) {
                    float32x4_t _v = vld1q_f32(r0);

                    // out row 0
                    float32x4_t _out00 = vmulq_lane_f32(_v, vget_low_f32(_k0), 0); // 0,2,4,6
                    float32x4_t _out01 = vmulq_lane_f32(_v, vget_low_f32(_k0), 1); // 1,3,5,7
                    float32x4_t _out02 = vmulq_lane_f32(_v, vget_high_f32(_k0), 0); // 2,4,6,8

                    float32x4x2_t _out0 = vld2q_f32(outptr0);
                    _out0.val[0] = vaddq_f32(_out0.val[0], _out00); // 0,2,4,6
                    _out0.val[1] = vaddq_f32(_out0.val[1], _out01); // 1,3,5,7
                    vst2q_f32(outptr0, _out0);

                    _out0 = vld2q_f32(outptr0 + 2);
                    _out0.val[0] = vaddq_f32(_out0.val[0], _out02); // 2,4,6,8
                    vst2q_f32(outptr0 + 2, _out0);

                    // out row 1
                    float32x4_t _out10 = vmulq_lane_f32(_v, vget_low_f32(_k1), 0); // 0,2,4,6
                    float32x4_t _out11 = vmulq_lane_f32(_v, vget_low_f32(_k1), 1); // 1,3,5,7
                    float32x4_t _out12 = vmulq_lane_f32(_v, vget_high_f32(_k1), 0); // 2,4,6,8

                    float32x4x2_t _out1 = vld2q_f32(outptr1);
                    _out1.val[0] = vaddq_f32(_out1.val[0], _out10); // 0,2,4,6
                    _out1.val[1] = vaddq_f32(_out1.val[1], _out11); // 1,3,5,7
                    vst2q_f32(outptr1, _out1);

                    _out1 = vld2q_f32(outptr1 + 2);
                    _out1.val[0] = vaddq_f32(_out1.val[0], _out12); // 2,4,6,8
                    vst2q_f32(outptr1 + 2, _out1);

                    // out row 2
                    float32x4_t _out20 = vmulq_lane_f32(_v, vget_low_f32(_k2), 0); // 0,2,4,6
                    float32x4_t _out21 = vmulq_lane_f32(_v, vget_low_f32(_k2), 1); // 1,3,5,7
                    float32x4_t _out22 = vmulq_lane_f32(_v, vget_high_f32(_k2), 0); // 2,4,6,8

                    float32x4x2_t _out2 = vld2q_f32(outptr2);
                    _out2.val[0] = vaddq_f32(_out2.val[0], _out20); // 0,2,4,6
                    _out2.val[1] = vaddq_f32(_out2.val[1], _out21); // 1,3,5,7
                    vst2q_f32(outptr2, _out2);

                    _out2 = vld2q_f32(outptr2 + 2);
                    _out2.val[0] = vaddq_f32(_out2.val[0], _out22); // 2,4,6,8
                    vst2q_f32(outptr2 + 2, _out2);

                    r0 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                }

#endif // NEON

                for (; j < w; j++) {
                    float val = r0[0];

                    outptr0[0] += val * k0[0];
                    outptr0[1] += val * k0[1];
                    outptr0[2] += val * k0[2];

                    outptr1[0] += val * k1[0];
                    outptr1[1] += val * k1[1];
                    outptr1[2] += val * k1[2];

                    outptr2[0] += val * k2[0];
                    outptr2[1] += val * k2[1];
                    outptr2[2] += val * k2[2];

                    r0++;
                    outptr0 += 2;
                    outptr1 += 2;
                    outptr2 += 2;
                }
            }
        }
    }
}


// conv5x5s1
void conv5x5s1_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    // #pragma omp parallel for num_threads(opt.num_threads)
    int p, q;

    for (p = 0; p < outch; p++) {
        float* out = channel(top_blob, p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, bias0, top_blob.cstep);

        for (q = 0; q < inch; q++) {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 25 + q * 25;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            const float* r4 = img0 + w * 4;
            const float* r5 = img0 + w * 5;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

#if NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0 + 4);
            float32x4_t _k891011 = vld1q_f32(kernel0 + 8);
            float32x4_t _k12131415 = vld1q_f32(kernel0 + 12);
            float32x4_t _k16171819 = vld1q_f32(kernel0 + 16);
            float32x4_t _k20212223 = vld1q_f32(kernel0 + 20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);
#endif // NEON

            int i = 0;

            for (; i + 1 < outh; i += 2) {

#if NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__

                if (nn > 0) {
                    asm volatile(
                        // v11 = rx1 / rx3
                        // v12 = rx2
                        // v13 v14 = intermediate sum register

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v7.4s}, [%1]                  \n" // v7 = out

                        "0:                                        \n"

                        "prfm       pldl1keep, [%2, #128]          \n"
                        "ld1        {v8.4s}, [%2]                  \n" // v8 = out2

                        // r1
                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld1        {v9.4s, v10.4s}, [%4]          \n" // v9 v10 = r10 r14
                        "add        %4, %4, #16                    \n"

                        "ext        v11.16b, v9.16b, v10.16b, #4   \n" // r11
                        "fmul       v13.4s, v9.4s, %19.s[1]        \n"
                        "fmla       v8.4s,  v9.4s, %18.s[0]        \n"

                        "ext        v12.16b, v9.16b, v10.16b, #8   \n" // r12
                        "fmla       v7.4s,  v11.4s, %19.s[2]       \n"
                        "fmul       v14.4s, v11.4s, %18.s[1]       \n"

                        "ext        v11.16b, v9.16b, v10.16b, #12  \n" // r13
                        "fmla       v13.4s, v12.4s, %19.s[3]       \n"
                        "fmla       v8.4s,  v12.4s, %18.s[2]       \n"

                        "fmla       v7.4s,  v11.4s, %20.s[0]       \n"
                        "fmla       v14.4s, v11.4s, %18.s[3]       \n"

                        "prfm       pldl1keep, [%5, #256]          \n"

                        "fmla       v13.4s, v10.4s, %20.s[1]       \n"
                        "fmla       v8.4s,  v10.4s, %19.s[0]       \n"

                        // r2
                        "ld1        {v9.4s, v10.4s}, [%5]          \n" // v9 v10 = r20 r24
                        "add        %5, %5, #16                    \n"

                        "ext        v11.16b, v9.16b, v10.16b, #4   \n" // r21
                        "fmla       v7.4s,  v9.4s, %20.s[2]        \n"
                        "fmla       v14.4s, v9.4s, %19.s[1]        \n"

                        "ext        v12.16b, v9.16b, v10.16b, #8   \n" // r22
                        "fmla       v13.4s, v11.4s, %20.s[3]       \n"
                        "fmla       v8.4s,  v11.4s, %19.s[2]       \n"

                        "ext        v11.16b, v9.16b, v10.16b, #12  \n" // r23
                        "fmla       v7.4s,  v12.4s, %21.s[0]       \n"
                        "fmla       v14.4s, v12.4s, %19.s[3]       \n"

                        "fmla       v13.4s, v11.4s, %21.s[1]       \n"
                        "fmla       v8.4s,  v11.4s, %20.s[0]       \n"

                        "prfm       pldl1keep, [%6, #256]          \n"

                        "fmla       v7.4s,  v10.4s, %21.s[2]       \n"
                        "fmla       v14.4s, v10.4s, %20.s[1]       \n"

                        // r3
                        "ld1        {v9.4s, v10.4s}, [%6]          \n" // v9 v10 = r30 r34
                        "add        %6, %6, #16                    \n"

                        "ext        v11.16b, v9.16b, v10.16b, #4   \n" // r31
                        "fmla       v13.4s, v9.4s, %21.s[3]        \n"
                        "fmla       v8.4s,  v9.4s, %20.s[2]        \n"

                        "ext        v12.16b, v9.16b, v10.16b, #8   \n" // r32
                        "fmla       v7.4s,  v11.4s, %22.s[0]       \n"
                        "fmla       v14.4s, v11.4s, %20.s[3]       \n"

                        "ext        v11.16b, v9.16b, v10.16b, #12  \n" // r33
                        "fmla       v13.4s, v12.4s, %22.s[1]       \n"
                        "fmla       v8.4s,  v12.4s, %21.s[0]       \n"

                        "fmla       v7.4s,  v11.4s, %22.s[2]       \n"
                        "fmla       v14.4s, v11.4s, %21.s[1]       \n"

                        "prfm       pldl1keep, [%7, #256]          \n"

                        "fmla       v13.4s, v10.4s, %22.s[3]       \n"
                        "fmla       v8.4s,  v10.4s, %21.s[2]       \n"

                        // r4
                        "ld1        {v9.4s, v10.4s}, [%7]          \n" // v9 v10 = r40 r44
                        "add        %7, %7, #16                    \n"

                        "ext        v11.16b, v9.16b, v10.16b, #4   \n" // r41
                        "fmla       v7.4s,  v9.4s, %23.s[0]        \n"
                        "fmla       v14.4s, v9.4s, %21.s[3]        \n"

                        "ext        v12.16b, v9.16b, v10.16b, #8   \n" // r41
                        "fmla       v13.4s, v11.4s, %23.s[1]       \n"
                        "fmla       v8.4s,  v11.4s, %22.s[0]       \n"

                        "ext        v11.16b, v9.16b, v10.16b, #12  \n" // r41
                        "fmla       v7.4s,  v12.4s, %23.s[2]       \n"
                        "fmla       v14.4s, v12.4s, %22.s[1]       \n"

                        "fmla       v13.4s, v11.4s, %23.s[3]       \n"
                        "fmla       v8.4s,  v11.4s, %22.s[2]       \n"

                        "prfm       pldl1keep, [%3, #256]          \n"

                        "fmla       v7.4s,  v10.4s, %24.s[0]       \n"
                        "fmla       v14.4s, v10.4s, %22.s[3]       \n"

                        // r0 and r5
                        "ld1        {v9.4s, v10.4s}, [%3]          \n" // v9 v10 = r00 r04
                        "add        %3, %3, #16                    \n"

                        "ext        v11.16b, v9.16b, v10.16b, #4   \n" // r01
                        "fmla       v13.4s, v11.4s, %18.s[1]       \n"

                        "ext        v12.16b, v9.16b, v10.16b, #8   \n" // r02
                        "fmla       v7.4s, v12.4s, %18.s[2]        \n"

                        "ext        v11.16b, v9.16b, v10.16b, #12  \n" // r03

                        "prfm       pldl1keep, [%8, #256]          \n"

                        "fmla       v13.4s, v11.4s, %18.s[3]       \n"

                        // r5
                        "ld1        {v11.4s, v12.4s}, [%8]         \n" // v11 v12 = r50 r54
                        "add        %8, %8, #16                    \n"

                        "fmla       v8.4s,  v11.4s, %23.s[0]       \n"
                        "fmla       v14.4s, v12.4s, %24.s[0]       \n"

                        "fmla       v7.4s,  v9.4s,  %18.s[0]       \n"
                        "fmla       v13.4s, v10.4s, %19.s[0]       \n"

                        "ext        v9.16b,  v11.16b, v12.16b, #4  \n" // r51
                        "ext        v10.16b, v11.16b, v12.16b, #8  \n" // r52

                        "fmla       v14.4s, v9.4s, %23.s[1]        \n"

                        "ext        v9.16b, v11.16b, v12.16b, #12  \n" // r53
                        "fmla       v8.4s, v10.4s, %23.s[2]        \n"

                        "fmla       v14.4s, v9.4s, %23.s[3]        \n"

                        "fadd       v7.4s, v7.4s, v13.4s           \n"

                        "st1        {v7.4s}, [%1], #16             \n"

                        "fadd       v8.4s, v8.4s, v14.4s           \n"

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v7.4s}, [%1]                  \n" // v7 = out
                        "st1        {v8.4s}, [%2], #16             \n"

                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"
                        : "=r"(nn),    // %0
                        "=r"(outptr), // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),    // %3
                        "=r"(r1),    // %4
                        "=r"(r2),    // %5
                        "=r"(r3),    // %6
                        "=r"(r4),    // %7
                        "=r"(r5)     // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "7"(r4),
                        "8"(r5),
                        "w"(_k0123),   // %18
                        "w"(_k4567),   // %19
                        "w"(_k891011), // %20
                        "w"(_k12131415), // %21
                        "w"(_k16171819), // %22
                        "w"(_k20212223), // %23
                        "w"(_k24242424) // %24
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }

#else

                if (nn > 0) {
                    asm volatile(
                        //                     "veor       q13, q13            \n"
                        //                     "veor       q14, q14            \n"

                        "pld        [%1, #128]          \n"

                        "vld1.f32   {d14-d15}, [%1]     \n" // q7 = out

                        "0:                             \n"

                        // q11 = rx1 / rx3
                        // q12 = rx2

                        // q13 q14 = intermediate sum register

                        "pld        [%2, #128]          \n"

                        "vld1.f32   {d16-d17}, [%2]     \n" // q8 = out2

                        "pld        [%4, #256]          \n"

                        // r1
                        "vld1.f32   {d18-d21}, [%4]     \n" // q9 q10 = r10 r14
                        "add        %4, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n" // r11
                        "vmul.f32   q13, q9, %e19[1]    \n"
                        "vmla.f32   q8, q9, %e18[0]     \n"

                        "vext.32    q12, q9, q10, #2    \n" // r12
                        "vmla.f32   q7, q11, %f19[0]    \n"
                        "vmul.f32   q14, q11, %e18[1]   \n"

                        "vext.32    q11, q9, q10, #3    \n" // r13
                        "vmla.f32   q13, q12, %f19[1]   \n"
                        "vmla.f32   q8, q12, %f18[0]    \n"

                        "vmla.f32   q7, q11, %e20[0]    \n"
                        "vmla.f32   q14, q11, %f18[1]   \n"

                        "pld        [%5, #256]          \n"

                        "vmla.f32   q13, q10, %e20[1]   \n"
                        "vmla.f32   q8, q10, %e19[0]    \n"

                        // r2
                        "vld1.f32   {d18-d21}, [%5]     \n" // q9 q10 = r20 r24
                        "add        %5, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n" // r21
                        "vmla.f32   q7, q9, %f20[0]     \n"
                        "vmla.f32   q14, q9, %e19[1]    \n"

                        "vext.32    q12, q9, q10, #2    \n" // r22
                        "vmla.f32   q13, q11, %f20[1]   \n"
                        "vmla.f32   q8, q11, %f19[0]    \n"

                        "vext.32    q11, q9, q10, #3    \n" // r23
                        "vmla.f32   q7, q12, %e21[0]    \n"
                        "vmla.f32   q14, q12, %f19[1]   \n"

                        "vmla.f32   q13, q11, %e21[1]   \n"
                        "vmla.f32   q8, q11, %e20[0]    \n"

                        "pld        [%6, #256]          \n"

                        "vmla.f32   q7, q10, %f21[0]    \n"
                        "vmla.f32   q14, q10, %e20[1]   \n"

                        // r3
                        "vld1.f32   {d18-d21}, [%6]     \n" // q9 q10 = r30 r34
                        "add        %6, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n" // r31
                        "vmla.f32   q13, q9, %f21[1]    \n"
                        "vmla.f32   q8, q9, %f20[0]     \n"

                        "vext.32    q12, q9, q10, #2    \n" // r32
                        "vmla.f32   q7, q11, %e22[0]    \n"
                        "vmla.f32   q14, q11, %f20[1]   \n"

                        "vext.32    q11, q9, q10, #3    \n" // r33
                        "vmla.f32   q13, q12, %e22[1]   \n"
                        "vmla.f32   q8, q12, %e21[0]    \n"

                        "vmla.f32   q7, q11, %f22[0]    \n"
                        "vmla.f32   q14, q11, %e21[1]   \n"

                        "pld        [%7, #256]          \n"

                        "vmla.f32   q13, q10, %f22[1]   \n"
                        "vmla.f32   q8, q10, %f21[0]    \n"

                        // r4
                        "vld1.f32   {d18-d21}, [%7]     \n" // q9 q10 = r40 r44
                        "add        %7, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n" // r41
                        "vmla.f32   q7, q9, %e23[0]     \n"
                        "vmla.f32   q14, q9, %f21[1]    \n"

                        "vext.32    q12, q9, q10, #2    \n" // r42
                        "vmla.f32   q13, q11, %e23[1]   \n"
                        "vmla.f32   q8, q11, %e22[0]    \n"

                        "vext.32    q11, q9, q10, #3    \n" // r43
                        "vmla.f32   q7, q12, %f23[0]    \n"
                        "vmla.f32   q14, q12, %e22[1]   \n"

                        "vmla.f32   q13, q11, %f23[1]   \n"
                        "vmla.f32   q8, q11, %f22[0]    \n"

                        "pld        [%3, #256]          \n"

                        "vmla.f32   q7, q10, %e24[0]    \n"
                        "vmla.f32   q14, q10, %f22[1]   \n"

                        // r0 and r5
                        "vld1.f32   {d18-d21}, [%3]     \n" // q9 q10 = r00 r04
                        "add        %3, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n" // r01
                        "vmla.f32   q13, q11, %e18[1]   \n"

                        "vext.32    q12, q9, q10, #2    \n" // r02
                        "vmla.f32   q7, q12, %f18[0]    \n"

                        "vext.32    q11, q9, q10, #3    \n" // r03

                        "pld        [%8, #256]          \n"

                        "vmla.f32   q13, q11, %f18[1]   \n"

                        // r5
                        "vld1.f32   {d22-d25}, [%8]     \n" // q11 q12 = r50 r54
                        "add        %8, #16             \n"

                        "vmla.f32   q8, q11, %e23[0]    \n"
                        "vmla.f32   q14, q12, %e24[0]   \n"

                        "vmla.f32   q7, q9, %e18[0]     \n"
                        "vmla.f32   q13, q10, %e19[0]   \n"

                        "vext.32    q9, q11, q12, #1    \n" // r51
                        "vext.32    q10, q11, q12, #2   \n" // r52

                        "vmla.f32   q14, q9, %e23[1]    \n"

                        "vext.32    q9, q11, q12, #3    \n" // r53
                        "vmla.f32   q8, q10, %f23[0]    \n"

                        "vmla.f32   q14, q9, %f23[1]    \n"

                        "vadd.f32   q7, q7, q13         \n"

                        //                     "veor       q13, q13            \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "vadd.f32   q8, q8, q14         \n"

                        "pld        [%1, #128]          \n"

                        "vld1.f32   {d14-d15}, [%1]     \n" // q7 = out

                        //                     "veor       q14, q14            \n"

                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),    // %0
                        "=r"(outptr), // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),    // %3
                        "=r"(r1),    // %4
                        "=r"(r2),    // %5
                        "=r"(r3),    // %6
                        "=r"(r4),    // %7
                        "=r"(r5)     // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "7"(r4),
                        "8"(r5),
                        "w"(_k0123),   // %18
                        "w"(_k4567),   // %19
                        "w"(_k891011), // %20
                        "w"(_k12131415), // %21
                        "w"(_k16171819), // %22
                        "w"(_k20212223), // %23
                        "w"(_k24242424) // %24
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }

#endif // __aarch64__
#endif // NEON

                for (; remain > 0; remain--) {
                    float sum = 0;
                    float sum2 = 0;
#if NEON
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _k1 = vld1q_f32(k1);
                    float32x4_t _sum = vmulq_f32(_r1, _k1);
                    float32x4_t _sum2 = vmulq_f32(_r1, _k0123);

                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _k2 = vld1q_f32(k2);
                    _sum = vmlaq_f32(_sum, _r2, _k2);
                    _sum2 = vmlaq_f32(_sum2, _r2, _k1);

                    float32x4_t _r3 = vld1q_f32(r3);
                    float32x4_t _k3 = vld1q_f32(k3);
                    _sum = vmlaq_f32(_sum, _r3, _k3);
                    _sum2 = vmlaq_f32(_sum2, _r3, _k2);

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);
                    _sum2 = vmlaq_f32(_sum2, _r4, _k3);

                    float32x4_t _r0 = vld1q_f32(r0);
                    _sum = vmlaq_f32(_sum, _r0, _k0123);
                    float32x4_t _r5 = vld1q_f32(r5);
                    _sum2 = vmlaq_f32(_sum2, _r5, _k20212223);

                    float32x4_t _k_t4;
                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4;

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum = r4[4] * k4[4];

                    _r_t4 = vextq_f32(_r_t4, _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r4[4], _r_t4, 3);
                    _sum2 = vmlaq_f32(_sum2, _r_t4, _k_t4);

                    sum2 = r5[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                    float32x2_t _ss_ss2 = vpadd_f32(_ss, _ss2);

                    sum += vget_lane_f32(_ss_ss2, 0);
                    sum2 += vget_lane_f32(_ss_ss2, 1);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r1[3] * k0[3];
                    sum2 += r1[4] * k0[4];

                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r2[3] * k1[3];
                    sum2 += r2[4] * k1[4];

                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];
                    sum2 += r3[3] * k2[3];
                    sum2 += r3[4] * k2[4];

                    sum2 += r4[0] * k3[0];
                    sum2 += r4[1] * k3[1];
                    sum2 += r4[2] * k3[2];
                    sum2 += r4[3] * k3[3];
                    sum2 += r4[4] * k3[4];

                    sum2 += r5[0] * k4[0];
                    sum2 += r5[1] * k4[1];
                    sum2 += r5[2] * k4[2];
                    sum2 += r5[3] * k4[3];
                    sum2 += r5[4] * k4[4];
#endif // NEON
                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    outptr++;
                    outptr2++;
                }

                r0 += 4 + w;
                r1 += 4 + w;
                r2 += 4 + w;
                r3 += 4 + w;
                r4 += 4 + w;
                r5 += 4 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++) {

#if NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__

                if (nn > 0) {
                    asm volatile(
                        "prfm       pldl1keep, [%1, #128]          \n"
                        "prfm       pldl1keep, [%2, #256]          \n"

                        "ld1        {v8.4s, v9.4s}, [%2]           \n" // _r00 = vld1q_f32(r0+j);
                        "add        %2, %2, #16                    \n"

                        "0:                                        \n"

                        "ld1        {v7.4s}, [%1]                  \n" // _sum = vld1q_f32(outptr+j);

                        "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r01
                        "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r02
                        "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r03

                        "fmla       v7.4s,   v8.4s, %14.s[0]       \n"
                        "fmul       v13.4s, v10.4s, %14.s[1]       \n"

                        "prfm       pldl1keep, [%3, #256]          \n"

                        "fmul       v14.4s, v11.4s, %14.s[2]       \n"
                        "fmul       v15.4s, v12.4s, %14.s[3]       \n"
                        "fmla       v7.4s,   v9.4s, %15.s[0]       \n"

                        "ld1        {v8.4s, v9.4s}, [%3]           \n"
                        "add        %3, %3, #16                    \n"
                        "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r11
                        "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r12
                        "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r13

                        "fmla       v7.4s,   v8.4s, %15.s[1]       \n"
                        "fmla       v13.4s, v10.4s, %15.s[2]       \n"

                        "prfm       pldl1keep, [%4, #256]          \n"

                        "fmla       v14.4s, v11.4s, %15.s[3]       \n"
                        "fmla       v15.4s, v12.4s, %16.s[0]       \n"
                        "fmla       v7.4s,   v9.4s, %16.s[1]       \n"

                        "ld1        {v8.4s, v9.4s}, [%4]           \n"
                        "add        %4, %4, #16                    \n"
                        "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r21
                        "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r22
                        "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r23

                        "fmla       v7.4s,   v8.4s, %16.s[2]       \n"
                        "fmla       v13.4s, v10.4s, %16.s[3]       \n"

                        "prfm       pldl1keep, [%5, #256]          \n"

                        "fmla       v14.4s, v11.4s, %17.s[0]       \n"
                        "fmla       v15.4s, v12.4s, %17.s[1]       \n"
                        "fmla       v7.4s,   v9.4s, %17.s[2]       \n"

                        "ld1        {v8.4s, v9.4s}, [%5]           \n"
                        "add        %5, %5, #16                    \n"
                        "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r31
                        "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r32
                        "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r33

                        "fmla       v7.4s,   v8.4s, %17.s[3]       \n"
                        "fmla       v13.4s, v10.4s, %18.s[0]       \n"

                        "prfm       pldl1keep, [%6, #256]          \n"

                        "fmla       v14.4s, v11.4s, %18.s[1]       \n"
                        "fmla       v15.4s, v12.4s, %18.s[2]       \n"
                        "fmla       v7.4s,   v9.4s, %18.s[3]       \n"

                        "ld1        {v8.4s, v9.4s}, [%6]           \n"
                        "add        %6, %6, #16                    \n"
                        "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r41
                        "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r42
                        "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r43

                        "fmla       v7.4s,   v8.4s, %19.s[0]       \n"
                        "fmla       v13.4s, v10.4s, %19.s[1]       \n"
                        "fmla       v14.4s, v11.4s, %19.s[2]       \n"
                        "fmla       v15.4s, v12.4s, %19.s[3]       \n"
                        "fmla       v7.4s,   v9.4s, %20.s[0]       \n"

                        "fadd       v14.4s, v14.4s, v15.4s         \n"
                        "fadd       v7.4s,   v7.4s, v13.4s         \n"

                        "prfm       pldl1keep, [%2, #256]          \n"

                        "fadd       v7.4s,   v7.4s, v14.4s         \n"

                        "ld1        {v8.4s, v9.4s}, [%2]           \n"
                        "add        %2, %2, #16                    \n"

                        "st1        {v7.4s}, [%1], #16             \n"

                        "prfm       pldl1keep, [%1, #128]          \n"

                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"

                        "sub        %2, %2, #16                    \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4)    // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),   // %14
                        "w"(_k4567),   // %15
                        "w"(_k891011), // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424) // %20
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }

#else

                if (nn > 0) {
                    asm volatile(
                        //                     "veor       q15, q15            \n"// _sum3 = 0;

                        "pld        [%1, #128]          \n"

                        "pld        [%2, #256]          \n"

                        "vld1.f32   {d16-d19}, [%2]     \n" // _r00 = vld1q_f32(r0+j);
                        "add        %2, #16             \n"

                        "0:                             \n"

                        "vld1.f32   {d14-d15}, [%1]     \n" // _sum = vld1q_f32(outptr+j);
                        //                     "veor       q13, q13            \n"// _sum2 = 0;
                        //                     "veor       q14, q14            \n"// _sum3 = 0;

                        "vext.32    q10, q8, q9, #1     \n" // _r01
                        "vext.32    q11, q8, q9, #2     \n" // _r02
                        "vext.32    q12, q8, q9, #3     \n" // _r03

                        "vmla.f32   q7, q8, %e14[0]     \n"
                        "vmul.f32   q13, q10, %e14[1]   \n"

                        "pld        [%3, #256]          \n"

                        "vmul.f32   q14, q11, %f14[0]   \n"
                        "vmul.f32   q15, q12, %f14[1]   \n"
                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vld1.f32   {d16-d19}, [%3]     \n"
                        "add        %3, #16             \n"
                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"
                        "vext.32    q12, q8, q9, #3     \n"

                        "vmla.f32   q7, q8, %e15[1]     \n"
                        "vmla.f32   q13, q10, %f15[0]   \n"

                        "pld        [%4, #256]          \n"

                        "vmla.f32   q14, q11, %f15[1]   \n"
                        "vmla.f32   q15, q12, %e16[0]   \n"
                        "vmla.f32   q7, q9, %e16[1]     \n"

                        "vld1.f32   {d16-d19}, [%4]     \n"
                        "add        %4, #16             \n"
                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"
                        "vext.32    q12, q8, q9, #3     \n"

                        "vmla.f32   q7, q8, %f16[0]     \n"
                        "vmla.f32   q13, q10, %f16[1]   \n"

                        "pld        [%5, #256]          \n"

                        "vmla.f32   q14, q11, %e17[0]   \n"
                        "vmla.f32   q15, q12, %e17[1]   \n"
                        "vmla.f32   q7, q9, %f17[0]     \n"

                        "vld1.f32   {d16-d19}, [%5]     \n"
                        "add        %5, #16             \n"
                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"
                        "vext.32    q12, q8, q9, #3     \n"

                        "vmla.f32   q7, q8, %f17[1]     \n"
                        "vmla.f32   q13, q10, %e18[0]   \n"

                        "pld        [%6, #256]          \n"

                        "vmla.f32   q14, q11, %e18[1]   \n"
                        "vmla.f32   q15, q12, %f18[0]   \n"
                        "vmla.f32   q7, q9, %f18[1]     \n"

                        "vld1.f32   {d16-d19}, [%6]     \n"
                        "add        %6, #16             \n"
                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"
                        "vext.32    q12, q8, q9, #3     \n"

                        "vmla.f32   q7, q8, %e19[0]     \n"
                        "vmla.f32   q13, q10, %e19[1]   \n"
                        "vmla.f32   q14, q11, %f19[0]   \n"
                        "vmla.f32   q15, q12, %f19[1]   \n"
                        "vmla.f32   q7, q9, %e20[0]     \n"

                        "vadd.f32   q14, q14, q15       \n"
                        "vadd.f32   q7, q7, q13         \n"
                        //                     "veor       q15, q15            \n"// _sum3 = 0;

                        "pld        [%2, #256]          \n"

                        "vadd.f32   q7, q7, q14         \n"

                        "vld1.f32   {d16-d19}, [%2]     \n" // _r00 = vld1q_f32(r0+j);
                        "add        %2, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "pld        [%1, #128]          \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4)    // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),   // %14
                        "w"(_k4567),   // %15
                        "w"(_k891011), // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424) // %20
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }

#endif // __aarch64__
#endif // NEON

                for (; remain > 0; remain--) {
                    float sum = 0;
#if NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _sum = vmulq_f32(_r0, _k0123);

                    float32x4_t _r1 = vld1q_f32(r1);
                    _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                    float32x4_t _r2 = vld1q_f32(r2);
                    _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                    float32x4_t _r3 = vld1q_f32(r3);
                    _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);

                    float32x4_t _k_t4;
                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4;

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum = r4[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    sum += vget_lane_f32(_ss, 0);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
#endif
                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    outptr++;
                }

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
            }
        }
    }
}

// conv5x5s2
void conv5x5s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    // #pragma omp parallel for num_threads(opt.num_threads)
    int p, q;

    for (p = 0; p < outch; p++) {

        float* out = channel(top_blob, p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, bias0, top_blob.cstep);

        for (q = 0; q < inch; q++) {
            float* outptr = out;

            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 25 + q * 25;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            const float* r4 = img0 + w * 4;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

#if NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0 + 4);
            float32x4_t _k891011 = vld1q_f32(kernel0 + 8);
            float32x4_t _k12131415 = vld1q_f32(kernel0 + 12);
            float32x4_t _k16171819 = vld1q_f32(kernel0 + 16);
            float32x4_t _k20212223 = vld1q_f32(kernel0 + 20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);
#endif // NEON

            int i;

            for (i = 0; i < outh; i++) {

#if NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__

                if (nn > 0) {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n" // v8  = 0  2  4  6   q9  = 1  3  5  7

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v10.4s, v11.4s}, [%2]         \n" // v10 = 8 10 12 14   v11 = 9 11 13 15

                        "prfm       pldl1keep, [%1, #128]          \n"
                        "0:                                        \n"

                        "ld1        {v7.4s}, [%1]                  \n" // v7 = outptr

                        "ext        v12.16b, v8.16b, v10.16b, #4   \n" // v12 = 2 4 6 8
                        "ext        v11.16b, v9.16b, v11.16b, #4   \n" // v11 = 3 5 7 9
                        "ext        v10.16b, v8.16b, v10.16b, #8   \n" // v10 = 4 6 8 10

                        "fmla       v7.4s,  v8.4s, %14.s[0]        \n"
                        "fmul       v13.4s, v9.4s, %14.s[1]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"

                        "fmul       v14.4s, v12.4s, %14.s[2]       \n"
                        "fmul       v15.4s, v11.4s, %14.s[3]       \n"
                        "fmla       v7.4s,  v10.4s, %15.s[0]       \n"

                        "ld2        {v8.4s, v9.4s}, [%3], #32      \n"

                        "prfm       pldl1keep, [%3, #256]          \n"

                        "ld2        {v10.4s, v11.4s}, [%3]         \n"
                        "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                        "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                        "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                        "fmla       v7.4s,  v8.4s, %15.s[1]        \n"
                        "fmla       v13.4s, v9.4s, %15.s[2]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"

                        "fmla       v14.4s, v12.4s, %15.s[3]       \n"
                        "fmla       v15.4s, v11.4s, %16.s[0]       \n"
                        "fmla       v7.4s,  v10.4s, %16.s[1]       \n"

                        "ld2        {v8.4s, v9.4s}, [%4], #32      \n"

                        "prfm       pldl1keep, [%4, #256]          \n"

                        "ld2        {v10.4s, v11.4s}, [%4]         \n"
                        "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                        "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                        "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                        "fmla       v7.4s,  v8.4s, %16.s[2]        \n"
                        "fmla       v13.4s, v9.4s, %16.s[3]        \n"

                        "prfm       pldl1keep, [%5, #256]          \n"

                        "fmla       v14.4s, v12.4s, %17.s[0]       \n"
                        "fmla       v15.4s, v11.4s, %17.s[1]       \n"
                        "fmla       v7.4s,  v10.4s, %17.s[2]       \n"

                        "ld2        {v8.4s, v9.4s}, [%5], #32      \n"

                        "prfm       pldl1keep, [%5, #256]          \n"

                        "ld2        {v10.4s, v11.4s}, [%5]         \n"
                        "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                        "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                        "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                        "fmla       v7.4s,  v8.4s, %17.s[3]        \n"
                        "fmla       v13.4s, v9.4s, %18.s[0]        \n"

                        "prfm       pldl1keep, [%6, #256]          \n"

                        "fmla       v14.4s, v12.4s, %18.s[1]       \n"
                        "fmla       v15.4s, v11.4s, %18.s[2]       \n"
                        "fmla       v7.4s,  v10.4s, %18.s[3]       \n"

                        "ld2        {v8.4s, v9.4s}, [%6], #32      \n"

                        "prfm       pldl1keep, [%6, #256]          \n"

                        "ld2        {v10.4s, v11.4s}, [%6]         \n"
                        "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                        "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                        "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                        "fmla       v7.4s,   v8.4s, %19.s[0]       \n"
                        "fmla       v13.4s,  v9.4s, %19.s[1]       \n"
                        "fmla       v14.4s, v12.4s, %19.s[2]       \n"
                        "fmla       v15.4s, v11.4s, %19.s[3]       \n"
                        "fmla       v7.4s,  v10.4s, %20.s[0]       \n"

                        "prfm       pldl1keep, [%2, #256]          \n"

                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                        "fadd       v14.4s, v14.4s, v15.4s         \n"
                        "fadd       v7.4s,   v7.4s, v13.4s         \n"

                        "prfm       pldl1keep, [%2, #256]          \n"

                        "fadd       v7.4s, v7.4s, v14.4s           \n"

                        "ld2        {v10.4s, v11.4s}, [%2]         \n"
                        "st1        {v7.4s}, [%1], #16             \n"

                        "prfm       pldl1keep, [%1, #128]          \n"

                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"

                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4)    // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),   // %14
                        "w"(_k4567),   // %15
                        "w"(_k891011), // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424) // %20
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }

#else

                if (nn > 0) {
                    asm volatile(
                        //                     "veor       q15, q15            \n"// _sump3 = 0;
                        //                     "veor       q13, q13            \n"// _sump2 = 0;
                        //                     "veor       q14, q14            \n"// _sump3 = 0;

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n" // q8  = 0  2  4  6   q9  = 1  3  5  7

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d20-d23}, [%2]     \n" // q10 = 8 10 12 14   q11 = 9 11 13 15

                        "pld        [%1, #128]          \n"
                        "0:                             \n"

                        "vld1.f32   {d14-d15}, [%1]     \n" // q7 = outptr

                        "vext.32    q12, q8, q10, #1    \n" // q12 = 2 4 6 8
                        "vext.32    q11, q9, q11, #1    \n" // q11 = 3 5 7 9
                        "vext.32    q10, q8, q10, #2    \n" // q10 = 4 6 8 10

                        "vmla.f32   q7, q8, %e14[0]     \n"
                        "vmul.f32   q13, q9, %e14[1]    \n"

                        "pld        [%3, #256]          \n"

                        "vmul.f32   q14, q12, %f14[0]   \n"
                        "vmul.f32   q15, q11, %f14[1]   \n"
                        "vmla.f32   q7, q10, %e15[0]    \n"

                        "vld2.f32   {d16-d19}, [%3]!    \n"

                        "pld        [%3, #256]          \n"

                        "vld2.f32   {d20-d23}, [%3]     \n"
                        "vext.32    q12, q8, q10, #1    \n"
                        "vext.32    q11, q9, q11, #1    \n"
                        "vext.32    q10, q8, q10, #2    \n"

                        "vmla.f32   q7, q8, %e15[1]     \n"
                        "vmla.f32   q13, q9, %f15[0]    \n"

                        "pld        [%4, #256]          \n"

                        "vmla.f32   q14, q12, %f15[1]   \n"
                        "vmla.f32   q15, q11, %e16[0]   \n"
                        "vmla.f32   q7, q10, %e16[1]    \n"

                        "vld2.f32   {d16-d19}, [%4]!    \n"

                        "pld        [%4, #256]          \n"

                        "vld2.f32   {d20-d23}, [%4]     \n"
                        "vext.32    q12, q8, q10, #1    \n"
                        "vext.32    q11, q9, q11, #1    \n"
                        "vext.32    q10, q8, q10, #2    \n"

                        "vmla.f32   q7, q8, %f16[0]     \n"
                        "vmla.f32   q13, q9, %f16[1]    \n"

                        "pld        [%5, #256]          \n"

                        "vmla.f32   q14, q12, %e17[0]   \n"
                        "vmla.f32   q15, q11, %e17[1]   \n"
                        "vmla.f32   q7, q10, %f17[0]    \n"

                        "vld2.f32   {d16-d19}, [%5]!    \n"

                        "pld        [%5, #256]          \n"

                        "vld2.f32   {d20-d23}, [%5]     \n"
                        "vext.32    q12, q8, q10, #1    \n"
                        "vext.32    q11, q9, q11, #1    \n"
                        "vext.32    q10, q8, q10, #2    \n"

                        "vmla.f32   q7, q8, %f17[1]     \n"
                        "vmla.f32   q13, q9, %e18[0]    \n"

                        "pld        [%6, #256]          \n"

                        "vmla.f32   q14, q12, %e18[1]   \n"
                        "vmla.f32   q15, q11, %f18[0]   \n"
                        "vmla.f32   q7, q10, %f18[1]    \n"

                        "vld2.f32   {d16-d19}, [%6]!    \n"

                        "pld        [%6, #256]          \n"

                        "vld2.f32   {d20-d23}, [%6]     \n"
                        "vext.32    q12, q8, q10, #1    \n"
                        "vext.32    q11, q9, q11, #1    \n"
                        "vext.32    q10, q8, q10, #2    \n"

                        "vmla.f32   q7, q8, %e19[0]     \n"
                        "vmla.f32   q13, q9, %e19[1]    \n"
                        "vmla.f32   q14, q12, %f19[0]   \n"
                        "vmla.f32   q15, q11, %f19[1]   \n"
                        "vmla.f32   q7, q10, %e20[0]    \n"

                        "pld        [%2, #256]          \n"

                        "vld2.f32   {d16-d19}, [%2]!    \n" // q8  = 0  2  4  6   q9  = 1  3  5  7

                        "vadd.f32   q14, q14, q15       \n"
                        "vadd.f32   q7, q7, q13         \n"
                        //                     "veor       q15, q15            \n"// _sump3 = 0;
                        //                     "veor       q13, q13            \n"// _sump2 = 0;

                        "pld        [%2, #256]          \n"

                        "vadd.f32   q7, q7, q14         \n"

                        "vld2.f32   {d20-d23}, [%2]     \n" // q10 = 8 10 12 14   q11 = 9 11 13 15

                        //                     "veor       q14, q14            \n"// _sump3 = 0;

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "pld        [%1, #128]          \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #32             \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4)    // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),   // %14
                        "w"(_k4567),   // %15
                        "w"(_k891011), // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424) // %20
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }

#endif // __aarch64__
#endif // NEON

                for (; remain > 0; remain--) {
                    float sum = 0;
#if NEON
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _sum = vmulq_f32(_r0, _k0123);

                    float32x4_t _r1 = vld1q_f32(r1);
                    _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                    float32x4_t _r2 = vld1q_f32(r2);
                    _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                    float32x4_t _r3 = vld1q_f32(r3);
                    _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);

                    sum += r0[4] * k0[4];
                    sum += r1[4] * k1[4];
                    sum += r2[4] * k2[4];
                    sum += r3[4] * k3[4];
                    sum += r4[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    sum += vget_lane_f32(_ss, 0);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
#endif
                    *outptr += sum;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
            }
        }
    }
}

#if 0
void conv7x7s1_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;
    int p, q;

    for (p = 0; p < outch; p++) {
        float* out = channel(top_blob, p); // Mat out =top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, bias0, top_blob.cstep);

        for (q = 0; q < inch; q++) {
            float* outptr = out;

            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 49 + q * 49;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            const float* r4 = img0 + w * 4;
            const float* r5 = img0 + w * 5;
            const float* r6 = img0 + w * 6;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 7;
            const float* k2 = kernel0 + 14;
            const float* k3 = kernel0 + 21;
            const float* k4 = kernel0 + 28;
            const float* k5 = kernel0 + 35;
            const float* k6 = kernel0 + 42;

            int i = 0;

            for (; i < outh; i++) {
#if NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif //

#if NEON
#if __aarch64__
                float32x4_t _k0123 = vld1q_f32(k0);
                float32x4_t _k4567 = vld1q_f32(k0 + 4);
                float32x4_t _k78910 = vld1q_f32(k1);
                float32x4_t _k11121314 = vld1q_f32(k1 + 4);
                float32x4_t _k14151617 = vld1q_f32(k2);
                float32x4_t _k18192021 = vld1q_f32(k2 + 4);
                float32x4_t _k21222324 = vld1q_f32(k3);
                float32x4_t _k25262728 = vld1q_f32(k3 + 4);
                float32x4_t _k28293031 = vld1q_f32(k4);
                float32x4_t _k32333435 = vld1q_f32(k4 + 4);
                float32x4_t _k35363738 = vld1q_f32(k5);
                float32x4_t _k39404142 = vld1q_f32(k5 + 4);
                float32x4_t _k42434445 = vld1q_f32(k6);
                float32x4_t _k46474849 = vld1q_f32(k6 + 4);
#ifdef __clang__ // NEON && __aarch64__ && __clang__

                if (nn > 0) {
                    asm volatile(
                        // v0:  input / final output
                        // v1 v2 v3: = ri0 ri4 ri0n , i <-  1-7
                        // v4 = ri1 / ri3 / ri6
                        // v5 = ri2 / ri5
                        // v9 = intermediate sum register
                        "0:                                        \n"
                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v0.4s}, [%1]                  \n"

                        // i = 1
                        "prfm       pldl1keep, [%2, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%2]    \n"
                        "add        %2, %2, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmul       v9.4s, v1.4s, %18.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %18.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %18.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %18.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %19.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %19.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %19.s[2]         \n"

                        // i = 2
                        "prfm       pldl1keep, [%3, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%3]    \n" // v1 v2 v3: = r20 r24 r20n
                        "add        %3, %3, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n" // v4 = r21
                        "fmla       v9.4s, v1.4s, %20.s[0]         \n" // *+ r10
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n" // v5 = r22
                        "fmla       v0.4s, v4.4s, %20.s[1]         \n" // *+ r11
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n" // v4 = r23
                        "fmla       v9.4s, v5.4s, %20.s[2]         \n" // *+ r1
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n" // v5 = r25
                        "fmla       v0.4s, v4.4s, %20.s[3]         \n" // *+ r13
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n" // v4 = r26
                        "fmla       v9.4s, v2.4s, %21.s[0]         \n" // *+ r14
                        "fmla       v0.4s, v5.4s, %21.s[1]         \n" // *+ r15
                        "fmla       v9.4s, v4.4s, %21.s[2]         \n" // *+ r16

                        // i = 3
                        "prfm       pldl1keep, [%4, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%4]    \n"
                        "add        %4, %4, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmla       v9.4s, v1.4s, %22.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %22.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %22.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %23.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %23.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %23.s[2]         \n"

                        // i = 4
                        "prfm       pldl1keep, [%5, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%5]    \n"
                        "add        %5, %5, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmla       v9.4s, v1.4s, %24.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %24.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %24.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %25.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %25.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %25.s[2]         \n"

                        // i = 5
                        "prfm       pldl1keep, [%6, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%6]    \n"
                        "add        %6, %6, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmla       v9.4s, v1.4s, %26.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %26.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %26.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %27.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %27.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %27.s[2]         \n"

                        // i = 6
                        "prfm       pldl1keep, [%7, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%7]    \n"
                        "add        %7, %7, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmla       v9.4s, v1.4s, %28.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %28.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %28.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %29.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %29.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %29.s[2]         \n"

                        // i = 7
                        "prfm       pldl1keep, [%8, #384]          \n"
                        "ld1        {v1.4s, v2.4s, v3.4s}, [%8]    \n"
                        "add        %8, %8, #16                    \n"
                        "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                        "fmla       v9.4s, v1.4s, %30.s[0]         \n"
                        "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                        "fmla       v0.4s, v4.4s, %30.s[1]         \n"
                        "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                        "fmla       v9.4s, v5.4s, %30.s[2]         \n"
                        "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v4.4s, %30.s[3]         \n"
                        "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                        "fmla       v9.4s, v2.4s, %31.s[0]         \n"
                        "fmla       v0.4s, v5.4s, %31.s[1]         \n"
                        "fmla       v9.4s, v4.4s, %31.s[2]         \n"

                        "fadd       v0.4s, v0.4s, v9.4s            \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"

                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4),   // %6
                        "=r"(r5),   // %7
                        "=r"(r6)    // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "w"(_k0123),   // %18
                        "w"(_k4567),   // %19
                        "w"(_k78910),  // %20
                        "w"(_k11121314), // %21
                        "w"(_k14151617), // %22
                        "w"(_k18192021), // %23
                        "w"(_k21222324), // %24
                        "w"(_k25262728), // %25
                        "w"(_k28293031), // %26
                        "w"(_k32333435), // %27
                        "w"(_k35363738), // %28
                        "w"(_k39404142), // %29
                        "w"(_k42434445), // %30
                        "w"(_k46474849) // %31
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v9");
                }

#else

                /* NEON && __aarch64__ defined, but __clang__ not defined \
                When compiled with gcc, gcc does not accept over 30 operands*/
                for (; nn > 0; nn--) {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4_t _r00 = vld1q_f32(r0);           // 0 1 2 3
                    float32x4_t _r04 = vld1q_f32(r0 + 4);       // 4 5 6 7
                    float32x4_t _r00n = vld1q_f32(r0 + 8);      // 8 9 10 11
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1); // 1 2 3 4
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2); // 2 3 4 5
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3); // 3 4 5 6
                    float32x4_t _r05 = vextq_f32(_r04, _r00n, 1); // 5 6 7 8
                    float32x4_t _r06 = vextq_f32(_r04, _r00n, 2); // 6 7 8 9

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r14 = vld1q_f32(r1 + 4);
                    float32x4_t _r10n = vld1q_f32(r1 + 8);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);
                    float32x4_t _r15 = vextq_f32(_r14, _r10n, 1);
                    float32x4_t _r16 = vextq_f32(_r14, _r10n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r24 = vld1q_f32(r2 + 4);
                    float32x4_t _r20n = vld1q_f32(r2 + 8);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);
                    float32x4_t _r25 = vextq_f32(_r24, _r20n, 1);
                    float32x4_t _r26 = vextq_f32(_r24, _r20n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r34 = vld1q_f32(r3 + 4);
                    float32x4_t _r30n = vld1q_f32(r3 + 8);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);
                    float32x4_t _r35 = vextq_f32(_r34, _r30n, 1);
                    float32x4_t _r36 = vextq_f32(_r34, _r30n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r44 = vld1q_f32(r4 + 4);
                    float32x4_t _r40n = vld1q_f32(r4 + 8);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);
                    float32x4_t _r45 = vextq_f32(_r44, _r40n, 1);
                    float32x4_t _r46 = vextq_f32(_r44, _r40n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r54 = vld1q_f32(r5 + 4);
                    float32x4_t _r50n = vld1q_f32(r5 + 8);
                    float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
                    float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
                    float32x4_t _r53 = vextq_f32(_r50, _r54, 3);
                    float32x4_t _r55 = vextq_f32(_r54, _r50n, 1);
                    float32x4_t _r56 = vextq_f32(_r54, _r50n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4_t _r60 = vld1q_f32(r6);
                    float32x4_t _r64 = vld1q_f32(r6 + 4);
                    float32x4_t _r60n = vld1q_f32(r6 + 8);
                    float32x4_t _r61 = vextq_f32(_r60, _r64, 1);
                    float32x4_t _r62 = vextq_f32(_r60, _r64, 2);
                    float32x4_t _r63 = vextq_f32(_r60, _r64, 3);
                    float32x4_t _r65 = vextq_f32(_r64, _r60n, 1);
                    float32x4_t _r66 = vextq_f32(_r64, _r60n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptr, _sum);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    r6 += 4;
                    outptr += 4;
                }

#endif // __clang__
#else  //__aarch32__

                if (nn > 0) {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d24-d25}, [%1]     \n" // _sum
                        //                     "veor       q13, q13            \n"// _sum2 = 0;
                        //                     "veor       q14, q14            \n"// _sum3 = 0;
                        //                     "veor       q15, q15            \n"// _sum4 = 0;

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k0123 k4567
                        "add        %9, #28             \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%2]!      \n" // q0 = 0  1  2  3
                        "vmla.f32   q12, q0, d8[0]      \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2]       \n" // q2 = 4  5  6  7  q3 = 8  9 10 11
                        "vmul.f32   q13, q2, d10[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n" // q1 = 1  2  3  4
                        "vext.32    q10, q2, q3, #1     \n" // q10= 5  6  7  8
                        "vmul.f32   q14, q1, d8[1]      \n"
                        "vmul.f32   q15, q10, d10[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n" // q8 = 2  3  4  5
                        "vext.32    q11, q2, q3, #2     \n" // q11= 6  7  8  9
                        "vmla.f32   q12, q8, d9[0]      \n"
                        "vmla.f32   q13, q11, d11[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n" // q9 = 3  4  5  6
                        "vmla.f32   q14, q9, d9[1]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k78910 k11121314
                        "add        %9, #28             \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d0-d1}, [%3]!      \n"
                        "vmla.f32   q15, q0, d12[0]     \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d4-d7}, [%3]       \n"
                        "vmla.f32   q12, q2, d14[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q13, q1, d12[1]     \n"
                        "vmla.f32   q14, q10, d14[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q15, q8, d13[0]     \n"
                        "vmla.f32   q12, q11, d15[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q13, q9, d13[1]     \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k14151617 k18192021
                        "add        %9, #28             \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d0-d1}, [%4]!      \n"
                        "vmla.f32   q14, q0, d8[0]      \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d4-d7}, [%4]       \n"
                        "vmla.f32   q15, q2, d10[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q12, q1, d8[1]      \n"
                        "vmla.f32   q13, q10, d10[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q14, q8, d9[0]      \n"
                        "vmla.f32   q15, q11, d11[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q12, q9, d9[1]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k21222324 k25262728
                        "add        %9, #28             \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d0-d1}, [%5]!      \n"
                        "vmla.f32   q13, q0, d12[0]     \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5]       \n"
                        "vmla.f32   q14, q2, d14[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q15, q1, d12[1]     \n"
                        "vmla.f32   q12, q10, d14[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q13, q8, d13[0]     \n"
                        "vmla.f32   q14, q11, d15[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q15, q9, d13[1]     \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k28293031 k32333435
                        "add        %9, #28             \n"

                        "pld        [%6, #128]          \n"
                        "vld1.f32   {d0-d1}, [%6]!      \n"
                        "vmla.f32   q12, q0, d8[0]      \n"

                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d4-d7}, [%6]       \n"
                        "vmla.f32   q13, q2, d10[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q14, q1, d8[1]      \n"
                        "vmla.f32   q15, q10, d10[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q12, q8, d9[0]      \n"
                        "vmla.f32   q13, q11, d11[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q14, q9, d9[1]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k35363738 k39404142
                        "add        %9, #28             \n"

                        "pld        [%7, #128]          \n"
                        "vld1.f32   {d0-d1}, [%7]!      \n"
                        "vmla.f32   q15, q0, d12[0]     \n"

                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d4-d7}, [%7]       \n"
                        "vmla.f32   q12, q2, d14[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q13, q1, d12[1]     \n"
                        "vmla.f32   q14, q10, d14[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q15, q8, d13[0]     \n"
                        "vmla.f32   q12, q11, d15[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q13, q9, d13[1]     \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k42434445 k46474849
                        "sub        %9, #168            \n" // restore k0

                        "pld        [%8, #128]          \n"
                        "vld1.f32   {d0-d1}, [%8]!      \n"
                        "vmla.f32   q14, q0, d8[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d4-d7}, [%8]       \n"
                        "vmla.f32   q15, q2, d10[0]     \n"

                        "vext.32    q1, q0, q2, #1      \n"
                        "vext.32    q10, q2, q3, #1     \n"
                        "vmla.f32   q12, q1, d8[1]      \n"
                        "vmla.f32   q13, q10, d10[1]    \n"

                        "vext.32    q8, q0, q2, #2      \n"
                        "vext.32    q11, q2, q3, #2     \n"
                        "vmla.f32   q14, q8, d9[0]      \n"
                        "vmla.f32   q15, q11, d11[0]    \n"

                        "vext.32    q9, q0, q2, #3      \n"
                        "vmla.f32   q12, q9, d9[1]      \n"

                        "vadd.f32   q13, q13, q14       \n"
                        "vadd.f32   q13, q13, q15       \n"
                        "vadd.f32   q12, q12, q13       \n"

                        "vst1.f32   {d24-d25}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4),   // %6
                        "=r"(r5),   // %7
                        "=r"(r6),   // %8
                        "=r"(k0)    // %9
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(k0)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }

#endif // __aarch64__
#endif // NEON

                for (; remain > 0; remain--) {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];
                    sum += r0[5] * k0[5];
                    sum += r0[6] * k0[6];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];
                    sum += r1[5] * k1[5];
                    sum += r1[6] * k1[6];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];
                    sum += r2[5] * k2[5];
                    sum += r2[6] * k2[6];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];
                    sum += r3[5] * k3[5];
                    sum += r3[6] * k3[6];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
                    sum += r4[5] * k4[5];
                    sum += r4[6] * k4[6];

                    sum += r5[0] * k5[0];
                    sum += r5[1] * k5[1];
                    sum += r5[2] * k5[2];
                    sum += r5[3] * k5[3];
                    sum += r5[4] * k5[4];
                    sum += r5[5] * k5[5];
                    sum += r5[6] * k5[6];

                    sum += r6[0] * k6[0];
                    sum += r6[1] * k6[1];
                    sum += r6[2] * k6[2];
                    sum += r6[3] * k6[3];
                    sum += r6[4] * k6[4];
                    sum += r6[5] * k6[5];
                    sum += r6[6] * k6[6];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    r6++;
                    outptr++;
                }

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                r4 += 6;
                r5 += 6;
                r6 += 6;
            }
        }
    }
}

void conv7x7s2_neon(const Mat bottom_blob, Mat top_blob, const float* _kernel, const float* _bias)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;
    int p, q;

    for (p = 0; p < outch; p++) {
        float* out = channel(top_blob, p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, bias0, top_blob.cstep);

        for (q = 0; q < inch; q++) {
            float* outptr = out;

            const float* img0 = channel(bottom_blob, q);

            const float* kernel0 = kernel + p * inch * 49 + q * 49;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            const float* r4 = img0 + w * 4;
            const float* r5 = img0 + w * 5;
            const float* r6 = img0 + w * 6;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 7;
            const float* k2 = kernel0 + 14;
            const float* k3 = kernel0 + 21;
            const float* k4 = kernel0 + 28;
            const float* k5 = kernel0 + 35;
            const float* k6 = kernel0 + 42;

            int i = 0;

            for (; i < outh; i++) {
#if NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // NEON

#if NEON
#if __aarch64__
                float32x4_t _k0123 = vld1q_f32(k0);
                float32x4_t _k4567 = vld1q_f32(k0 + 4);
                float32x4_t _k78910 = vld1q_f32(k1);
                float32x4_t _k11121314 = vld1q_f32(k1 + 4);
                float32x4_t _k14151617 = vld1q_f32(k2);
                float32x4_t _k18192021 = vld1q_f32(k2 + 4);
                float32x4_t _k21222324 = vld1q_f32(k3);
                float32x4_t _k25262728 = vld1q_f32(k3 + 4);
                float32x4_t _k28293031 = vld1q_f32(k4);
                float32x4_t _k32333435 = vld1q_f32(k4 + 4);
                float32x4_t _k35363738 = vld1q_f32(k5);
                float32x4_t _k39404142 = vld1q_f32(k5 + 4);
                float32x4_t _k42434445 = vld1q_f32(k6);
                float32x4_t _k46474849 = vld1q_f32(k6 + 4);
#ifdef __clang__ // NEON && __aarch64__ && __clang__

                if (nn > 0) {
                    asm volatile(
                        // v0:  input / final output
                        // v1 v2: = _ri0/_ri1  first
                        // v3 v4: =                  then _r0_8101214/_r0_9111315
                        // v5 = ri2 / ri4 / ri6
                        // v6 = ri3 / ri5
                        // v9 = intermediate sum register
                        "0:                                        \n"
                        "prfm       pldl1keep, [%1, #128]          \n"
                        "ld1        {v0.4s}, [%1]                  \n"

                        // i = 1
                        "prfm       pldl1keep, [%2, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%2]           \n" // v1  v2 = _r00  _r01
                        "add        %2, %2, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%2]           \n" // v3  v4 = _r0_8101214 / _r0_9111315
                        "fmul       v9.4s, v1.4s, %18.s[0]         \n" // *+ _r00
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n" // v5 = _r02
                        "fmla       v0.4s, v2.4s, %18.s[1]         \n" // *+ _r01
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n" // v6 = _r03
                        "fmla       v9.4s, v5.4s, %18.s[2]         \n" // *+ _r02
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n" // v5 = _r04
                        "fmla       v0.4s, v6.4s, %18.s[3]         \n" // *+ _r03
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n" // v6 = _r05
                        "fmla       v9.4s, v5.4s, %19.s[0]         \n" // *+ _r04
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n" // v5 = _r06
                        "fmla       v0.4s, v6.4s, %19.s[1]         \n" // *+ _r05
                        "fmla       v9.4s, v5.4s, %19.s[2]         \n" // *+ _r06

                        // i = 2
                        "prfm       pldl1keep, [%3, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%3]           \n"
                        "add        %3, %3, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%3]           \n"
                        "fmla       v9.4s, v1.4s, %20.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %20.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %20.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %20.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %21.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %21.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %21.s[2]         \n"

                        // i = 3
                        "prfm       pldl1keep, [%4, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%4]           \n"
                        "add        %4, %4, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%4]           \n"
                        "fmla       v9.4s, v1.4s, %22.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %22.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %22.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %23.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %23.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %23.s[2]         \n"

                        // i = 4
                        "prfm       pldl1keep, [%5, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%5]           \n"
                        "add        %5, %5, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%5]           \n"
                        "fmla       v9.4s, v1.4s, %24.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %24.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %24.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %25.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %25.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %25.s[2]         \n"

                        // i = 5
                        "prfm       pldl1keep, [%6, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%6]           \n"
                        "add        %6, %6, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%6]           \n"
                        "fmla       v9.4s, v1.4s, %26.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %26.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %26.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %27.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %27.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %27.s[2]         \n"

                        // i = 6
                        "prfm       pldl1keep, [%7, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%7]           \n"
                        "add        %7, %7, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%7]           \n"
                        "fmla       v9.4s, v1.4s, %28.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %28.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %28.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %29.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %29.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %29.s[2]         \n"

                        // i = 7
                        "prfm       pldl1keep, [%8, #512]          \n"
                        "ld2        {v1.4s, v2.4s}, [%8]           \n"
                        "add        %8, %8, #32                    \n"
                        "ld2        {v3.4s, v4.4s}, [%8]           \n"
                        "fmla       v9.4s, v1.4s, %30.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                        "fmla       v0.4s, v2.4s, %30.s[1]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                        "fmla       v9.4s, v5.4s, %30.s[2]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                        "fmla       v0.4s, v6.4s, %30.s[3]         \n"
                        "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                        "fmla       v9.4s, v5.4s, %31.s[0]         \n"
                        "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                        "fmla       v0.4s, v6.4s, %31.s[1]         \n"
                        "fmla       v9.4s, v5.4s, %31.s[2]         \n"

                        "fadd       v0.4s, v0.4s, v9.4s            \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "subs       %w0, %w0, #1                   \n"
                        "bne        0b                             \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4),   // %6
                        "=r"(r5),   // %7
                        "=r"(r6)    // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "w"(_k0123),   // %18
                        "w"(_k4567),   // %19
                        "w"(_k78910),  // %20
                        "w"(_k11121314), // %21
                        "w"(_k14151617), // %22
                        "w"(_k18192021), // %23
                        "w"(_k21222324), // %24
                        "w"(_k25262728), // %25
                        "w"(_k28293031), // %26
                        "w"(_k32333435), // %27
                        "w"(_k35363738), // %28
                        "w"(_k39404142), // %29
                        "w"(_k42434445), // %30
                        "w"(_k46474849) // %31
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v9");
                }

#else /* NEON && __aarch64__ defined, but __clang__ not defined \
                When compiled with gcc, gcc does not accept over 30 operands*/

                for (; nn > 0; nn--) {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4x2_t _r00_02461357 = vld2q_f32(r0);
                    float32x4x2_t _r00nx2 = vld2q_f32(r0 + 8);
                    float32x4_t _r0_8101214 = _r00nx2.val[0];           // 8 10 12 14
                    float32x4_t _r0_9111315 = _r00nx2.val[1];           // 9 11 13 15
                    float32x4_t _r00 = _r00_02461357.val[0];            // 0 2 4 6
                    float32x4_t _r01 = _r00_02461357.val[1];            // 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0_8101214, 1); // 2 4 6 8
                    float32x4_t _r03 = vextq_f32(_r01, _r0_9111315, 1); // 3 5 7 9
                    float32x4_t _r04 = vextq_f32(_r00, _r0_8101214, 2); // 4 6 8 10
                    float32x4_t _r05 = vextq_f32(_r01, _r0_9111315, 2); // 5 7 9 11
                    float32x4_t _r06 = vextq_f32(_r00, _r0_8101214, 3); // 6 8 10 12

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4x2_t _r10_02461357 = vld2q_f32(r1);
                    float32x4x2_t _r10nx2 = vld2q_f32(r1 + 8);
                    float32x4_t _r1_8101214 = _r10nx2.val[0];
                    float32x4_t _r1_9111315 = _r10nx2.val[1];
                    float32x4_t _r10 = _r10_02461357.val[0];
                    float32x4_t _r11 = _r10_02461357.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1_8101214, 1);
                    float32x4_t _r13 = vextq_f32(_r11, _r1_9111315, 1);
                    float32x4_t _r14 = vextq_f32(_r10, _r1_8101214, 2);
                    float32x4_t _r15 = vextq_f32(_r11, _r1_9111315, 2);
                    float32x4_t _r16 = vextq_f32(_r10, _r1_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);

                    float32x4x2_t _r20_02461357 = vld2q_f32(r2);
                    float32x4x2_t _r20nx2 = vld2q_f32(r2 + 8);
                    float32x4_t _r2_8101214 = _r20nx2.val[0];
                    float32x4_t _r2_9111315 = _r20nx2.val[1];
                    float32x4_t _r20 = _r20_02461357.val[0];
                    float32x4_t _r21 = _r20_02461357.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2_8101214, 1);
                    float32x4_t _r23 = vextq_f32(_r21, _r2_9111315, 1);
                    float32x4_t _r24 = vextq_f32(_r20, _r2_8101214, 2);
                    float32x4_t _r25 = vextq_f32(_r21, _r2_9111315, 2);
                    float32x4_t _r26 = vextq_f32(_r20, _r2_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4x2_t _r30_02461357 = vld2q_f32(r3);
                    float32x4x2_t _r30nx2 = vld2q_f32(r3 + 8);
                    float32x4_t _r3_8101214 = _r30nx2.val[0];
                    float32x4_t _r3_9111315 = _r30nx2.val[1];
                    float32x4_t _r30 = _r30_02461357.val[0];
                    float32x4_t _r31 = _r30_02461357.val[1];
                    float32x4_t _r32 = vextq_f32(_r30, _r3_8101214, 1);
                    float32x4_t _r33 = vextq_f32(_r31, _r3_9111315, 1);
                    float32x4_t _r34 = vextq_f32(_r30, _r3_8101214, 2);
                    float32x4_t _r35 = vextq_f32(_r31, _r3_9111315, 2);
                    float32x4_t _r36 = vextq_f32(_r30, _r3_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);

                    float32x4x2_t _r40_02461357 = vld2q_f32(r4);
                    float32x4x2_t _r40nx2 = vld2q_f32(r4 + 8);
                    float32x4_t _r4_8101214 = _r40nx2.val[0];
                    float32x4_t _r4_9111315 = _r40nx2.val[1];
                    float32x4_t _r40 = _r40_02461357.val[0];
                    float32x4_t _r41 = _r40_02461357.val[1];
                    float32x4_t _r42 = vextq_f32(_r40, _r4_8101214, 1);
                    float32x4_t _r43 = vextq_f32(_r41, _r4_9111315, 1);
                    float32x4_t _r44 = vextq_f32(_r40, _r4_8101214, 2);
                    float32x4_t _r45 = vextq_f32(_r41, _r4_9111315, 2);
                    float32x4_t _r46 = vextq_f32(_r40, _r4_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);

                    float32x4x2_t _r50_02461357 = vld2q_f32(r5);
                    float32x4x2_t _r50nx2 = vld2q_f32(r5 + 8);
                    float32x4_t _r5_8101214 = _r50nx2.val[0];
                    float32x4_t _r5_9111315 = _r50nx2.val[1];
                    float32x4_t _r50 = _r50_02461357.val[0];
                    float32x4_t _r51 = _r50_02461357.val[1];
                    float32x4_t _r52 = vextq_f32(_r50, _r5_8101214, 1);
                    float32x4_t _r53 = vextq_f32(_r51, _r5_9111315, 1);
                    float32x4_t _r54 = vextq_f32(_r50, _r5_8101214, 2);
                    float32x4_t _r55 = vextq_f32(_r51, _r5_9111315, 2);
                    float32x4_t _r56 = vextq_f32(_r50, _r5_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4x2_t _r60_02461357 = vld2q_f32(r6);
                    float32x4x2_t _r60nx2 = vld2q_f32(r6 + 8);
                    float32x4_t _r6_8101214 = _r60nx2.val[0];
                    float32x4_t _r6_9111315 = _r60nx2.val[1];
                    float32x4_t _r60 = _r60_02461357.val[0];
                    float32x4_t _r61 = _r60_02461357.val[1];
                    float32x4_t _r62 = vextq_f32(_r60, _r6_8101214, 1);
                    float32x4_t _r63 = vextq_f32(_r61, _r6_9111315, 1);
                    float32x4_t _r64 = vextq_f32(_r60, _r6_8101214, 2);
                    float32x4_t _r65 = vextq_f32(_r61, _r6_9111315, 2);
                    float32x4_t _r66 = vextq_f32(_r60, _r6_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptr, _sum);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    r6 += 8;
                    outptr += 4;
                }

#endif // __clang__
#else

                if (nn > 0) {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d26-d27}, [%1]     \n" // _sum
                        //                     "veor       q14, q14            \n"// _sum2 = 0;
                        //                     "veor       q15, q15            \n"// _sum3 = 0;

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k0123 k4567
                        "add        %9, #28             \n"

                        "pld        [%2, #512]          \n"
                        "vld2.f32   {d0-d3}, [%2]!      \n" // q0 = 0  2  4  6  q1 = 1  3  5  7
                        "vmla.f32   q13, q0, d8[0]      \n"
                        "vmul.f32   q14, q1, d8[1]      \n"

                        "vld2.f32   {d4-d7}, [%2]       \n" // q2 = 8 10 12 14  q3 = 9 11 13 15
                        "vext.32    q8, q0, q2, #1      \n" // q8 = 2  4  6  8
                        "vext.32    q9, q1, q3, #1      \n" // q9 = 3  5  7  9
                        "vmul.f32   q15, q8, d9[0]      \n"
                        "vmla.f32   q13, q9, d9[1]      \n"

                        "vext.32    q10, q0, q2, #2     \n" // q10= 4  6  8 10
                        "vext.32    q11, q1, q3, #2     \n" // q11= 5  7  9 11
                        "vmla.f32   q14, q10, d10[0]    \n"
                        "vmla.f32   q15, q11, d10[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n" // q12= 6  8 10 12
                        "vmla.f32   q13, q12, d11[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k78910 k11121314
                        "add        %9, #28             \n"

                        "pld        [%3, #512]          \n"
                        "vld2.f32   {d0-d3}, [%3]!      \n"
                        "vmla.f32   q14, q0, d12[0]     \n"
                        "vmla.f32   q15, q1, d12[1]     \n"

                        "vld2.f32   {d4-d7}, [%3]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q13, q8, d13[0]     \n"
                        "vmla.f32   q14, q9, d13[1]     \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q15, q10, d14[0]    \n"
                        "vmla.f32   q13, q11, d14[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q14, q12, d15[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k14151617 k18192021
                        "add        %9, #28             \n"

                        "pld        [%4, #512]          \n"
                        "vld2.f32   {d0-d3}, [%4]!      \n"
                        "vmla.f32   q15, q0, d8[0]      \n"
                        "vmla.f32   q13, q1, d8[1]      \n"

                        "vld2.f32   {d4-d7}, [%4]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q14, q8, d9[0]      \n"
                        "vmla.f32   q15, q9, d9[1]      \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q13, q10, d10[0]    \n"
                        "vmla.f32   q14, q11, d10[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q15, q12, d11[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k21222324 k25262728
                        "add        %9, #28             \n"

                        "pld        [%5, #512]          \n"
                        "vld2.f32   {d0-d3}, [%5]!      \n"
                        "vmla.f32   q13, q0, d12[0]     \n"
                        "vmla.f32   q14, q1, d12[1]     \n"

                        "vld2.f32   {d4-d7}, [%5]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q15, q8, d13[0]     \n"
                        "vmla.f32   q13, q9, d13[1]     \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q14, q10, d14[0]    \n"
                        "vmla.f32   q15, q11, d14[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q13, q12, d15[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k28293031 k32333435
                        "add        %9, #28             \n"

                        "pld        [%6, #512]          \n"
                        "vld2.f32   {d0-d3}, [%6]!      \n"
                        "vmla.f32   q14, q0, d8[0]      \n"
                        "vmla.f32   q15, q1, d8[1]      \n"

                        "vld2.f32   {d4-d7}, [%6]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q13, q8, d9[0]      \n"
                        "vmla.f32   q14, q9, d9[1]      \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q15, q10, d10[0]    \n"
                        "vmla.f32   q13, q11, d10[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q14, q12, d11[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d12-d15}, [%9]     \n" // q6 q7 = k35363738 k39404142
                        "add        %9, #28             \n"

                        "pld        [%7, #512]          \n"
                        "vld2.f32   {d0-d3}, [%7]!      \n"
                        "vmla.f32   q15, q0, d12[0]     \n"
                        "vmla.f32   q13, q1, d12[1]     \n"

                        "vld2.f32   {d4-d7}, [%7]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q14, q8, d13[0]     \n"
                        "vmla.f32   q15, q9, d13[1]     \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q13, q10, d14[0]    \n"
                        "vmla.f32   q14, q11, d14[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q15, q12, d15[0]    \n"

                        "pld        [%9, #256]          \n"
                        "vld1.f32   {d8-d11}, [%9]      \n" // q4 q5 = k42434445 k46474849
                        "sub        %9, #168            \n" // restore k0

                        "pld        [%8, #512]          \n"
                        "vld2.f32   {d0-d3}, [%8]!      \n"
                        "vmla.f32   q13, q0, d8[0]      \n"
                        "vmla.f32   q14, q1, d8[1]      \n"

                        "vld2.f32   {d4-d7}, [%8]       \n"
                        "vext.32    q8, q0, q2, #1      \n"
                        "vext.32    q9, q1, q3, #1      \n"
                        "vmla.f32   q15, q8, d9[0]      \n"
                        "vmla.f32   q13, q9, d9[1]      \n"

                        "vext.32    q10, q0, q2, #2     \n"
                        "vext.32    q11, q1, q3, #2     \n"
                        "vmla.f32   q14, q10, d10[0]    \n"
                        "vmla.f32   q15, q11, d10[1]    \n"

                        "vext.32    q12, q0, q2, #3     \n"
                        "vmla.f32   q13, q12, d11[0]    \n"

                        "vadd.f32   q14, q14, q15       \n"
                        "vadd.f32   q13, q13, q14       \n"

                        "vst1.f32   {d26-d27}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),   // %0
                        "=r"(outptr), // %1
                        "=r"(r0),   // %2
                        "=r"(r1),   // %3
                        "=r"(r2),   // %4
                        "=r"(r3),   // %5
                        "=r"(r4),   // %6
                        "=r"(r5),   // %7
                        "=r"(r6),   // %8
                        "=r"(k0)    // %9
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(k0)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }

#endif // __aarch64__
#endif // NEON

                for (; remain > 0; remain--) {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];
                    sum += r0[5] * k0[5];
                    sum += r0[6] * k0[6];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];
                    sum += r1[5] * k1[5];
                    sum += r1[6] * k1[6];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];
                    sum += r2[5] * k2[5];
                    sum += r2[6] * k2[6];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];
                    sum += r3[5] * k3[5];
                    sum += r3[6] * k3[6];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
                    sum += r4[5] * k4[5];
                    sum += r4[6] * k4[6];

                    sum += r5[0] * k5[0];
                    sum += r5[1] * k5[1];
                    sum += r5[2] * k5[2];
                    sum += r5[3] * k5[3];
                    sum += r5[4] * k5[4];
                    sum += r5[5] * k5[5];
                    sum += r5[6] * k5[6];

                    sum += r6[0] * k6[0];
                    sum += r6[1] * k6[1];
                    sum += r6[2] * k6[2];
                    sum += r6[3] * k6[3];
                    sum += r6[4] * k6[4];
                    sum += r6[5] * k6[5];
                    sum += r6[6] * k6[6];

                    *outptr += sum;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
            }
        }
    }
}
#endif


PUBLIC void deconvcrop(Mat ori_blob, Mat crop_blob, int pad)
{
    int i = 0, j = 0;
    int w = ori_blob.w;
    int inch = ori_blob.c;

    int outw = crop_blob.w;
    int outh = crop_blob.h;

    float* img;
    float* img_out;

    for (; i < inch; i++) {
        img = channel(ori_blob, i);
        img_out = channel(crop_blob, i);

        for (j = 0; j < outh; j++) {
            memcpy(img_out + j * outw, img + (j + pad) * w + pad, outw * sizeof(float));
        }
    }
}

PUBLIC void get_warp_coeffs(int* buf, float theta, int patch_size, int sample_size, float nX, float nY)
{
    float cos_theta = SL_cos(theta);
    float sin_theta = SL_sin(theta);
    // float stride = 2 / (float)(patch_size - 1);
    float warp_ratio = sample_size / 2;
    int size = patch_size * patch_size;
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    float* x_warp = (float*)buf;
    float* y_warp = x_warp + patch_stride;

    float* x = y_warp + patch_stride;
    float* y = x + patch_stride;

    int i;

    for (i = 0; i < patch_size - 1; i++) {
        x[i] = ((float) -1.0f + (float)(i % patch_size * 2) / (float)(patch_size - 1)) * warp_ratio;
        fill(y + i * patch_size, x[i], patch_size);
    }

    x[patch_size - 1] = warp_ratio;

    fill(y + (patch_size - 1) * patch_size, warp_ratio, patch_size);

    for (i = 1; i < patch_size; i++) {
        memcpy(x + i * patch_size, x, patch_size * sizeof(float));
    }

    i = 0;
#if NEON
    const float* x_ptr = x;
    const float* y_ptr = y;
    float* x_warp_ptr = x_warp;
    float* y_warp_ptr = y_warp;
    float32x4_t _cos_theta = vdupq_n_f32(cos_theta);
    float32x4_t _psin_theta = vdupq_n_f32(sin_theta);
    float32x4_t _nsin_theta = vdupq_n_f32(-sin_theta);
    float32x4_t _warp;

    for (i = 0; i < size - 4; i += 4) {
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _cos_theta), vmulq_f32(vld1q_f32(y_ptr), _psin_theta));
        vst1q_f32(x_warp_ptr, _warp);
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _nsin_theta), vmulq_f32(vld1q_f32(y_ptr), _cos_theta));
        vst1q_f32(y_warp_ptr, _warp);
        x_ptr += 4;
        y_ptr += 4;
        x_warp_ptr += 4;
        y_warp_ptr += 4;
    }

#endif

    for (; i < size; i++) {
        x_warp[i] = x[i] * cos_theta + y[i] * sin_theta;
        y_warp[i] = -x[i] * sin_theta + y[i] * cos_theta;
    }

    int* xofs = buf;
    int* yofs = xofs + patch_stride;

    float* alpha0 = (float*)(yofs) + patch_stride;
    float* alpha1 = alpha0 + patch_stride;
    float* beta0 = alpha1 + patch_stride;
    float* beta1 = beta0 + patch_stride;


    for (i = 0; i < size; i++) {
        float temp;

        temp = x_warp[i] + nX;
        xofs[i] = SL_Floor(temp);
        alpha0[i] = (temp - xofs[i]);
        alpha1[i] = 1 - alpha0[i];
        /*xofs[i] += nX;*/

        temp = y_warp[i] + nY;
        yofs[i] = SL_Floor(temp);
        beta0[i] = (temp - yofs[i]);
        beta1[i] = 1 - beta0[i];
        // yofs[i] += nY;
    }
}

PUBLIC void get_warp_coeffs_rect(int* buf, float theta, int *patch_size, int *sample_size, float nX, float nY)
{
    float cos_theta = SL_cos(theta);
    float sin_theta = SL_sin(theta);
    // float stride = 2 / (float)(patch_size - 1);
	float warp_ratio_0 = sample_size[0] / 2;
	float warp_ratio_1 = sample_size[1] / 2;

    int size = patch_size[0] * patch_size[1];
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    float* x_warp = (float*)buf;
    float* y_warp = x_warp + patch_stride;

    float* x = y_warp + patch_stride;
    float* y = x + patch_stride;

	int i;
	for (i = 0; i < patch_size[1] - 1; i++)
	{
		x[i] = (-1.0 + (i % patch_size[1] * 2) / (float)(patch_size[1] - 1)) * warp_ratio_1;
	}
	x[patch_size[1] - 1] = warp_ratio_1;

	float temp_v;
	for (i = 0; i < patch_size[0]; i++)
	{
		temp_v = (-1.0 + (i % patch_size[0] * 2) / (float)(patch_size[0] - 1)) * warp_ratio_0;
		fill(y + i * patch_size[1], temp_v, patch_size[1]);
	}

	for (i = 1; i < patch_size[0]; i++)
	{
		memcpy(x + i * patch_size[1], x, patch_size[1] * sizeof(float));
	}

    i = 0;
#if NEON
    const float* x_ptr = x;
    const float* y_ptr = y;
    float* x_warp_ptr = x_warp;
    float* y_warp_ptr = y_warp;
    float32x4_t _cos_theta = vdupq_n_f32(cos_theta);
    float32x4_t _psin_theta = vdupq_n_f32(sin_theta);
    float32x4_t _nsin_theta = vdupq_n_f32(-sin_theta);
    float32x4_t _warp;

    for (i = 0; i < size - 4; i += 4) {
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _cos_theta), vmulq_f32(vld1q_f32(y_ptr), _psin_theta));
        vst1q_f32(x_warp_ptr, _warp);
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _nsin_theta), vmulq_f32(vld1q_f32(y_ptr), _cos_theta));
        vst1q_f32(y_warp_ptr, _warp);
        x_ptr += 4;
        y_ptr += 4;
        x_warp_ptr += 4;
        y_warp_ptr += 4;
    }

#endif

    for (; i < size; i++) {
        x_warp[i] = x[i] * cos_theta + y[i] * sin_theta;
        y_warp[i] = -x[i] * sin_theta + y[i] * cos_theta;
    }

    int* xofs = buf;
    int* yofs = xofs + patch_stride;

    float* alpha0 = (float*)(yofs) + patch_stride;
    float* alpha1 = alpha0 + patch_stride;
    float* beta0 = alpha1 + patch_stride;
    float* beta1 = beta0 + patch_stride;


    for (i = 0; i < size; i++) {
        float temp;

        temp = x_warp[i] + nX;
        xofs[i] = SL_Floor(temp);
        alpha0[i] = (temp - xofs[i]);
        alpha1[i] = 1 - alpha0[i];
        /*xofs[i] += nX;*/

        temp = y_warp[i] + nY;
        yofs[i] = SL_Floor(temp);
        beta0[i] = (temp - yofs[i]);
        beta1[i] = 1 - beta0[i];
        // yofs[i] += nY;
    }
}

PUBLIC int bilinear_warp_neon(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int patch_size, int sample_size)
{
    // memory consuming: [2(xofs, yofs) + 4(alpha, beta) + 4(Q)] * patch_size * patch_size
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    // int outch = top_blob.c;

    if (outh == 0 || outw == 0) {
        return -1;
    }

    int size = patch_size * patch_size;
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    int* buf = (int*)new_memory(top_blob);

    int* xofs = buf;
    int* yofs = xofs + patch_stride;

    float* alpha0 = (float*)(yofs) + patch_stride;
    float* alpha1 = alpha0 + patch_stride;
    float* beta0 = alpha1 + patch_stride;
    float* beta1 = beta0 + patch_stride;

    get_warp_coeffs(buf, theta, patch_size, sample_size, nX, nY);

    // #pragma omp parallel for num_threads(opt.num_threads)
    int q;
    int i;
    int Q11_index = -1;
    int Q21_index = -1;
    int Q12_index = -1;
    int Q22_index = -1;

    for (q = 0; q < channels; q++) {
        const unsigned char* src = bottom_blob.data + q * w * h;
        float* dst = channel(top_blob, q);

        float* Q11 = beta1 + patch_stride;
        float* Q21 = Q11 + patch_stride;
        float* Q12 = Q21 + patch_stride;
        float* Q22 = Q12 + patch_stride;

        // load src data
        for (i = 0; i < size; i++) {
            Q11_index = -1;
            Q21_index = -1;
            Q12_index = -1;
            Q22_index = -1;

            // check edge
            if ((-1 <= xofs[i]) && (xofs[i] < w) && (-1 <= yofs[i]) && (yofs[i] < h)) {
                Q11_index = yofs[i] * w + xofs[i];
                Q21_index = Q11_index + 1;
                Q12_index = (yofs[i] + 1) * w + xofs[i];
                Q22_index = Q12_index + 1;
            }

            if (yofs[i] == -1) {
                Q11_index = -1;
                Q21_index = -1;
            }

            if (yofs[i] == (h - 1)) {
                Q12_index = -1;
                Q22_index = -1;
            }

            if (xofs[i] == -1) {
                Q11_index = -1;
                Q12_index = -1;
            }

            if (xofs[i] == (w - 1)) {
                Q21_index = -1;
                Q22_index = -1;
            }

            Q11[i] = Q11_index > 0 ? (float)src[Q11_index] : 0;
            Q21[i] = Q21_index > 0 ? (float)src[Q21_index] : 0;
            Q12[i] = Q12_index > 0 ? (float)src[Q12_index] : 0;
            Q22[i] = Q22_index > 0 ? (float)src[Q22_index] : 0;
        }

        i = 0;
#if NEON
        const float* Q11_ptr = Q11;
        const float* Q21_ptr = Q21;
        const float* Q12_ptr = Q12;
        const float* Q22_ptr = Q22;
        const float* alpha0_ptr = alpha0;
        const float* alpha1_ptr = alpha1;
        const float* beta0_ptr = beta0;
        const float* beta1_ptr = beta1;
        float* dst_ptr = dst;

        float32x4_t _Q11;
        float32x4_t _Q21;
        float32x4_t _Q12;
        float32x4_t _Q22;

        for (i = 0; i < size - 4; i += 4) {
            _Q11 = vmulq_f32(vmulq_f32(vld1q_f32(Q11_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta1_ptr));
            _Q21 = vmulq_f32(vmulq_f32(vld1q_f32(Q21_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta1_ptr));
            _Q12 = vmulq_f32(vmulq_f32(vld1q_f32(Q12_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta0_ptr));
            _Q22 = vmulq_f32(vmulq_f32(vld1q_f32(Q22_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta0_ptr));

            vst1q_f32(dst_ptr, vaddq_f32(vaddq_f32(vaddq_f32(_Q11, _Q21), _Q12), _Q22));

            Q11_ptr += 4;
            Q21_ptr += 4;
            Q12_ptr += 4;
            Q22_ptr += 4;
            alpha0_ptr += 4;
            alpha1_ptr += 4;
            beta0_ptr += 4;
            beta1_ptr += 4;
            dst_ptr += 4;
        }

#endif

        for (; i < size; i++) {
            dst[i] = Q11[i] * alpha1[i] * beta1[i] +
                     Q21[i] * alpha0[i] * beta1[i] +
                     Q12[i] * alpha1[i] * beta0[i] +
                     Q22[i] * alpha0[i] * beta0[i];
        }
    }

    return 0;
}

PUBLIC int bilinear_warp_neon_rect(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int *patch_size, int *sample_size)
{
    // memory consuming: [2(xofs, yofs) + 4(alpha, beta) + 4(Q)] * patch_size * patch_size
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    // int outch = top_blob.c;

    if (outh == 0 || outw == 0) {
        return -1;
    }

    int size = patch_size[0] * patch_size[1];
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    int* buf = (int*)new_memory(top_blob);

    int* xofs = buf;
    int* yofs = xofs + patch_stride;

    float* alpha0 = (float*)(yofs) + patch_stride;
    float* alpha1 = alpha0 + patch_stride;
    float* beta0 = alpha1 + patch_stride;
    float* beta1 = beta0 + patch_stride;

    get_warp_coeffs_rect(buf, theta, patch_size, sample_size, nX, nY);

    // #pragma omp parallel for num_threads(opt.num_threads)
    int q;
    int i;
    int Q11_index = -1;
    int Q21_index = -1;
    int Q12_index = -1;
    int Q22_index = -1;

    for (q = 0; q < channels; q++) {
        const unsigned char* src = bottom_blob.data + q * w * h;
        float* dst = channel(top_blob, q);

        float* Q11 = beta1 + patch_stride;
        float* Q21 = Q11 + patch_stride;
        float* Q12 = Q21 + patch_stride;
        float* Q22 = Q12 + patch_stride;

        // load src data
        for (i = 0; i < size; i++) {
            Q11_index = -1;
            Q21_index = -1;
            Q12_index = -1;
            Q22_index = -1;

            // check edge
            if ((-1 <= xofs[i]) && (xofs[i] < w) && (-1 <= yofs[i]) && (yofs[i] < h)) {
                Q11_index = yofs[i] * w + xofs[i];
                Q21_index = Q11_index + 1;
                Q12_index = (yofs[i] + 1) * w + xofs[i];
                Q22_index = Q12_index + 1;
            }

            if (yofs[i] == -1) {
                Q11_index = -1;
                Q21_index = -1;
            }

            if (yofs[i] == (h - 1)) {
                Q12_index = -1;
                Q22_index = -1;
            }

            if (xofs[i] == -1) {
                Q11_index = -1;
                Q12_index = -1;
            }

            if (xofs[i] == (w - 1)) {
                Q21_index = -1;
                Q22_index = -1;
            }

            Q11[i] = Q11_index > 0 ? (float)src[Q11_index] : 0;
            Q21[i] = Q21_index > 0 ? (float)src[Q21_index] : 0;
            Q12[i] = Q12_index > 0 ? (float)src[Q12_index] : 0;
            Q22[i] = Q22_index > 0 ? (float)src[Q22_index] : 0;
        }

        i = 0;
#if NEON
        const float* Q11_ptr = Q11;
        const float* Q21_ptr = Q21;
        const float* Q12_ptr = Q12;
        const float* Q22_ptr = Q22;
        const float* alpha0_ptr = alpha0;
        const float* alpha1_ptr = alpha1;
        const float* beta0_ptr = beta0;
        const float* beta1_ptr = beta1;
        float* dst_ptr = dst;

        float32x4_t _Q11;
        float32x4_t _Q21;
        float32x4_t _Q12;
        float32x4_t _Q22;

        for (i = 0; i < size - 4; i += 4) {
            _Q11 = vmulq_f32(vmulq_f32(vld1q_f32(Q11_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta1_ptr));
            _Q21 = vmulq_f32(vmulq_f32(vld1q_f32(Q21_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta1_ptr));
            _Q12 = vmulq_f32(vmulq_f32(vld1q_f32(Q12_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta0_ptr));
            _Q22 = vmulq_f32(vmulq_f32(vld1q_f32(Q22_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta0_ptr));

            vst1q_f32(dst_ptr, vaddq_f32(vaddq_f32(vaddq_f32(_Q11, _Q21), _Q12), _Q22));

            Q11_ptr += 4;
            Q21_ptr += 4;
            Q12_ptr += 4;
            Q22_ptr += 4;
            alpha0_ptr += 4;
            alpha1_ptr += 4;
            beta0_ptr += 4;
            beta1_ptr += 4;
            dst_ptr += 4;
        }

#endif

        for (; i < size; i++) {
            dst[i] = Q11[i] * alpha1[i] * beta1[i] +
                     Q21[i] * alpha0[i] * beta1[i] +
                     Q12[i] * alpha1[i] * beta0[i] +
                     Q22[i] * alpha0[i] * beta0[i];
        }
    }

    return 0;
}

PUBLIC void get_doublewarp_coeffs(int* buf, float theta, int patch_size, int sample_size1, int sample_size2, float nX, float nY)
{
    float cos_theta = SL_cos(theta);
    float sin_theta = SL_sin(theta);
    float stride = 2 / (float)(patch_size - 1);
    float warp_ratio = (float)sample_size1 / 2;
    float scale_ratio = (float)sample_size2 / (float)sample_size1;
    int size = patch_size * patch_size;
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    float* x_warp = (float*)buf;
    float* y_warp = x_warp + 2 * patch_stride;

    float* x = y_warp + 2 * patch_stride;
    float* y = x + 2 * patch_stride;

    int i;

    for (i = 0; i < patch_size - 1; i++) {
        x[i] = (-1.0 + (i * stride)) * warp_ratio;
        fill(y + i * patch_size, x[i], patch_size);
    }

    x[patch_size - 1] = warp_ratio;

    fill(y + (patch_size - 1) * patch_size, warp_ratio, patch_size);

    for (i = 1; i < patch_size; i++) {
        memcpy(x + i * patch_size, x, patch_size * sizeof(float));
    }

    i = 0;
#if NEON
    const float* x_ptr = x;
    const float* y_ptr = y;
    float* x_warp_ptr = x_warp;
    float* y_warp_ptr = y_warp;
    float32x4_t _cos_theta = vdupq_n_f32(cos_theta);
    float32x4_t _psin_theta = vdupq_n_f32(sin_theta);
    float32x4_t _nsin_theta = vdupq_n_f32(-sin_theta);
    float32x4_t _warp;

    for (i = 0; i < size - 4; i += 4) {
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _cos_theta), vmulq_f32(vld1q_f32(y_ptr), _psin_theta));
        vst1q_f32(x_warp_ptr, _warp);
        _warp = vaddq_f32(vmulq_f32(vld1q_f32(x_ptr), _nsin_theta), vmulq_f32(vld1q_f32(y_ptr), _cos_theta));
        vst1q_f32(y_warp_ptr, _warp);
        x_ptr += 4;
        y_ptr += 4;
        x_warp_ptr += 4;
        y_warp_ptr += 4;
    }

#endif

    for (; i < size; i++) {
        x_warp[i] = x[i] * cos_theta + y[i] * sin_theta;
        y_warp[i] = -x[i] * sin_theta + y[i] * cos_theta;
    }

    int* xofs = buf;
    int* yofs = xofs + 2 * patch_stride;

    float* alpha0 = (float*)(yofs) + 2 * patch_stride;
    float* alpha1 = alpha0 + 2 * patch_stride;
    float* beta0 = alpha1 + 2 * patch_stride;
    float* beta1 = beta0 + 2 * patch_stride;

    float temp, temp1;

    for (i = 0; i < size; i++) {
        temp = x_warp[i] + nX;
        temp1 = x_warp[i] * scale_ratio + nX;
        xofs[i] = SL_Floor(temp);
        xofs[i + patch_stride] = SL_Floor(temp1);
        alpha0[i] = (temp - xofs[i]);
        alpha0[i + patch_stride] = (temp1 - xofs[i + patch_stride]);
        alpha1[i] = 1 - alpha0[i];
        alpha1[i + patch_stride] = 1 - alpha0[i + patch_stride];
        /*xofs[i] += nX;*/

        temp = y_warp[i] + nY;
        temp1 = y_warp[i] * scale_ratio + nY;
        yofs[i] = SL_Floor(temp);
        yofs[i + patch_stride] = SL_Floor(temp1);
        beta0[i] = (temp - yofs[i]);
        beta0[i + patch_stride] = (temp1 - yofs[i + patch_stride]);
        beta1[i] = 1 - beta0[i];
        beta1[i + patch_stride] = 1 - beta0[i + patch_stride];
        //yofs[i] += nY;
    }
}

PUBLIC int bilinear_doublewarp_neon(const MatImg bottom_blob, Mat top_blob, float theta, float nX, float nY, int patch_size, int sample_size1, int sample_size2)
{
    //memory consuming: [2(xofs, yofs) + 4(alpha, beta) + 4(Q)] * patch_size * patch_size
    int h = bottom_blob.h;
    //int w = bottom_blob.w;
    //int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;


    if (outh == 0 || outw == 0) {
        return -1;
    }

    int size = patch_size * patch_size;
    int patch_stride = alignPtr(size, MALLOC_ALIGN);

    int* buf = (int*)new_memory(top_blob);

    int* xofs = buf;
    int* yofs = xofs + 2 * patch_stride;

    float* alpha0 = (float*)(yofs) + 2 * patch_stride;
    float* alpha1 = alpha0 + 2 * patch_stride;
    float* beta0 = alpha1 + 2 * patch_stride;
    float* beta1 = beta0 + 2 * patch_stride;


    get_doublewarp_coeffs(buf, theta, patch_size, sample_size1, sample_size2, nX, nY);

    //#pragma omp parallel for num_threads(opt.num_threads)
    int q;
    int i;
    int Q11_index = -1;
    int Q21_index = -1;
    int Q12_index = -1;
    int Q22_index = -1;

    float* Q11 = beta1 + 2 * patch_stride;
    float* Q21 = Q11 + patch_stride;
    float* Q12 = Q21 + patch_stride;
    float* Q22 = Q12 + patch_stride;

    for (q = 0; q < outch; q++) {
        const unsigned char* src = bottom_blob.data;
        float* dst = channel(top_blob, q);
        xofs = xofs + q * patch_stride;
        yofs = yofs + q * patch_stride;

        alpha0 = alpha0 + q * patch_stride;
        alpha1 = alpha1 + q * patch_stride;
        beta0 = beta0 + q * patch_stride;
        beta1 = beta1 + q * patch_stride;

        //load src data
        for (i = 0; i < size; i++) {
            Q11_index = -1;
            Q21_index = -1;
            Q12_index = -1;
            Q22_index = -1;

            //check edge
            if ((-1 <= xofs[i]) && (xofs[i] < (bottom_blob.w)) && (-1 <= yofs[i]) && (yofs[i] < (bottom_blob.h))) {
                Q11_index = yofs[i] * bottom_blob.w + xofs[i];
                Q21_index = Q11_index + 1;
                Q12_index = (yofs[i] + 1) * bottom_blob.w + xofs[i];
                Q22_index = Q12_index + 1;
            }

            if (yofs[i] == -1) {
                Q11_index = -1;
                Q21_index = -1;
            }

            if (yofs[i] == (bottom_blob.h) - 1) {
                Q12_index = -1;
                Q22_index = -1;
            }

            if (xofs[i] == -1) {
                Q11_index = -1;
                Q12_index = -1;
            }

            if (xofs[i] == (bottom_blob.w) - 1) {
                Q21_index = -1;
                Q22_index = -1;
            }

            Q11[i] = Q11_index > 0 ? (float)src[Q11_index] : 0;
            Q21[i] = Q21_index > 0 ? (float)src[Q21_index] : 0;
            Q12[i] = Q12_index > 0 ? (float)src[Q12_index] : 0;
            Q22[i] = Q22_index > 0 ? (float)src[Q22_index] : 0;
        }

        i = 0;
#if NEON
        const float* Q11_ptr = Q11;
        const float* Q21_ptr = Q21;
        const float* Q12_ptr = Q12;
        const float* Q22_ptr = Q22;
        const float* alpha0_ptr = alpha0;
        const float* alpha1_ptr = alpha1;
        const float* beta0_ptr = beta0;
        const float* beta1_ptr = beta1;
        float* dst_ptr = dst;

        float32x4_t _Q11;
        float32x4_t _Q21;
        float32x4_t _Q12;
        float32x4_t _Q22;

        for (i = 0; i < size - 4; i += 4) {
            _Q11 = vmulq_f32(vmulq_f32(vld1q_f32(Q11_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta1_ptr));
            _Q21 = vmulq_f32(vmulq_f32(vld1q_f32(Q21_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta1_ptr));
            _Q12 = vmulq_f32(vmulq_f32(vld1q_f32(Q12_ptr), vld1q_f32(alpha1_ptr)), vld1q_f32(beta0_ptr));
            _Q22 = vmulq_f32(vmulq_f32(vld1q_f32(Q22_ptr), vld1q_f32(alpha0_ptr)), vld1q_f32(beta0_ptr));

            vst1q_f32(dst_ptr, vaddq_f32(vaddq_f32(vaddq_f32(_Q11, _Q21), _Q12), _Q22));

            Q11_ptr += 4;
            Q21_ptr += 4;
            Q12_ptr += 4;
            Q22_ptr += 4;
            alpha0_ptr += 4;
            alpha1_ptr += 4;
            beta0_ptr += 4;
            beta1_ptr += 4;
            dst_ptr += 4;
        }

#endif

        for (; i < size; i++) {
            dst[i] = Q11[i] * alpha1[i] * beta1[i] + \
                     Q21[i] * alpha0[i] * beta1[i] + \
                     Q12[i] * alpha1[i] * beta0[i] + \
                     Q22[i] * alpha0[i] * beta0[i];
        }
    }

    return 0;
}


// get std
PUBLIC float std_neon(float* src, float mean, int size)
{
    int i = 0;
    float std = 0.f;
#if NEON
    const float* img_in = src;
    float32x4_t _std = vdupq_n_f32(0.f);
    float32x4_t _mean = vdupq_n_f32(-mean);
    float32x4_t _temp;

    for (i = 0; i < size - 4; i += 4) {
        _temp = vaddq_f32(vld1q_f32(img_in), _mean);
        _std = vaddq_f32(vmulq_f32(_temp, _temp), _std);
        img_in += 4;
    }

#if __aarch64__
    std = vaddvq_f32(_std);
#else
    float32x2_t _sum = vadd_f32(vget_low_f32(_std), vget_high_f32(_std));
    float32x2_t _ss2 = vpadd_f32(_sum, _sum);
    std = vget_lane_f32(_ss2, 0);
#endif

#endif

    for (; i < size; i++) {
        std += (*(src + i) - mean) * ((*(src + i) - mean));
    }

    std /= (size - 1);

    return std;
}

// LeakyRelu
PUBLIC void leakyrelu_neon(Mat mat, float slope)
{
    float* ptr = mat.data;
    int size = total(mat);
#if NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // NEON

#if NEON
#if __aarch64__
    float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _slope = vdupq_n_f32(slope);

    for (; nn > 0; nn--) {
        float32x4_t _p = vld1q_f32(ptr);
        uint32x4_t _lemask = vcleq_f32(_p, _zero);
        float32x4_t _ps = vmulq_f32(_p, _slope);
        _p = vbslq_f32(_lemask, _ps, _p);
        vst1q_f32(ptr, _p);

        ptr += 4;
    }

#else

    if (nn > 0) {
        asm volatile(
            "veor       q1, q0, q0          \n"
            "vdup.f32   q2, %4              \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1 :128]  \n"
            "vcle.f32   q3, q0, q1          \n"
            "vmul.f32   q4, q0, q2          \n"
            "vbit.32    q0, q4, q3          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d1}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn), // %0
            "=r"(ptr) // %1
            : "0"(nn),
            "1"(ptr),
            "r"(slope) // %4
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
    }

#endif // __aarch64__
#endif // NEON

    for (; remain > 0; remain--) {
        if (*ptr < 0) {
            *ptr *= slope;
        }

        ptr++;
    }
}


PUBLIC void normalize_neon(float* src, float* dst, const float mean, const float std, int size)
{
    int i = 0;
#if NEON
    const float* img_in = src;
    float* img_out = dst;
    float32x4_t _grayscale = vdupq_n_f32(1 / std);
    float32x4_t _mean = vdupq_n_f32(-mean);
    float32x4_t sub;

    for (i = 0; i < size - 4; i += 4) {
        sub = vmulq_f32(vaddq_f32(vld1q_f32(img_in), _mean), _grayscale);
        vst1q_f32(img_out, sub);
        img_in += 4;
        img_out += 4;
    }

#endif

    for (; i < size; i++) {
        *(dst + i) = (*(src + i) - mean) / std;
    }
}

PUBLIC void copy(const Mat src, Mat dst)
{
    // int w = src.w;
    // int h = src.h;
    int i = 0;
    int inch = src.c;
    float* img;
    float* img_out;

    for (; i < inch; i++) {
        img = channel(src, i);
        img_out = channel(dst, i);
        memcpy(img_out, img, src.cstep * sizeof(float));
    }
}

PUBLIC int short_to_float(float* param_f, short* param_s, int len, int expansion)
{
    float expansion_f = 1.0f / expansion;
    int remain = 0;

    for (; remain < len; remain++) {
        *(param_f + remain) = *(param_s + remain) * expansion_f;
    }

    return 0;
}

PUBLIC void* sl_aligned_malloc(size_t size, size_t align)
{
    void* origin_p;
    void* aligned_p;

    if (align & (align - 1)) {
        //????alignment????2??n??????????
        return ((void*)NULL);
    }

    if (!size) {
        return ((void*)NULL);
    }

    //??alignment??????????2*sizeof(void*),??????????????
    if (align < 2 * sizeof(void*)) {
        align = 2 * sizeof(void*);
    }

    origin_p = malloc(size + align);

    if (!origin_p) {
        return ((void*)NULL);
    }

    // Align  We have at least sizeof (void *) space below malloc'd ptr.
    aligned_p = (void*)(((size_t)origin_p + align) & ~((size_t)align - 1));

    ((void**)aligned_p)[-1] = origin_p; // save the original ptr

    return aligned_p;
}

PUBLIC void sl_aligned_free(void* p)
{
    if (p) {
        free(((void**)p)[-1]);
    }
}

#if 0
PUBLIC void* fast_malloc(size_t size)
{
#if _MSC_VER
    return sl_aligned_malloc(size, MALLOC_ALIGN);
#elif (defined __linux__) || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;

    if (posix_memalign(&ptr, MALLOC_ALIGN, size + MALLOC_OVERREAD)) {
        ptr = 0;
    }

    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(MALLOC_ALIGN, size + MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN + MALLOC_OVERREAD);

    if (!udata) {
        return 0;
    }

    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}
PUBLIC void fast_free(void* ptr)
{
    if (ptr) {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined __linux__) || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}

#endif