/******************************************************************************
 * Copyright (c); 2018-2019 Silead Inc.
 * All rights reserved
 *
 * File:         SL_Math.h
 * Description:  Silead Math Functions
 * Author:       Warren Zhao
 * Create:       2017-07-10
 *
 * The present software is the confidential and proprietary information of
 * Silead Inc. You shall not disclose the present software and shall use it
 * only in accordance with the terms of the license agreement you entered
 * into with Silead Inc. This software may be subject to export or import
 * laws in certain countries.
 *
 * ----------------------------- Revision History -----------------------------
 * <Reviser>            <Date>          <Version>       <Description>
 * Warren Zhao          2017-07-10      A001            Init Version
 * Frank Zhou           2020-01-14      A002            Improve Precision
 *
 *****************************************************************************/

#ifndef SL_MATH_H
#define SL_MATH_H

#ifndef SL_MATH_TEST
#define SL_MATH_TEST     0
#endif

#ifndef MATH_2_SL_MATH
#define MATH_2_SL_MATH  0//static declare follows non-static error
#endif

/************************* Const Macro Declaration ***************************/

#ifndef SL_E
#define SL_E             2.7182818284590452354      //e
#endif

#ifndef SL_LOG2E
#define SL_LOG2E         1.4426950408889634074      //log2(e); = 1/ln(2);
#endif

#ifndef SL_LOG10E
#define SL_LOG10E        0.43429448190325182765     //log10(e); = 1/ln(10);
#endif

#ifndef SL_LN2
#define SL_LN2           0.69314718055994530942     //ln(2);
#endif

#ifndef SL_LN10
#define SL_LN10          2.30258509299404568402     //ln(10);
#endif

#ifndef SL_PI
#define SL_PI            3.14159265358979323846     //π
#endif

#ifndef SL_2PI
#define SL_2PI           6.28318530717985647692     //2π
#endif

#ifndef SL_PI_DIV_2
#define SL_PI_DIV_2      1.57079632679489661923     //π/ 2
#endif

#ifndef SL_PI_DIV_4
#define SL_PI_DIV_4      0.78539816339744830962     //π/ 4
#endif

#ifndef SL_1_DIV_PI
#define SL_1_DIV_PI      0.31830988618379067154     //1 / π
#endif

#ifndef SL_2_DIV_PI
#define SL_2_DIV_PI      0.63661977236758134308     //2 / π
#endif

#ifndef SL_1_DIV_SQRT2
#define SL_1_DIV_SQRT2   0.70710678118654752440     //1 / √π
#endif

#ifndef SL_2_DIV_SQRTPI
#define SL_2_DIV_SQRTPI  1.12837916709551257390     //2 / √π
#endif

#ifndef SL_SQRT2
#define SL_SQRT2         1.41421356237309504880     //1 / √2
#endif

#ifndef NAN
#define NAN              -1.0//0.0 / 0.0                  //Not a number
#endif

#ifndef INF_P
#define INF_P            -1.0//1.0 / 0.0                  //Positive Infinity
#endif

#ifndef INF_N
#define INF_N           -1.0//-1.0 / 0.0                  //Negative Infinity
#endif

#ifndef NULL
#define NULL            (void*)(0)
#endif

/************************* Func Macro Declaration ****************************/

#ifndef ABS                                         //绝对值函数
#define ABS(x)           (((x) >= 0) ? (x) : (-(x)))
#endif

#ifndef GetMin                                      //三个数中取最小值
#define GetMin(a, b, c)  ((a > b ? b : a) > c ? c : (a > b ? b : a))
#endif

/************************* Math2SL_Math Transfer *****************************/

#if MATH_2_SL_MATH

#ifndef    abs
#define    abs     SL_abs
#endif

#ifndef    absf
#define    absf    SL_absf
#endif

#ifndef    fabs
#define    fabs    SL_fabs
#endif

#ifndef    fabsf
#define    fabsf   SL_fabsf
#endif

#ifndef    floor
#define    floor   SL_floor
#endif

#ifndef    ceil
#define    ceil    SL_ceil
#endif

#ifndef    round
#define    round   SL_round
#endif

#ifndef    modf
#define    modf    SL_modf
#endif

#ifndef    sqrt
#define    sqrt    SL_sqrt
#endif

#ifndef    sqrtf
#define    sqrtf   SL_sqrtf
#endif

#ifndef    sqrti
#define    sqrti   SL_sqrti
#endif

#ifndef    sin
#define    sin     SL_sin
#endif

#ifndef    cos
#define    cos     SL_cos
#endif

#ifndef    sinf
#define    sinf    SL_sinf
#endif

#ifndef    cosf
#define    cosf    SL_cosf
#endif

#ifndef    ln
#define    ln      SL_ln
#endif

#ifndef    log
#define    log     SL_ln
#endif

#ifndef    log2
#define    log2    SL_log2
#endif

#ifndef    log10
#define    log10   SL_log10
#endif

#ifndef    exp
#define    exp     SL_exp
#endif

#ifndef    pow
#define    pow     SL_pow
#endif

#ifndef    powf
#define    powf     SL_powf
#endif

#ifndef    asin
#define    asin    SL_asin
#endif

#ifndef    acos
#define    acos    SL_acos
#endif

#ifndef    atan
#define    atan    SL_atan
#endif

#ifndef    atan2
#define    atan2   SL_atan2
#endif

#endif  //end MATH_2_SL_MATH

/********************** External Func Declaration ***************************/
//To be compatible with the old interface
#define SL_abs                SL_int_abs
#define SL_absf               SL_Float_abs
#define SL_floor              SL_Floor
#define SL_ceil               SL_Ceil
#define SL_round              SL_Round
#define SL_sqrtf              SL_Sqrt
#define SL_sqrti              SL_int_Sqrt
#define SL_sinf               SL_Sin
#define SL_cosf               SL_Cos
#define SL_ln                 SL_Ln
#define SL_log_a_b            SL_Math_Log
#define SL_exp                SL_exp_appro
#define SL_ipow               SL_Pow
#define SL_asin               SL_asin
#define SL_acos               SL_acos
#define SL_atan               SL_atan


#ifdef __cplusplus
extern "C" {
#endif

int    SL_abs( int x );
float  SL_absf( float x );
double SL_fabs( double x );
float  SL_fabsf( float x );

int    SL_floor( double x );
int    SL_ceil( double x );
int    SL_round( double x );
double SL_modf( double x, double* intpart );

double SL_sqrt( double x );
float  SL_sqrtf( float x );
int    SL_sqrti( int x );

double SL_sin( double x );
double SL_cos( double x );
float  SL_sinf( float x );
float  SL_cosf( float x );

double SL_ln( double x );
double SL_log2( double x );
double SL_log10( double x );
double SL_log_a_b( double a, double b );

double SL_exp( double x );
double SL_ipow( double a, int b );
double SL_pow( double a, double b );
float  SL_powf( double a, double b );

double SL_asin( double x );
double SL_acos( double x );
double SL_atan( double x );
double SL_atan2( double y, double x );

#ifdef __cplusplus
}
#endif

#endif //end SL_MATH_H
