/******************************************************************************
 * Copyright (c) 2018-2019 Silead Inc.
 * All rights reserved
 *
 * File:         SL_Math.c
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

#include "SL_Math.h"
// #include "alog.h"
#if defined(WIN32) || defined(MAKE_SO) || defined(RUN_TST) // #ifdef MAKE_SO
/*********************** Static Func Declaration ******************************/

static int equal(double a, double b);
static double SL_log(double x);
static double F1(double x);
static double simpson(double a, double b);
static double asr(double a, double b, double eps, double A);
static double _asr(double a, double b, double eps);
static double SL_atanh(double x);

/**************************** Func Definition *********************************/

static int equal(double a, double b)
{
    return (a - b < 1e-7) && (a - b > -1e-7);
}

int SL_abs(int x)
{
    return ABS(x);
}

float SL_absf(float x)
{
    return ABS(x);
}

double SL_fabs(double x)
{
    return ABS(x);
}

float SL_fabsf(float x)
{
    return ABS(x);
}

int SL_floor(double x)
{
    return (x < 0) && ((float)((int)(x) - x) != 0.0f) ? (int)(x - 1) : (int)x;
}

int SL_ceil(double x)
{
    return (x > 0) && ((float)((int)(x) - x) != 0.0f) ? (int)(x + 1) : (int)x;
}

int SL_round(double x)
{
    return (x >= 0.0) ? (int)(x + 0.5) : (int)(x - 0.5);
}

double SL_modf(double x, double* intpart)
{
    *intpart = (double)SL_round(x);
    return (x - *intpart);
}

double SL_sqrt(double x)
{
    if (x < 0) {
        return NAN;
    }

    double xhalf = 0.5 * x;
    long long i = *(long long*)&x;
    i = 0x5fe6ec85e7de30da - (i >> 1);
    x = *(double*)&i;
    x = x * (1.5 - xhalf * x * x);
    x = x * (1.5 - xhalf * x * x);
    x = x * (1.5 - xhalf * x * x);
    x = x * (1.5 - xhalf * x * x);
    return 1 / x;
}

float SL_sqrtf(float x)
{
    if (x < 0) {
        return NAN;
    }

    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f375a86 - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    x = x * (1.5f - xhalf * x * x);
    x = x * (1.5f - xhalf * x * x);
    return 1 / x;
}

int SL_sqrti(int x)
{
    if (x < 0) {
        return -1;
    }

    return (int)SL_sqrtf((float)x);
}

double SL_sin(double x)
{
    // 定义域(-∞, +∞) →[-π/2, +π/2]
    if (x > SL_PI_DIV_2) {
        x = SL_PI - (x - SL_2PI * (int)(x / SL_2PI));
    } else if (x < -SL_PI_DIV_2) {
        x = -(SL_PI + (x - SL_2PI * (int)(x / SL_2PI)));
    }

    // 幂级数求近似解
    int i = 1;
    double res = x, term = x, xx_n = -x * x;

    do {
        i += 2;
        term *= xx_n / (i * i - i);
        res += term;
    } while (SL_fabs(term) >= 1e-7);

    return res;
}

double SL_cos(double x)
{
    return SL_sin(SL_PI_DIV_2 - x);
}

float SL_sinf(float x)
{
    return (float)SL_sin(x);
}

float SL_cosf(float x)
{
    return (float)SL_cos(x);
}

static double SL_log(double x)
{
    if (x < 0) {
        return NAN;
    }

    if (equal(x, 0.0)) {
        return INF_N;
    }

    double cnt1 = 0.0;
    double cnt2 = 0.0;

    if (x > 10.0)
        while (x > 10.0) {
            x /= 10.0;
            cnt1++;
        } else if (x < 0.1)
        while (x < 0.1) {
            x *= 10;
            cnt2--;
        }

    return cnt1 * SL_LN10 + cnt2 * SL_LN10 + 2 * SL_atanh((x - 1.0) / (x + 1.0));
}

double SL_ln(double x)
{
    return SL_log(x);
}

double SL_log2(double x)
{
    return SL_ln(x) / SL_ln(2.0);
}

double SL_log10(double x)
{
    return SL_ln(x) / SL_ln(10.0);
}

double SL_log_a_b(double a, double b)
{
    if (a < 0 || b < 0) {
        return NAN;
    }

    if (equal(b, 0.0)) {
        return INF_N;
    }

    return SL_ln(b) / SL_ln(a);
}
#if 1
double SL_exp(double x) // old implementation
{
    int i = 0;
    x = 1.0 + x / 131072;

    for (i = 0; i < 17; i++) {
        x *= x;
    }

    return x;
}
#else
double SL_exp(double x)
{
    long long k = SL_floor(x * SL_LOG2E);
    double r = x - k * SL_LN2;
    long long offset = 0x1L;
    int sgn = k >= 0;

    for (k = ABS(k); k > 31; k -= 31) {
        offset <<= 31;
    }

    offset <<= k;
    int i = 0;
    double res = 1, term = 1;

    do {
        i++;
        term *= r / i;
        res += term;
    } while (SL_fabs(term) >= 1e-14);

    return (sgn) ? (res * offset) : (res / offset);
}
#endif

double SL_ipow(double a, int b)
{
    if (equal(a, 0.0) && !(b == 0)) {
        return 0.0;
    }

    if ((b == 0) || equal(a, 1.0)) {
        return 1.0;
    }

    if (b < 0) {
        return 1 / SL_ipow(a, -b);
    }

    double res = 1.0;

    while (b) {
        if (b & 1) {
            res *= a;
        }

        a *= a;
        b >>= 1;
    }

    return res;
}

double SL_pow(double a, double b)
{
    if (equal(a, 0.0) && !equal(b, 0.0)) {
        return 0.0;
    }

    if (equal(b, 0.0) || equal(a, 1.0)) {
        return 1.0;
    }

    if (equal((int)b, b)) {
        return SL_ipow(a, (int)b);
    }

    return SL_exp(b * SL_ln(a));
}

float SL_powf(double a, double b)
{
    return (float)SL_pow(a, b);
}

static double F1(double x)
{
    return 1 / (double)SL_sqrt(1.0 - (x * x));
}

static double simpson(double a, double b)
{
    double c = a + (b - a) / 2;
    return (F1(a) + 4 * F1(c) + F1(b)) * (b - a) / 6;
    return -1;
}

static double asr(double a, double b, double eps, double A)
{
    double c = a + (b - a) / 2;
    double L = simpson(a, c), R = simpson(c, b);

    if (SL_fabs((float)(L + R - A)) <= 15 * eps) {
        return L + R + (L + R - A) / 15.0;
    }

    return asr(a, c, eps / 2, L) + asr(c, b, eps / 2, R);
}

static double _asr(double a, double b, double eps)
{
    return asr(a, b, eps, simpson(a, b));
}

double SL_asin(double x)
{
    if (SL_fabs(x) > 1) {
        return NAN;
    }

    double fl = 1.0;

    if (x < 0) {
        fl *= -1;
        x *= -1;
    }

    if (equal(x, 1)) {
        return SL_PI / 2;
    }

    return (fl * _asr(0, x, 1e-8));
}

double SL_acos(double x)
{
    if (SL_fabs(x) > 1) {
        return NAN;
    }

    return SL_PI / 2 - SL_asin(x);
}

double SL_atan(double x)
{
    if (x < 0) {
        return -SL_atan(-x);
    }

    if (x > 1) {
        return SL_PI / 2 - SL_atan(1 / x);
    }

    if (x > 0.5) {
        return 2 * SL_atan((SL_sqrt((double)(1 + x * x)) - 1) / x);
    }

    int i = 1;
    double res = x, term = x, xx_n = -x * x;

    do {
        i += 2;
        term *= xx_n;
        res += term / i;
    } while (SL_fabs(term / i) >= 1e-14);

    return res;
}

double SL_atan2(double y, double x)
{
    if (equal(x, 0.0) && equal(y, 0.0)) {
        return NAN;
    }

    if (equal(x, 0.0)) {
        return y > 0.0 ? SL_PI_DIV_2 : -SL_PI_DIV_2;
    }

    if (equal(y, 0.0)) {
        return x > 0.0 ? 0.0 : SL_PI;
    }

    return x > 0.0 ? SL_atan(y / x) : SL_atan(y / x) + SL_PI;
}

static double SL_atanh(double x)
{
    if (x < -1) {
        return INF_N;
    }

    if (x > 1) {
        return INF_P;
    }

    int i = 1;
    double res = x, term = x, xx = x * x;

    do {
        i += 2;
        term *= xx * (i - 2) / i;
        res += term;
    } while (SL_fabs(term) >= 1e-14);

    return res;
}

/******************************* Test ***********************************/

#if SL_MATH_TEST

int main()
{
    double x[12] = {0.00027654, 0.00745329, 0.02348719, 0.729635, 6.257912, 53.142678, 139.81298, 527.298769, 2739.21963, 32854.347902, 2783456.10937, 27395628.9201};
    double y[12] = {0.0};

    int i = 0, loop = 0;

    double my_time = 0.0;
    clock_t start, end;
    start = clock();

    for (loop = 0; loop < 1000000; loop++) {
        for (i = 0; i < 12; i++) {
            y[i] = SL_sinf(x[i]);
        }
    }

    end = clock();
    my_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f", my_time);
    return 0;
}
#endif
#endif
