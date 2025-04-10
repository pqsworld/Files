#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>
#include "stdint.h"
#include "../alog.h"
#include "net_api.h"

/*原则上保持一致
  300009300：
  30000：2023年0月0日
  93：芯片
  00：版本迭代
*/
int32_t get_version_patch(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_descriptor:  304109302");
    SL_LOGE( "net_ver_descriptor:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_enhance(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_enhance:  303219304");
    SL_LOGE( "net_ver_descriptor:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_exp(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_exp:  303179302");
    SL_LOGE( "net_ver_descriptor:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_mistouch(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_mistouch:  304273501");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_mask(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_mask:  303219305");
    SL_LOGE( "net_ver_mask:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_spoof(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_spoof:  304273501");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_spd(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_spd:  302149301");
    SL_LOGE( "net_ver_spd:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}

int32_t get_version_quality(void)
{
#if NOT_WINSO
    //SL_LOGE( "net_ver_quality:  303239301");
    SL_LOGE( "net_ver_quality:  NOT_SUPPORT");
#endif
    return SL_RET_SUCCESS;
}

int32_t GetVersion(void)
{
    int32_t net_ver = 304273501;
    //N代表正式版本号，T代表测试版本号
    SL_LOGE( "net_ver:  N.304273501, build_date:%s, build_time:%s", __DATE__, __TIME__);
    //get_version_patch();
    //get_version_enhance();
    //get_version_exp();
    get_version_mistouch();
    //get_version_mask();
    get_version_spoof();
    //get_version_spd();
    //get_version_quality();
    return net_ver;
}

/*
#include "stdio.h"
#include <stdlib.h>
#include <stdarg.h>
#include <utils/Log.h>
//#include <windows.h>

#define LOG_TAG "algo"
#define LOG_NDEBUG 0


void SL_Log(const char *str, ...)
{
    const char* filename = ".\\Log\\SileadLog.txt";
    va_list ap;
    FILE *fp = NULL;

#ifdef __STDC_WANT_SECURE_LIB__
    if (fopen_s(&fp, filename, "at") == 0)
#else
    if (fp = fopen(filename, "at"))
#endif

    {
        va_start(ap, str);
        vfprintf(fp, str, ap);
//      fprintf(fp, "");
        va_end(ap);
        fclose(fp);
        fp = NULL;
    }
}

void SL_LOGE(const char *str, ...)
{
    //SYSTEMTIME currentTime;
    //GetSystemTime(&currentTime);
    //SL_Log("%02u:%02u:%02u.%03u\t", currentTime.wHour + 8, currentTime.wMinute, currentTime.wSecond,
    //  currentTime.wMilliseconds);

    const char* filename = ".\\Log\\SileadLog.txt";
    va_list ap;
    FILE *fp = NULL;

#ifdef __STDC_WANT_SECURE_LIB__
    if (fopen_s(&fp, filename, "at") == 0)
#else
    if (fp = fopen(filename, "at"))
#endif

    {
        va_start(ap, str);
        vfprintf(fp, str, ap);
        //      fprintf(fp, "");
        va_end(ap);
        fclose(fp);
        fp = NULL;
    }
}

void SL_LOGE(const char *str, ...)
{
    //SYSTEMTIME currentTime;
    //GetSystemTime(&currentTime);
    //SL_Log("%02u:%02u:%02u.%03u\t", currentTime.wHour + 8, currentTime.wMinute, currentTime.wSecond,
    //  currentTime.wMilliseconds);

    const char* filename = ".\\Log\\SileadLog.txt";
    va_list ap;
    FILE *fp = NULL;

#ifdef __STDC_WANT_SECURE_LIB__
    if (fopen_s(&fp, filename, "at") == 0)
#else
    if (fp = fopen(filename, "at"))
#endif

    {
        va_start(ap, str);
        vfprintf(fp, str, ap);
        //      fprintf(fp, "");
        va_end(ap);
        fclose(fp);
        fp = NULL;
    }
}*/
