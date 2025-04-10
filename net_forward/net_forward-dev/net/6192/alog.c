#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>
#include "stdint.h"
#include "../alog.h"

/*原则上保持一致
  300009300：
  30000：2023年0月0日
  93：芯片
  00：版本迭代
*/
int32_t get_version_patch(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_descriptor: 306139202 ");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_enhance(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_enhance:  303179201");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_exp(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_exp:  303179201");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_mistouch(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_mistouch:  304279201");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_spoof(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_spoof:  304279201");
#endif
    return SL_RET_SUCCESS;
}

int32_t get_version_spd(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_spd:  305089201");
#endif
    return SL_RET_SUCCESS;
}
int32_t get_version_quality(void)
{
#if NOT_WINSO
    SL_LOGE( "net_ver_quality:  305089201");
#endif
    return SL_RET_SUCCESS;
}

int32_t GetVersion(void)
{
    int32_t net_ver = 306139206;
    //N代表正式版本号，T代表测试版本号
    SL_LOGE( "net_ver:  N.306139206, build_date:%s, build_time:%s", __DATE__, __TIME__);
    get_version_enhance();
    get_version_exp();
    get_version_mistouch();
    get_version_spoof();
    get_version_spd();
    get_version_quality();
    get_version_patch();
    return net_ver;
}
