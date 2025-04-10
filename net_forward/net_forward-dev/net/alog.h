#ifndef LOG_FILE_H
#define LOG_FILE_H

#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>
#include "stdint.h"
#ifndef UNUSED
#define UNUSED(v) (void)(v)
#endif

#if(!RUN_TST)
#define RUN_TST 0
#endif

#if(!BENCHMARK)
#define BENCHMARK 0
#endif

#ifdef MAKE_SO             //so库首先定义，可被覆盖
#include <utils/Log.h>
#include"android.h"
#define SL_LOGI  sil_logi
#define SL_LOGD  sil_logd
#define SL_LOGE  sil_loge
#endif

#ifdef SL_LOGI             //清除宏环境
#undef SL_LOGI
#undef SL_LOGD
#undef SL_LOGE
#endif

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined _WINDOWS)
#define FLAG_WIN 1
#else
#define FLAG_WIN 0
#endif


#if (FLAG_WIN||RUN_TST) //WIN环境||TST环境

#define SL_LOGI printf
#define SL_LOGD printf
#define SL_LOGE printf

#else
extern void sil_loge( const char* fmt, ... );
extern void sil_logd( const char* fmt, ... );
extern void sil_logi( const char* fmt, ... );

#define SL_LOGD sil_logd
#define SL_LOGE sil_loge

#ifdef CUST_OPPO
#define SL_LOGI sil_logi
#else
#define SL_LOGI sil_loge
#endif

#endif
//#define malloc  sil_malloc
//#define free    sil_free

// int abs(int j);

#if (FLAG_WIN || defined MAKE_SO)
#define NOT_WINSO 0
#else
#define NOT_WINSO 1
#endif

#define SL_ERR_PARAM                -101
#define SL_ERR_PARAM_LOG SL_LOGE("%d,%s,param fail",__LINE__,__FUNCTION__)
#define SL_ERR_MALLOC               -102
#define SL_ERR_MALLOC_LOG SL_LOGE("%d,%s,malloc fail",__LINE__,__FUNCTION__)
#define SL_ERR_POINTER_NULL         -112
#define SL_ERR_POINTER_NULL_LOG SL_LOGE("%d,%s,pointer null fail",__LINE__,__FUNCTION__)
#define SL_RET_SUCCESS                 0
#define SL_RET_FAIL                   -1

int32_t get_version_enhance(void);
int32_t get_version_exp(void);
int32_t get_version_mistouch(void);
int32_t get_version_spoof(void);
int32_t get_version_spd(void);
int32_t get_version_mask(void);
int32_t get_version_patch(void);
int32_t get_version_quality(void);
int32_t GetVersion(void);

#endif
