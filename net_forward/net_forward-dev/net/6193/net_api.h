#ifndef __NET_API_H__
#define __NET_API_H__

#include <stdint.h>

// #include "net_param/struct_enhance.h"
#if (defined WIN32 || defined _WIN32 || defined WINCE) //&& defined JLW_API_EXPORTS
#define JLW_EXPORTS __declspec(dllexport)
#else
#define JLW_EXPORTS
#endif
#ifdef __cplusplus
extend "C" {
#endif
    JLW_EXPORTS int32_t GetVersion( void);

    JLW_EXPORTS int desc_exp( unsigned char* src, const int h, const int w, unsigned short * dst, const int ph, const int pw );
    JLW_EXPORTS int scratch_mask(unsigned char* src, int h, int w, unsigned char* mask, int outh, int outw, int threshold, int* score);
    JLW_EXPORTS int net_enhance( unsigned char* src, const int h, const int w, unsigned char* dst, const int outh, const int outw );
    // descriptor
    JLW_EXPORTS int descriptor_net_init_patch_short( int patch_size, void** net_memory );
    JLW_EXPORTS int descriptor_net_patch_short( unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int patch_size, int sample_size, uint32_t* desc, void* net_memory );
    JLW_EXPORTS int descriptor_net_deinit_patch( void* net_memory );
    JLW_EXPORTS int descriptor_net_init_patch_short_rect(int* patch_size, void** net_memory);
    JLW_EXPORTS int descriptor_net_patch_short_rect(unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int* patch_size, int* sample_size, uint32_t* desc, void* net_memory);

    JLW_EXPORTS int spd_check(int* src, const int h, const int w, float * out);
    JLW_EXPORTS int spoof_check(unsigned char* src, const int width, const int height, int enable_learn, int enroll_flag, float * template_feature, int* score);
    JLW_EXPORTS int mistouch_check(unsigned char* img, int width, int height, int* score);
    JLW_EXPORTS int net_quality(unsigned char* img, int height, int width, int* score);

#ifdef __cplusplus
}
#endif
#endif
