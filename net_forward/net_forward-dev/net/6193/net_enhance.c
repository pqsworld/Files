#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_enhance.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Conv3x3_Short* block_enhance_features_deconv1;
    Block_Conv3x3_Short* block_enhance_features_deconv2_1;
    Block_Conv3x3_Short* block_enhance_features_deconv2_2;
    Block_Conv3x3_Short* block_enhance_features_deconv3;
    Block_Sep_Short* block_enhance_features_deconv4_1;
    Block_Sep_Short* block_enhance_features_deconv4_2;
    Block_Resnet_Short* block_enhance_features_resnet1;
    Block_Conv3x3_Short* block_enhance_features_upconv3_3x3;
    Block_Conv1x1_Short* block_enhance_features_upconv3_1x1;
    Block_Conv3x3_Short* block_enhance_features_upconv3;
    Block_Conv3x3_Short* block_enhance_features_upconv2;
    Block_Conv3x3_Short* block_enhance_features_upconv2_1;
    Block_Conv3x3_Short* block_enhance_features_upconv2_2;
    Block_Conv3x3_Short* block_enhance_features_upconv1;
    Block_Conv1x1_Short* block_enhance_features_upconv1_1x1;
    Block_Conv3x3_Short* block_enhance_features_upconv0;
    char* ver_enhance;
} NetEnhanceShort;

NetEnhanceShort get_param_enhance()
{
    static char* ver_enhance = para_ver_enhance;

    static Block_Conv3x3_Short block_enhance_features_deconv1 = {
        para_enhance_features_deconv1_weight,
        para_enhance_features_deconv1_bias,
        2, // stride
        1, // padding
        4, // outchannel
        para_enhance_maxcoe,
        sizeof(para_enhance_features_deconv1_weight) / 2,
        sizeof(para_enhance_features_deconv1_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_deconv2_1 = {
        para_enhance_features_deconv2_1_weight,
        para_enhance_features_deconv2_1_bias,
        1,
        1,
        4,
        para_enhance_maxcoe + 2,
        sizeof(para_enhance_features_deconv2_1_weight) / 2,
        sizeof(para_enhance_features_deconv2_1_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_deconv2_2 = {
        para_enhance_features_deconv2_2_weight,
        para_enhance_features_deconv2_2_bias,
        1,
        1,
        4,
        para_enhance_maxcoe + 4,
        sizeof(para_enhance_features_deconv2_2_weight) / 2,
        sizeof(para_enhance_features_deconv2_2_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_deconv3 = {
        para_enhance_features_deconv3_weight,
        para_enhance_features_deconv3_bias,
        2,
        1,
        16,
        para_enhance_maxcoe + 6,
        sizeof(para_enhance_features_deconv3_weight) / 2,
        sizeof(para_enhance_features_deconv3_bias) / 2,
    };
    static Block_Sep_Short block_enhance_features_deconv4_1 = {
        para_enhance_features_deconv4_1_weight,
        para_enhance_features_deconv4_1_bias,
        para_enhance_features_deconv4_11_weight,
        para_enhance_features_deconv4_11_bias,
        1,
        1,
        16,
        16,
        para_enhance_maxcoe + 8,
        sizeof(para_enhance_features_deconv4_1_weight) / 2,
        sizeof(para_enhance_features_deconv4_1_bias) / 2,
        sizeof(para_enhance_features_deconv4_11_weight) / 2,
        sizeof(para_enhance_features_deconv4_11_bias) / 2,
    };

    static Block_Sep_Short block_enhance_features_deconv4_2 = {
        para_enhance_features_deconv4_2_weight,
        para_enhance_features_deconv4_2_bias,
        para_enhance_features_deconv4_12_weight,
        para_enhance_features_deconv4_12_bias,
        1,
        1,
        16,
        16,
        para_enhance_maxcoe + 12,
        sizeof(para_enhance_features_deconv4_2_weight) / 2,
        sizeof(para_enhance_features_deconv4_2_bias) / 2,
        sizeof(para_enhance_features_deconv4_12_weight) / 2,
        sizeof(para_enhance_features_deconv4_12_bias) / 2,
    };

    static Block_Resnet_Short block_enhance_features_resnet1 = {
        para_enhance_features_resnet1_1_weight,
        para_enhance_features_resnet1_1_bias,
        para_enhance_features_resnet1_2_weight,
        para_enhance_features_resnet1_2_bias,
        para_enhance_features_resnet1_3_weight,
        para_enhance_features_resnet1_3_bias,
        para_enhance_features_resnet1_4_weight,
        para_enhance_features_resnet1_4_bias,
        1,
        1,
        32,
        32,
        para_enhance_maxcoe + 16,
        sizeof(para_enhance_features_resnet1_1_weight) / 2,
        sizeof(para_enhance_features_resnet1_1_bias) / 2,
        sizeof(para_enhance_features_resnet1_2_weight) / 2,
        sizeof(para_enhance_features_resnet1_2_bias) / 2,
        sizeof(para_enhance_features_resnet1_3_weight) / 2,
        sizeof(para_enhance_features_resnet1_3_bias) / 2,
        sizeof(para_enhance_features_resnet1_4_weight) / 2,
        sizeof(para_enhance_features_resnet1_4_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv3_3x3 = {
        para_enhance_features_upconv3_3x3_weight,
        para_enhance_features_upconv3_3x3_bias,
        1,
        1,
        32,
        para_enhance_maxcoe + 24,
        sizeof(para_enhance_features_upconv3_3x3_weight) / 2,
        sizeof(para_enhance_features_upconv3_3x3_bias) / 2,
    };
    static Block_Conv1x1_Short block_enhance_features_upconv3_1x1 = {
        para_enhance_features_upconv3_1x1_weight,
        para_enhance_features_upconv3_1x1_bias,
        16,
        para_enhance_maxcoe + 26,
        sizeof(para_enhance_features_upconv3_1x1_weight) / 2,
        sizeof(para_enhance_features_upconv3_1x1_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv3 = {
        para_enhance_features_upconv3_3x3_1_weight,
        para_enhance_features_upconv3_3x3_1_bias,
        1,
        1,
        16,
        para_enhance_maxcoe + 28,
        sizeof(para_enhance_features_upconv3_3x3_1_weight) / 2,
        sizeof(para_enhance_features_upconv3_3x3_1_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv2 = {
        para_enhance_features_upconv2_weight,
        para_enhance_features_upconv2_bias,
        1,
        1,
        8,
        para_enhance_maxcoe + 30,
        sizeof(para_enhance_features_upconv2_weight) / 2,
        sizeof(para_enhance_features_upconv2_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv2_1 = {
        para_enhance_features_upconv2_1_weight,
        para_enhance_features_upconv2_1_bias,
        1,
        1,
        4,
        para_enhance_maxcoe + 32,
        sizeof(para_enhance_features_upconv2_1_weight) / 2,
        sizeof(para_enhance_features_upconv2_1_bias) / 2,
    };
    static Block_Conv3x3_Short block_enhance_features_upconv2_2 = {
        para_enhance_features_upconv2_2_weight,
        para_enhance_features_upconv2_2_bias,
        1,
        1,
        4,
        para_enhance_maxcoe + 34,
        sizeof(para_enhance_features_upconv2_2_weight) / 2,
        sizeof(para_enhance_features_upconv2_2_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv1 = {
        para_enhance_features_upconv1_3x3_weight,
        para_enhance_features_upconv1_3x3_bias,
        1,
        1,
        8,
        para_enhance_maxcoe + 36,
        sizeof(para_enhance_features_upconv1_3x3_weight) / 2,
        sizeof(para_enhance_features_upconv1_3x3_bias) / 2,
    };
    static Block_Conv1x1_Short block_enhance_features_upconv1_1x1 = {
        para_enhance_features_upconv1_1x1_weight,
        para_enhance_features_upconv1_1x1_bias,
        4,
        para_enhance_maxcoe + 38,
        sizeof(para_enhance_features_upconv1_1x1_weight) / 2,
        sizeof(para_enhance_features_upconv1_1x1_bias) / 2,
    };

    static Block_Conv3x3_Short block_enhance_features_upconv0 = {
        para_enhance_features_upconv0_weight,
        para_enhance_features_upconv0_bias,
        2,
        1,
        1,
        para_enhance_maxcoe + 40,
        sizeof(para_enhance_features_upconv0_weight) / 2,
        sizeof(para_enhance_features_upconv0_bias) / 2,
    };

    NetEnhanceShort net = {
        &block_enhance_features_deconv1,
        &block_enhance_features_deconv2_1,
        &block_enhance_features_deconv2_2,
        &block_enhance_features_deconv3,
        &block_enhance_features_deconv4_1,
        &block_enhance_features_deconv4_2,
        &block_enhance_features_resnet1,
        &block_enhance_features_upconv3_3x3,
        &block_enhance_features_upconv3_1x1,
        &block_enhance_features_upconv3,
        &block_enhance_features_upconv2,
        &block_enhance_features_upconv2_1,
        &block_enhance_features_upconv2_2,
        &block_enhance_features_upconv1,
        &block_enhance_features_upconv1_1x1,
        &block_enhance_features_upconv0,
        ver_enhance,
    };
    return net;
};




static int pix2pix_resnetblockdwshort(float* src, short h, short w, NetEnhanceShort net)
{
    int hc = h, wc = w;
    int ori_size = hc * wc;
    //int j = 0;

    const int LEFT_STRIDE = alignPtr(ori_size, MALLOC_ALIGN);
    int size = 12 * ori_size;

    float* memory = (float*)sl_aligned_malloc(size * sizeof(float), MALLOC_ALIGN);

    if (!memory) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(memory, 0, size * sizeof(float));

    float* img_diff = (float*)alignPtr((size_t)memory, MALLOC_ALIGN);
    //normalization
    normalize_neon(src, img_diff, 127.5f, 127.5f, ori_size);

    Mat input0 = newMat(img_diff, wc, hc, 1);

    conv_block_s(&input0, net.block_enhance_features_deconv1, 3, 0);

    leakyrelu_neon(input0, 0.2);
    conv_block_s(&input0, net.block_enhance_features_deconv2_1, 3, 0);
    float* leftp = new_memory(input0);
    Mat deconv2_2 = newMat(leftp, input0.w, input0.h, input0.c);
    memset(leftp, 0, total(deconv2_2) * sizeof(float));
    copy(input0, deconv2_2);
    leakyrelu_neon(deconv2_2, 0.2f);

    conv_block_s(&deconv2_2, net.block_enhance_features_deconv2_2, 3, 0);

    Mat inputmp = newMat(img_diff, deconv2_2.w, deconv2_2.h, deconv2_2.c * 2);

    conv_block_s(&inputmp, net.block_enhance_features_deconv3, 3, 1);

    leakyrelu_neon(inputmp, 0.2);

    conv_sep_block_s(&inputmp, net.block_enhance_features_deconv4_1);
    float* leftp2 = new_memory(inputmp);
    Mat deconv4_2 = newMat(leftp2, inputmp.w, inputmp.h, inputmp.c);
    memset(leftp2, 0, total(deconv4_2) * sizeof(float));
    copy(inputmp, deconv4_2);
    leakyrelu_neon(deconv4_2, 0.2);
    conv_sep_block_s(&deconv4_2, net.block_enhance_features_deconv4_2);

    Mat inputmp0 = newMat(img_diff, inputmp.w, inputmp.h, inputmp.c * 2);


    float* right = new_memory(inputmp0) + 4 * LEFT_STRIDE;
    resnet3x3_blockdw_s(&inputmp0, net.block_enhance_features_resnet1);

    relu_neon(inputmp0);

    conv_group_single_block_s(&inputmp0, net.block_enhance_features_upconv3_3x3, 3, 4);
    conv1x1_blocktool_s(&inputmp0, net.block_enhance_features_upconv3_1x1, right + LEFT_STRIDE);

    float* tmright = inputmp0.data + inputmp0.cstep * inputmp0.c * 2;
    Mat out_up2 = newMat(tmright, inputmp0.w * 2, inputmp0.h * 2, inputmp.c);
    memset(tmright, 0, total(out_up2) * sizeof(float));
    bilinear_neon_cnn(inputmp0, out_up2, 0);

    conv_block_s(&out_up2, net.block_enhance_features_upconv3, 3, 0);

    relu_neon(out_up2);
    conv_block_s(&out_up2, net.block_enhance_features_upconv2, 3, 0);

    relu_neon(out_up2);

    conv_block_s(&out_up2, net.block_enhance_features_upconv2_1, 3, 0);

    Mat upconv2_2 = newMat(new_memory(out_up2), out_up2.w, out_up2.h, out_up2.c);
    memset(new_memory(out_up2), 0, total(upconv2_2) * sizeof(float));
    copy(out_up2, upconv2_2);
    relu_neon(upconv2_2);
    conv_block_s(&upconv2_2, net.block_enhance_features_upconv2_2, 3, 0);

    Mat upconv2 = newMat(tmright, upconv2_2.w, upconv2_2.h, upconv2_2.c * 2);
    relu_neon(upconv2);
    conv_group_single_block_s(&upconv2, net.block_enhance_features_upconv1, 3, 4);
    conv1x1_blocktool_s(&upconv2, net.block_enhance_features_upconv1_1x1, right + 2 * LEFT_STRIDE);

    deconv4x4_block_s(&upconv2, net.block_enhance_features_upconv0);
    tanh_neon(upconv2);

    int i = 0;

    for (i = 0; i < ori_size; i++) {
        float tmpf = (upconv2.data[i] + 1) * 127.5f;
        tmpf = (float)(tmpf < 0 ? 0 : (tmpf > 255 ? 255 : tmpf));
        // short tmp = (short)(tmpf < 0 ? 0 : (tmpf > 255 ? 255 : tmpf));
        short tmp = (tmpf >= 0.0) ? (int)(tmpf + 0.5) : (int)(tmpf - 0.5);

        src[i] = tmp;
    }

    if (memory) {
        sl_aligned_free(memory);
    }

    return SL_RET_SUCCESS;
}

int net_enhance(unsigned char* src, const int h, const int w, unsigned char* dst, const int outh, const int outw)
{
#if BENCHMARK
    double start = get_current_time();
#endif
    get_version_enhance();

    if (h <= 0 || w <= 0 || !src  || !dst) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    /*int oh = h + 2;
    int ow = w;*/
    int pxx = 2;
    int pyy = 1;
    int oh = h + 2 * pyy;
    int ow = w + 2 * pxx;
    int ori_size = oh * ow;
    float* dst_f = (float*)malloc(ori_size * sizeof(float));

    if (!dst_f) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(dst_f, 0, ori_size * sizeof(float));

    int i = 0, j = 0;

    // 上下填充255
    for (i = 0; i < ori_size; i++) {
        dst_f[i] = 255;
    }

    //for (i = 0; i < h; i++) {
    //    for (j = 0; j < w; j++) {
    //        dst_f[(i + 3) * ow + j + 4] = (float)src[i * w + j];// 1 2
    //    }
    //}

    //外侧2丢掉
    for (i = 2; i < h - 2; i++) {
        for (j = 2; j < w - 2; j++) {
            dst_f[(i + pyy) * ow + j + pxx] = (float)src[i * w + j];
        }
    }

    int ret = 0;
    ret = pix2pix_resnetblockdwshort(dst_f, oh, ow, get_param_enhance());

    if (ret < 0) {
        free(dst_f);
        return SL_RET_FAIL;
    }

    // for (j = 0; j <w*h; j++)
    //{
    //    unsigned char tmp = dst_f[j+w];
    //    if (tmp < 0) tmp = 0;
    //    if (tmp > 255) tmp = 255;
    //    dst[j] = tmp;
    // }

    for (i = 0; i < outh; i++) {
        for (j = 0; j < outw; j++) {
            unsigned char tmp = dst_f[(i + pyy) * ow + j + pxx];
            //255>tmp>0 for uchar

            dst[i * outw + j] = tmp;
        }
    }

    free(dst_f);
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}
