#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_quality.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Conv3x3_Short* block_quality_preconv3x3_1;
    Block_Conv3x3_Short* block_quality_preconv3x3_2;
    Block_Conv3x3_Short* block_quality_dsconv3x3_1;
    Block_Conv3x3_Short* block_quality_dsconv3x3_2;
    Block_Bneck_SE_Short* block_quality_block1;
    Block_Bneck_SE_Short* block_quality_block2;
    Block_Conv1x1_Short* block_quality_reconv1x1;
    char* ver_quality;
} NetQualityShort;

static NetQualityShort get_param_quality()
{
    static char* ver_quality = para_ver_quality;
    static Block_Conv3x3_Short block_quality_preconv3x3_1 = {
        para_quality_conv3x3_pre1_weight,
        para_quality_conv3x3_pre1_bias,
        1,
        1,
        4,
        para_quality_magnification_factor,
        sizeof(para_quality_conv3x3_pre1_weight) / 2,
        sizeof(para_quality_conv3x3_pre1_bias) / 2,
    };

    static Block_Conv3x3_Short block_quality_preconv3x3_2 = {
        para_quality_conv3x3_pre2_weight,
        para_quality_conv3x3_pre2_bias,
        1,
        1,
        4,
        para_quality_magnification_factor + 2,
        sizeof(para_quality_conv3x3_pre2_weight) / 2,
        sizeof(para_quality_conv3x3_pre2_bias) / 2,
    };

    static Block_Conv3x3_Short block_quality_dsconv3x3_1 = {
        para_quality_conv3x3_ds1_weight,
        para_quality_conv3x3_ds1_bias,
        2,
        1,
        8,
        para_quality_magnification_factor + 4,
        sizeof(para_quality_conv3x3_ds1_weight) / 2,
        sizeof(para_quality_conv3x3_ds1_bias) / 2,
    };

    static Block_Conv3x3_Short block_quality_dsconv3x3_2 = {
        para_quality_conv3x3_ds2_weight,
        para_quality_conv3x3_ds2_bias,
        2,
        1,
        16,
        para_quality_magnification_factor + 6,
        sizeof(para_quality_conv3x3_ds2_weight) / 2,
        sizeof(para_quality_conv3x3_ds2_bias) / 2,
    };

    static Block_Bneck_SE_Short block_quality_block1 = {
        para_quality_conv1x1s1_di_1_weight,
        para_quality_conv1x1s1_di_1_bias,
        para_quality_convdw3x3s2_1_weight,
        para_quality_convdw3x3s2_1_bias,
        para_quality_conv1x1s1_dd_1_weight,
        para_quality_conv1x1s1_dd_1_bias,
        0,
        0,
        0,
        0,
        2,
        1,
        32,
        40,
        para_quality_magnification_factor + 8,
        sizeof(para_quality_conv1x1s1_di_1_weight) / 2,
        sizeof(para_quality_conv1x1s1_di_1_bias) / 2,
        sizeof(para_quality_convdw3x3s2_1_weight) / 2,
        sizeof(para_quality_convdw3x3s2_1_bias) / 2,
        sizeof(para_quality_conv1x1s1_dd_1_weight) / 2,
        sizeof(para_quality_conv1x1s1_dd_1_bias) / 2,
        0,
        0,
        0,
        0,
    };

    static Block_Bneck_SE_Short block_quality_block2 = {
        0,
        0,
        para_quality_convdw3x3s1_2_weight,
        para_quality_convdw3x3s1_2_bias,
        para_quality_conv1x1s1_dd_2_weight,
        para_quality_conv1x1s1_dd_2_bias,
        para_quality_conv1x1s1_dd_se_2_weight,
        para_quality_conv1x1s1_dd_se_2_bias,
        para_quality_conv1x1s1_di_se_2_weight,
        para_quality_conv1x1s1_di_se_2_bias,
        1,
        1,
        32,
        32,
        para_quality_magnification_factor + 14,
        0,
        0,
        sizeof(para_quality_convdw3x3s1_2_weight) / 2,
        sizeof(para_quality_convdw3x3s1_2_bias) / 2,
        sizeof(para_quality_conv1x1s1_dd_2_weight) / 2,
        sizeof(para_quality_conv1x1s1_dd_2_bias) / 2,
        sizeof(para_quality_conv1x1s1_dd_se_2_weight) / 2,
        sizeof(para_quality_conv1x1s1_dd_se_2_bias) / 2,
        sizeof(para_quality_conv1x1s1_di_se_2_weight) / 2,
        sizeof(para_quality_conv1x1s1_di_se_2_bias) / 2,
    };

    static Block_Conv1x1_Short block_quality_reconv1x1 = {
        para_quality_reconv1x1_weight,
        para_quality_reconv1x1_bias,
        1,
        para_quality_magnification_factor + 22,
        sizeof(para_quality_reconv1x1_weight) / 2,
        sizeof(para_quality_reconv1x1_bias) / 2,
    };

    NetQualityShort net = {
        &block_quality_preconv3x3_1,
        &block_quality_preconv3x3_2,
        &block_quality_dsconv3x3_1,
        &block_quality_dsconv3x3_2,
        &block_quality_block1,
        &block_quality_block2,
        &block_quality_reconv1x1,
        ver_quality,
    };
    return net;
};


//quality
//forward
static int net_forward_quality_short(float* src, int h, int w, NetQualityShort net, int* score)
{
    int size = h * w;
    const int LEFT_STRIDE = alignPtr(4.5 * h * w, MALLOC_ALIGN);

    //memory block
    float* memory = (float*)malloc(9 * size * sizeof(float));

    if (!memory) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    //memset(memory, 0, 10 * size * sizeof(float));

    float* tensor = (float*)alignPtr((size_t)memory, MALLOC_ALIGN); //16位对齐
    totensor_neon(src, tensor, size);
    Mat input = newMat(tensor, w, h, 1);


    //左右memory的指针
    float* right = tensor + LEFT_STRIDE;

    //pre
    input = conv3x3_block_short(input, net.block_quality_preconv3x3_1, right, 0, 2);
    input = conv3x3_block_short(input, net.block_quality_preconv3x3_2, right, 0, 2);

    //down sampling
    input = conv3x3_block_short(input, net.block_quality_dsconv3x3_1, right, 0, 2); // down sampling 1
    input = conv3x3_block_short(input, net.block_quality_dsconv3x3_2, right, 0, 2); // down sampling 2
    Mat right_cp = newMat(right, input.w, input.h, input.c);
    memcpy(right_cp.data, input.data, total(input) * sizeof(float));
    right = right + total(input);
    //input = conv3x3_block(input, net.dsconv3x3_3, right, 0); //down sampling 3
    ////input = conv3x3_block(input, net.dsconv3x3_4, right, 0); //down sampling 4

    Mat mat_row_de = newMat(right, input.w, input.h / 2, input.c);
    pooling2x1s2x1_max_neon(input, mat_row_de);
    memcpy(input.data, mat_row_de.data, total(mat_row_de) * sizeof(float));
    mat_row_de.data = input.data;
    //bottleneck
    input = bottleneck1_short(mat_row_de, net.block_quality_block1);
    input = bottleneck2_short(input, net.block_quality_block2);
    //input = bottleneck(input, net.block2);
    //input = bottleneck(input, net.block3);
    //input = bottleneck(input, net.block4);
    //input = bottleneck(input, net.block5);
    //input = bottleneck(input, net.block6);
    //input = bottleneck(input, net.block7);
    //input = bottleneck(input, net.block8);
    //input = bottleneck(input, net.block9);

    //int wc = input.w, hc = input.h;
    //up sampling
    Mat mat_avgpool = newMat(right, 1, 1, net.block_quality_block2->out_channel);
    pooling_global(input, mat_avgpool, PoolMethod_AVE);
    Mat postconv1x1 = newMat(new_memory(mat_avgpool), 1, 1, net.block_quality_reconv1x1->out_channel);
    float* weight_f = new_memory(postconv1x1);
    float* bias_f = weight_f + net.block_quality_reconv1x1->len_conv1x1_weight;
    short_to_float(weight_f, net.block_quality_reconv1x1->conv1x1_weight, net.block_quality_reconv1x1->len_conv1x1_weight, net.block_quality_reconv1x1->magnification[0]);
    short_to_float(bias_f, net.block_quality_reconv1x1->conv1x1_bias, net.block_quality_reconv1x1->len_conv1x1_bias, net.block_quality_reconv1x1->magnification[1]);
    conv1x1s1_neon(mat_avgpool, postconv1x1, weight_f, bias_f);
    hsigmoid_neon(postconv1x1);
    *score = (int)(postconv1x1.data[0] * 100 + 0.5);
    // delete[] memory;
    free(memory);
    return SL_RET_SUCCESS;
}



int net_quality(unsigned char* src, int h, int w, int* score)
{
    get_version_quality();
#if BENCHMARK
    double start = get_current_time();
#endif
    int quality_score = 0;
    int ret = 0;

    //*score = 100;
    if (h != 180 || w != 30 || !src || !score) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int hcrop = h, wcrop = w;
    float* src_f = (float*)malloc(hcrop * wcrop * sizeof(float));

    if (!src_f) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(src_f, 0, hcrop * wcrop * sizeof(float));
    int r_h = 120;
    int r_w = 20;
    float* dst = (float*)malloc(r_w * r_h * sizeof(float)); // 网络处理图像的尺寸为96x96

    if (!dst) {
        free(src_f);
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(dst, 0, r_w * r_h * sizeof(float));

    int i, j;

    for (i = 0; i < hcrop; i++) {
        for (j = 0; j < wcrop; j++) {
            src_f[i * wcrop + j] = (float)src[i * w + j];
        }
    }

    //将图像resize到96x96
    Mat input = newMat(src_f, wcrop, hcrop, 1);
    Mat output = newMat(dst, r_w, r_h, 1);

    //将图片resize到96x96
    ret = bilinear_neon_cnn(input, output, 0);

    if (ret < 0) {
        SL_LOGE("calculate_quality bilinear_neon_cnn error");
        free(src_f);
        free(dst);
        return SL_RET_FAIL;
    }

    free(src_f);
    src_f = NULL;
    ret = net_forward_quality_short(output.data, r_h, r_w, get_param_quality(), &quality_score);

    if (ret < 0) {
        SL_LOGE("calculate_quality net_forward_quality_short error");
        free(dst);
        return SL_RET_FAIL;
    }

    *score = quality_score;

    if (quality_score < 0) {
        *score = 0;
    } else if (quality_score > 100) {
        *score = 100;
    }

    free(dst);
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif

    return ret;
}