#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_spd.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Conv3x3_Short* preconv3x3s2;
    Block_Bneck_SE_Short* block1;
    Block_Bneck_SE_Short* block2;
    Block_Bneck_SE_Short* block3;
    Block_Bneck_SE_Short* block4;
    Block_Bneck_SE_Short* block5;
    Block_Bneck_SE_Short* block6;
    Block_Conv1x1_Short* postconv1x1;
    char* ver_spd;
} NetSpdShort;

// #if 1
static NetSpdShort get_para_spd_v1()
{
    static char* ver_spd = para_ver_spd;

    static Block_Conv3x3_Short preconv3x3s2 = {
        conv3x3_weight,
        conv3x3_bias,
        2,
        3,
        8,
        magnification_factor,
        sizeof(conv3x3_weight) / 2,
        sizeof(conv3x3_bias) / 2,
    };

    static Block_Bneck_SE_Short block1 = {
        conv1x1s1_di_1_weight,
        conv1x1s1_di_1_bias,
        convdw3x3s2_1_weight,
        convdw3x3s2_1_bias,
        conv1x1s1_dd_1_weight,
        conv1x1s1_dd_1_bias,
        conv1x1s1_dd_se_1_weight,
        conv1x1s1_dd_se_1_bias,
        conv1x1s1_di_se_1_weight,
        conv1x1s1_di_se_1_bias,
        2,
        1,
        16,
        4,
        magnification_factor + 2,
        sizeof(conv1x1s1_di_1_weight) / 2,
        sizeof(conv1x1s1_di_1_bias) / 2,
        sizeof(convdw3x3s2_1_weight) / 2,
        sizeof(convdw3x3s2_1_bias) / 2,
        sizeof(conv1x1s1_dd_1_weight) / 2,
        sizeof(conv1x1s1_dd_1_bias) / 2,
        sizeof(conv1x1s1_dd_se_1_weight) / 2,
        sizeof(conv1x1s1_dd_se_1_bias) / 2,
        sizeof(conv1x1s1_di_se_1_weight) / 2,
        sizeof(conv1x1s1_di_se_1_bias) / 2,
    };

    static Block_Bneck_SE_Short block2 = {
        conv1x1s1_di_2_weight,
        conv1x1s1_di_2_bias,
        convdw3x3s1_2_weight,
        convdw3x3s1_2_bias,
        conv1x1s1_dd_2_weight,
        conv1x1s1_dd_2_bias,
        conv1x1s1_dd_se_2_weight,
        conv1x1s1_dd_se_2_bias,
        conv1x1s1_di_se_2_weight,
        conv1x1s1_di_se_2_bias,
        1,
        1,
        16,
        4,
        magnification_factor + 12,
        sizeof(conv1x1s1_di_2_weight) / 2,
        sizeof(conv1x1s1_di_2_bias) / 2,
        sizeof(convdw3x3s1_2_weight) / 2,
        sizeof(convdw3x3s1_2_bias) / 2,
        sizeof(conv1x1s1_dd_2_weight) / 2,
        sizeof(conv1x1s1_dd_2_bias) / 2,
        sizeof(conv1x1s1_dd_se_2_weight) / 2,
        sizeof(conv1x1s1_dd_se_2_bias) / 2,
        sizeof(conv1x1s1_di_se_2_weight) / 2,
        sizeof(conv1x1s1_di_se_2_bias) / 2,
    };

    static Block_Bneck_SE_Short block3 = {
        conv1x1s1_di_3_weight,
        conv1x1s1_di_3_bias,
        convdw3x3s2_3_weight,
        convdw3x3s2_3_bias,
        conv1x1s1_dd_3_weight,
        conv1x1s1_dd_3_bias,
        conv1x1s1_dd_se_3_weight,
        conv1x1s1_dd_se_3_bias,
        conv1x1s1_di_se_3_weight,
        conv1x1s1_di_se_3_bias,
        2,
        1,
        20,
        4,
        magnification_factor + 22,
        sizeof(conv1x1s1_di_3_weight) / 2,
        sizeof(conv1x1s1_di_3_bias) / 2,
        sizeof(convdw3x3s2_3_weight) / 2,
        sizeof(convdw3x3s2_3_bias) / 2,
        sizeof(conv1x1s1_dd_3_weight) / 2,
        sizeof(conv1x1s1_dd_3_bias) / 2,
        sizeof(conv1x1s1_dd_se_3_weight) / 2,
        sizeof(conv1x1s1_dd_se_3_bias) / 2,
        sizeof(conv1x1s1_di_se_3_weight) / 2,
        sizeof(conv1x1s1_di_se_3_bias) / 2,
    };

    static Block_Bneck_SE_Short block4 = {
        conv1x1s1_di_4_weight,
        conv1x1s1_di_4_bias,
        convdw3x3s1_4_weight,
        convdw3x3s1_4_bias,
        conv1x1s1_dd_4_weight,
        conv1x1s1_dd_4_bias,
        conv1x1s1_dd_se_4_weight,
        conv1x1s1_dd_se_4_bias,
        conv1x1s1_di_se_4_weight,
        conv1x1s1_di_se_4_bias,
        1,
        1,
        20,
        4,
        magnification_factor + 32,
        sizeof(conv1x1s1_di_4_weight) / 2,
        sizeof(conv1x1s1_di_4_bias) / 2,
        sizeof(convdw3x3s1_4_weight) / 2,
        sizeof(convdw3x3s1_4_bias) / 2,
        sizeof(conv1x1s1_dd_4_weight) / 2,
        sizeof(conv1x1s1_dd_4_bias) / 2,
        sizeof(conv1x1s1_dd_se_4_weight) / 2,
        sizeof(conv1x1s1_dd_se_4_bias) / 2,
        sizeof(conv1x1s1_di_se_4_weight) / 2,
        sizeof(conv1x1s1_di_se_4_bias) / 2,
    };

    static Block_Bneck_SE_Short block5 = {
        conv1x1s1_di_5_weight,
        conv1x1s1_di_5_bias,
        convdw3x3s2_5_weight,
        convdw3x3s2_5_bias,
        conv1x1s1_dd_5_weight,
        conv1x1s1_dd_5_bias,
        conv1x1s1_dd_se_5_weight,
        conv1x1s1_dd_se_5_bias,
        conv1x1s1_di_se_5_weight,
        conv1x1s1_di_se_5_bias,
        2,
        1,
        24,
        4,
        magnification_factor + 42,
        sizeof(conv1x1s1_di_5_weight) / 2,
        sizeof(conv1x1s1_di_5_bias) / 2,
        sizeof(convdw3x3s2_5_weight) / 2,
        sizeof(convdw3x3s2_5_bias) / 2,
        sizeof(conv1x1s1_dd_5_weight) / 2,
        sizeof(conv1x1s1_dd_5_bias) / 2,
        sizeof(conv1x1s1_dd_se_5_weight) / 2,
        sizeof(conv1x1s1_dd_se_5_bias) / 2,
        sizeof(conv1x1s1_di_se_5_weight) / 2,
        sizeof(conv1x1s1_di_se_5_bias) / 2,
    };

    static Block_Bneck_SE_Short block6 = {
        conv1x1s1_di_6_weight,
        conv1x1s1_di_6_bias,
        convdw3x3s1_6_weight,
        convdw3x3s1_6_bias,
        conv1x1s1_dd_6_weight,
        conv1x1s1_dd_6_bias,
        conv1x1s1_dd_se_6_weight,
        conv1x1s1_dd_se_6_bias,
        conv1x1s1_di_se_6_weight,
        conv1x1s1_di_se_6_bias,
        1,
        1,
        24,
        4,
        magnification_factor + 52,
        sizeof(conv1x1s1_di_6_weight) / 2,
        sizeof(conv1x1s1_di_6_bias) / 2,
        sizeof(convdw3x3s1_6_weight) / 2,
        sizeof(convdw3x3s1_6_bias) / 2,
        sizeof(conv1x1s1_dd_6_weight) / 2,
        sizeof(conv1x1s1_dd_6_bias) / 2,
        sizeof(conv1x1s1_dd_se_6_weight) / 2,
        sizeof(conv1x1s1_dd_se_6_bias) / 2,
        sizeof(conv1x1s1_di_se_6_weight) / 2,
        sizeof(conv1x1s1_di_se_6_bias) / 2,
    };
    static Block_Conv1x1_Short postconv1x1s1 = {
        conv1x1s1_weight,
        conv1x1s1_bias,
        2,
        magnification_factor + 62,
        sizeof(conv1x1s1_weight) / 2,
        sizeof(conv1x1s1_bias) / 2,
    };

    NetSpdShort net = {
        &preconv3x3s2,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
        ver_spd,
    };
    return net;
}

//Block_Bneck_Se
static Mat bottleneck_se_short(Mat input, Block_Bneck_SE_Short* block)
{
    //init
    const int t = block->ratio; //ratio

    //conv1x1
    //conv1x1s1_neon
    int h = input.h, w = input.w, c = input.c * t, stride = block->stride;

    Mat mat_conv1x1_di = newMat(new_memory(input), w, h, c);
    float* weight_f = new_memory(mat_conv1x1_di);
    float* bias_f = weight_f + block->len_conv1x1_di_weight;
    short_to_float(weight_f, block->conv1x1_di_weight, block->len_conv1x1_di_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv1x1_di_bias, block->len_conv1x1_di_bias, block->magnification[1]);
    conv1x1s1_neon(input, mat_conv1x1_di, weight_f, bias_f);
    hswish_neon(mat_conv1x1_di);

    //conv3x3
    //padding
    //convdw3x3s1_neon or convdw3x3s2_neon
    Mat mat_conv3x3padding = newMat(new_memory(mat_conv1x1_di), w + 2 * block->padding, h + 2 * block->padding, c);
    memset(mat_conv3x3padding.data, 0, total(mat_conv3x3padding) * sizeof(float));
    padding(mat_conv1x1_di, mat_conv3x3padding, block->padding, block->padding, 0, 0);
    w = (w + 2 * block->padding - 3) / stride + 1, h = (h + 2 * block->padding - 3) / stride + 1;
    Mat mat_conv3x3 = newMat(mat_conv1x1_di.data, w, h, c); //回到mat_conv1x1_di.data指针，节省内存

    weight_f = new_memory(mat_conv3x3padding);
    bias_f = weight_f + block->len_convdw3x3_weight;
    short_to_float(weight_f, block->convdw3x3_weight, block->len_convdw3x3_weight, block->magnification[2]);
    short_to_float(bias_f, block->convdw3x3_bias, block->len_convdw3x3_bias, block->magnification[3]);

    if (stride == 1) {
        convdw3x3s1_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    } else if (stride == 2) {
        convdw3x3s2_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    }

    hswish_neon(mat_conv3x3);

    //conv1x1
    //conv1x1s1_neon
    c = block->out_channel;
    Mat mat_conv1x1_dd = newMat(new_memory(mat_conv3x3), w, h, c);
    weight_f = new_memory(mat_conv1x1_dd);
    bias_f = weight_f + block->len_conv1x1_dd_weight;
    short_to_float(weight_f, block->conv1x1_dd_weight, block->len_conv1x1_dd_weight, block->magnification[4]);
    short_to_float(bias_f, block->conv1x1_dd_bias, block->len_conv1x1_dd_bias, block->magnification[5]);
    conv1x1s1_neon(mat_conv3x3, mat_conv1x1_dd, weight_f, bias_f);

    //se
    //pooling_global
    //conv1x1s1_neon
    //conv1x1s1_neon
    Mat mat_global_pooling = newMat(new_memory(mat_conv1x1_dd), 1, 1, c);
    pooling_global(mat_conv1x1_dd, mat_global_pooling, PoolMethod_AVE);
    c /= 4;
    Mat mat_conv1x1dd_se = newMat(new_memory(mat_global_pooling), 1, 1, c);
    weight_f = new_memory(mat_conv1x1dd_se);
    bias_f = weight_f + block->len_conv1x1_dd_se_weight;
    short_to_float(weight_f, block->conv1x1_dd_se_weight, block->len_conv1x1_dd_se_weight, block->magnification[6]);
    short_to_float(bias_f, block->conv1x1_dd_se_bias, block->len_conv1x1_dd_se_bias, block->magnification[7]);
    conv1x1s1_neon(mat_global_pooling, mat_conv1x1dd_se, weight_f, bias_f);
    relu_neon(mat_conv1x1dd_se);
    c = block->out_channel;
    Mat mat_conv1x1di_se = newMat(new_memory(mat_conv1x1dd_se), 1, 1, c);
    weight_f = new_memory(mat_conv1x1di_se);
    bias_f = weight_f + block->len_conv1x1_di_se_weight;
    short_to_float(weight_f, block->conv1x1_di_se_weight, block->len_conv1x1_di_se_weight, block->magnification[8]);
    short_to_float(bias_f, block->conv1x1_di_se_bias, block->len_conv1x1_di_se_bias, block->magnification[9]);
    conv1x1s1_neon(mat_conv1x1dd_se, mat_conv1x1di_se, weight_f, bias_f);
    hsigmoid_neon(mat_conv1x1di_se);

    mat_scale_neon_inplace(mat_conv1x1_dd, mat_conv1x1di_se);

    //shortcut

    if (stride == 1) {
        mat_add_neon_inplace(input, mat_conv1x1_dd);
        return input;
    }

    Mat output = newMat(input.data, w, h, c);
    memcpy(output.data, mat_conv1x1_dd.data, mat_conv1x1_dd.cstep * c * sizeof(float));
    return output;
}

static int int2float(int* src, float* dst, int h, int w, const int cut) // int2float and cut
{
    int i, j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            dst[i * w + j] = (float)src[(i + cut) * w + j];
        }
    }

    return SL_RET_SUCCESS;
}

// extern void *aligned_malloc(size_t size, size_t align);
// forward

static int net_forward_spd(int* img, int h, int w, NetSpdShort net, float* out)
{
    if (img == NULL || out == NULL || h < 0 || w < 0) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    const int cut = 0;
    int hc, wc;
    hc = h - 2 * cut;           // [132, 32] -> [128, 32]
    wc = w;
    // wc = w - 2 * cut;
    int size = hc * wc;


    //memory block
    float* memory = (float*)malloc(7.2 * size * sizeof(float));       // 7*128*32*4 = 112k

    if (!memory) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }


    //左右memory的指针
    //float* left = memory;
    float* tensor = (float*)alignPtr((size_t)memory, MALLOC_ALIGN);
    float* right = tensor + alignPtr(6 * size, MALLOC_ALIGN);
    int2float(img, tensor, hc, wc, cut);                                    // CenterCrop
    totensor_neon(tensor, tensor, size);                                    // uint8 -> double [0, 1]
    Mat input = newMat(tensor, wc, hc, 1);
    // pre
    input = conv3x3_block_short(input, net.preconv3x3s2, right, 1, 0);
    //conv
    input = bottleneck_se_short(input, net.block1);
    input = bottleneck_se_short(input, net.block2);
    input = bottleneck_se_short(input, net.block3);
    input = bottleneck_se_short(input, net.block4);
    input = bottleneck_se_short(input, net.block5);
    input = bottleneck_se_short(input, net.block6);


    //global pooling
    Mat mat_avgpool = newMat(right, 1, 1, net.block6->out_channel);
    pooling_global(input, mat_avgpool, PoolMethod_AVE);
    //class
    Mat postconv1x1 = newMat(tensor, 1, 1, net.postconv1x1->out_channel);
    float* weight_f = new_memory(postconv1x1);
    float* bias_f = weight_f + net.postconv1x1->len_conv1x1_weight;
    short_to_float(weight_f, net.postconv1x1->conv1x1_weight, net.postconv1x1->len_conv1x1_weight, net.postconv1x1->magnification[0]);
    short_to_float(bias_f, net.postconv1x1->conv1x1_bias, net.postconv1x1->len_conv1x1_bias, net.postconv1x1->magnification[1]);
    conv1x1s1_neon(mat_avgpool, postconv1x1, weight_f, bias_f);

    flatten(postconv1x1, memory);

    //归一化
    float* soft_img = memory + net.postconv1x1->out_channel;
    soft_max(memory, net.postconv1x1->out_channel, soft_img);

    memcpy(out, soft_img, net.postconv1x1->out_channel * sizeof(float));

    //返回类别,0:手指 1:foreign matter
    int pos = 0;
    int i = 1;

    for (i = 1; i < net.postconv1x1->out_channel; i++) {
        if (soft_img[i] > soft_img[pos]) {
            pos = i;
        }
    }


    //printf("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", soft_img[0], soft_img[1], soft_img[2], soft_img[3], soft_img[4]);
    if (memory) {
        free(memory);
    }

    return pos;
}
int spd_check(int* src, const int h, const int w, float* out)
{
    get_version_spd();
#if BENCHMARK
    double start = get_current_time();
#endif
    SL_LOGD("spd_check image w=%d,h=%d", w, h);

    if (h != 118 || w != 32 || !src || !out) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int ret = 0;

    //int sum = 0;
    //int i, j;
    //for (i = 0; i < h; i++){
    //  for (j = 0; j < w; j++){
    //      //printf("%d", src[i * w + j]);
    //      sum += src[i * w + j];
    //  }
    //}

    ret = net_forward_spd(src, h, w, get_para_spd_v1(), out);
    //ret = net_forward_VGG(src, h, w, get_vgg_net_parameters(), out);
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}
