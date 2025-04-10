#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_mask.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Conv3x3_Short* block_mask_preconv3x3_1;
    Block_Conv3x3_Short* block_mask_preconv3x3_2;
    Block_Conv3x3_Short* block_mask_dsconv3x3_1;
    Block_Conv3x3_Short* block_mask_dsconv3x3_2;
    Block_Bneck_SE_Short* block_mask_block1;
    Block_Bneck_SE_Short* block_mask_block2;
    Block_Conv1x1_Short* block_mask_reconv1x1;
    Block_Conv3x3_Short* block_mask_usdeconv3x3_1;
    Block_Conv1x1_Short* block_mask_usdeconv1x1_1;
    Block_Conv3x3_Short* block_mask_usdeconv3x3_2;
    Block_Conv3x3_Short* block_mask_usdeconv3x3_3;
    Block_Conv3x3_Short* block_mask_postconv3x3;
    Block_Conv3x3_Short* block_mask_generateconv3x3;
    char* ver_mask;
} NetMask6Short;

static NetMask6Short get_para_mask_v6()
{
    static char* ver_mask = para_ver_mask;

    static Block_Conv3x3_Short block_mask_preconv3x3_1 = {
        para_mask_conv3x3_pre1_weight,
        para_mask_conv3x3_pre1_bias,
        1,
        1,
        4,
        para_mask_magnification_factor,
        sizeof(para_mask_conv3x3_pre1_weight) / 2,
        sizeof(para_mask_conv3x3_pre1_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_preconv3x3_2 = {
        para_mask_conv3x3_pre2_weight,
        para_mask_conv3x3_pre2_bias,
        1,
        1,
        4,
        para_mask_magnification_factor + 2,
        sizeof(para_mask_conv3x3_pre2_weight) / 2,
        sizeof(para_mask_conv3x3_pre2_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_dsconv3x3_1 = {
        para_mask_conv3x3_ds1_weight,
        para_mask_conv3x3_ds1_bias,
        2,
        1,
        8,
        para_mask_magnification_factor + 4,
        sizeof(para_mask_conv3x3_ds1_weight) / 2,
        sizeof(para_mask_conv3x3_ds1_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_dsconv3x3_2 = {
        para_mask_conv3x3_ds2_weight,
        para_mask_conv3x3_ds2_bias,
        2,
        1,
        16,
        para_mask_magnification_factor + 6,
        sizeof(para_mask_conv3x3_ds2_weight) / 2,
        sizeof(para_mask_conv3x3_ds2_bias) / 2,
    };

    static Block_Bneck_SE_Short block_mask_block1 = {
        para_mask_conv1x1s1_di_1_weight,
        para_mask_conv1x1s1_di_1_bias,
        para_mask_convdw3x3s2_1_weight,
        para_mask_convdw3x3s2_1_bias,
        para_mask_conv1x1s1_dd_1_weight,
        para_mask_conv1x1s1_dd_1_bias,
        0,
        0,
        0,
        0,
        2,
        1,
        32,
        40,
        para_mask_magnification_factor + 8,
        sizeof(para_mask_conv1x1s1_di_1_weight) / 2,
        sizeof(para_mask_conv1x1s1_di_1_bias) / 2,
        sizeof(para_mask_convdw3x3s2_1_weight) / 2,
        sizeof(para_mask_convdw3x3s2_1_bias) / 2,
        sizeof(para_mask_conv1x1s1_dd_1_weight) / 2,
        sizeof(para_mask_conv1x1s1_dd_1_bias) / 2,
        0,
        0,
        0,
        0,
    };

    static Block_Bneck_SE_Short block_mask_block2 = {
        0,
        0,
        para_mask_convdw3x3s1_2_weight,
        para_mask_convdw3x3s1_2_bias,
        para_mask_conv1x1s1_dd_2_weight,
        para_mask_conv1x1s1_dd_2_bias,
        para_mask_conv1x1s1_dd_se_2_weight,
        para_mask_conv1x1s1_dd_se_2_bias,
        para_mask_conv1x1s1_di_se_2_weight,
        para_mask_conv1x1s1_di_se_2_bias,
        1,
        1,
        32,
        32,
        para_mask_magnification_factor + 14,
        0,
        0,
        sizeof(para_mask_convdw3x3s1_2_weight) / 2,
        sizeof(para_mask_convdw3x3s1_2_bias) / 2,
        sizeof(para_mask_conv1x1s1_dd_2_weight) / 2,
        sizeof(para_mask_conv1x1s1_dd_2_bias) / 2,
        sizeof(para_mask_conv1x1s1_dd_se_2_weight) / 2,
        sizeof(para_mask_conv1x1s1_dd_se_2_bias) / 2,
        sizeof(para_mask_conv1x1s1_di_se_2_weight) / 2,
        sizeof(para_mask_conv1x1s1_di_se_2_bias) / 2,
    };

    static Block_Conv1x1_Short block_mask_reconv1x1 = {
        para_mask_reconv1x1_weight,
        para_mask_reconv1x1_bias,
        1,
        para_mask_magnification_factor + 22,
        sizeof(para_mask_reconv1x1_weight) / 2,
        sizeof(para_mask_reconv1x1_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_usdeconv3x3_1 = {
        para_mask_deconv3x3_us1_weight,
        para_mask_deconv3x3_us1_bias,
        2,
        1,
        16,
        para_mask_magnification_factor + 24,
        sizeof(para_mask_deconv3x3_us1_weight) / 2,
        sizeof(para_mask_deconv3x3_us1_bias) / 2,
    };
    static Block_Conv1x1_Short block_mask_usdeconv1x1_1 = {
        para_mask_deconv1x1_us1_weight,
        para_mask_deconv1x1_us1_bias,
        16,
        para_mask_magnification_factor + 26,
        sizeof(para_mask_deconv1x1_us1_weight) / 2,
        sizeof(para_mask_deconv1x1_us1_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_usdeconv3x3_2 = {
        para_mask_deconv3x3_us2_weight,
        para_mask_deconv3x3_us2_bias,
        2,
        1,
        8,
        para_mask_magnification_factor + 28,
        sizeof(para_mask_deconv3x3_us2_weight) / 2,
        sizeof(para_mask_deconv3x3_us2_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_usdeconv3x3_3 = {
        para_mask_deconv3x3_us3_weight,
        para_mask_deconv3x3_us3_bias,
        2,
        1,
        4,
        para_mask_magnification_factor + 30,
        sizeof(para_mask_deconv3x3_us3_weight) / 2,
        sizeof(para_mask_deconv3x3_us3_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_postconv3x3 = {
        para_mask_conv3x3_post_weight,
        para_mask_conv3x3_post_bias,
        1,
        1,
        4,
        para_mask_magnification_factor + 32,
        sizeof(para_mask_conv3x3_post_weight) / 2,
        sizeof(para_mask_conv3x3_post_bias) / 2,
    };

    static Block_Conv3x3_Short block_mask_generateconv3x3 = {
        para_mask_conv3x3_generate_weight,
        para_mask_conv3x3_generate_bias,
        1,
        1,
        1,
        para_mask_magnification_factor + 34,
        sizeof(para_mask_conv3x3_generate_weight) / 2,
        sizeof(para_mask_conv3x3_generate_bias) / 2,
    };


    NetMask6Short net = {
        &block_mask_preconv3x3_1,
        &block_mask_preconv3x3_2,
        &block_mask_dsconv3x3_1,
        &block_mask_dsconv3x3_2,
        &block_mask_block1,
        &block_mask_block2,
        &block_mask_reconv1x1,
        &block_mask_usdeconv3x3_1,
        &block_mask_usdeconv1x1_1,
        &block_mask_usdeconv3x3_2,
        &block_mask_usdeconv3x3_3,
        &block_mask_postconv3x3,
        &block_mask_generateconv3x3,
        ver_mask,
    };
    return net;
};

static Mat upconv3x3_block_fz_short(Mat input, Block_Conv3x3_Short* block, float* rightpointer) // �������ݱ���λ��left_memory
{
    int wc = input.w, hc = input.h;
    int group = 4;
    int out_c = block->out_channel / group;
    int inp_c = input.c / group;
    //up sampling
    hc = hc * 2, wc = wc * 2; //up sampling 1
    int i = 0;
    Mat input_group = input;
    input_group.c = inp_c;
    int jump_inp_mem = input.cstep * inp_c;
    Mat usdeconv3x3s2_1 = newMat(rightpointer, wc + 1, hc + 1, out_c);
    int jump_out_mem = usdeconv3x3s2_1.cstep * out_c;
    float* weight_f = usdeconv3x3s2_1.data + usdeconv3x3s2_1.cstep * block->out_channel;
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);

    float* weight = weight_f;
    float* bias = bias_f;
    int jump_weight = block->out_channel * input.c * 3 * 3 / group / group;
    int jump_bias = block->out_channel / group;

    for (; i < group; i++) {
        //Mat usdeconv3x3s2_1 = newMat(rightpointer, wc + 1, hc + 1, block->out_channel);
        deconv3x3s2_neon(input_group, usdeconv3x3s2_1, weight, bias);
        input_group.data += jump_inp_mem;
        usdeconv3x3s2_1.data += jump_out_mem;
        weight += jump_weight;
        bias += jump_bias;
    }

    usdeconv3x3s2_1.data = rightpointer;
    usdeconv3x3s2_1.c = block->out_channel;
    Mat usdeconv3x3s2_1_crop = newMat(rightpointer, wc, hc, block->out_channel);
    deconvcrop(usdeconv3x3s2_1, usdeconv3x3s2_1_crop, 1);
    relu_neon(usdeconv3x3s2_1_crop);
    return usdeconv3x3s2_1_crop;
}
static Mat upconv3x3_block_short(Mat input, Block_Conv3x3_Short* block, float* rightpointer) // �������ݱ���λ��left_memory
{

    int wc = input.w, hc = input.h;
    //up sampling
    hc = hc * 2, wc = wc * 2; //up sampling 1
    Mat usdeconv3x3s2_1 = newMat(rightpointer, wc + 1, hc + 1, block->out_channel);
    float* weight_f = new_memory(usdeconv3x3s2_1);
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);
    deconv3x3s2_neon(input, usdeconv3x3s2_1, weight_f, bias_f);
    Mat usdeconv3x3s2_1_crop = newMat(rightpointer, wc, hc, block->out_channel);
    deconvcrop(usdeconv3x3s2_1, usdeconv3x3s2_1_crop, 1);
    relu_neon(usdeconv3x3s2_1_crop);
    return usdeconv3x3s2_1_crop;
}
static Mat conv1x1_block_m_short(Mat input, Block_Conv1x1_Short* block, float* right)
{
    int w = input.w, h = input.h;
    Mat mat_convs = newMat(right, w, h, block->out_channel);
    float* weight_f = new_memory(mat_convs);
    float* bias_f = weight_f + block->len_conv1x1_weight;
    short_to_float(weight_f, block->conv1x1_weight, block->len_conv1x1_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv1x1_bias, block->len_conv1x1_bias, block->magnification[1]);
    conv1x1s1_neon(input, mat_convs, weight_f, bias_f);
    relu_neon(mat_convs);
    return mat_convs;
}


//mask
//forward
static int net_forward_mask_short(float* src, int h, int w, NetMask6Short net, int* score)
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
    float* left = tensor;
    float* right = tensor + LEFT_STRIDE;

    //pre
    input = conv3x3_block_short(input, net.block_mask_preconv3x3_1, right, 0, 2);
    input = conv3x3_block_short(input, net.block_mask_preconv3x3_2, right, 0, 2);

    // down sampling
    input = conv3x3_block_short(input, net.block_mask_dsconv3x3_1, right, 0, 2); // down sampling 1
    input = conv3x3_block_short(input, net.block_mask_dsconv3x3_2, right, 0, 2); // down sampling 2
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
    input = bottleneck1_short(mat_row_de, net.block_mask_block1);
    input = bottleneck2_short(input, net.block_mask_block2);
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
    Mat mat_avgpool = newMat(right, 1, 1, net.block_mask_block2->out_channel);
    pooling_global(input, mat_avgpool, PoolMethod_AVE);
    Mat postconv1x1 = newMat(new_memory(mat_avgpool), 1, 1, net.block_mask_reconv1x1->out_channel);
    float* weight_f = new_memory(postconv1x1);
    float* bias_f = weight_f + net.block_mask_reconv1x1->len_conv1x1_weight;
    short_to_float(weight_f, net.block_mask_reconv1x1->conv1x1_weight, net.block_mask_reconv1x1->len_conv1x1_weight, net.block_mask_reconv1x1->magnification[0]);
    short_to_float(bias_f, net.block_mask_reconv1x1->conv1x1_bias, net.block_mask_reconv1x1->len_conv1x1_bias, net.block_mask_reconv1x1->magnification[1]);
    conv1x1s1_neon(mat_avgpool, postconv1x1, weight_f, bias_f);
    hsigmoid_neon(postconv1x1);
    *score = (int)(postconv1x1.data[0] * 100 + 0.5);


    input = upconv3x3_block_fz_short(input, net.block_mask_usdeconv3x3_1, right);
    input = conv1x1_block_m_short(input, net.block_mask_usdeconv1x1_1, left);
    Mat mat_row_up = newMat(right, input.w, input.h * 2, input.c);
    bilinear_neon_cnn(input, mat_row_up, 0);

    mat_row_up.data = right_cp.data;
    mat_row_up.c = mat_row_up.c + right_cp.c;
    input = upconv3x3_block_short(mat_row_up, net.block_mask_usdeconv3x3_2, left);
    right = right_cp.data;
    input = upconv3x3_block_short(input, net.block_mask_usdeconv3x3_3, right);



    //post
    input = conv3x3_block_short(input, net.block_mask_postconv3x3, left, 0, 2);



    //generate
    Mat generateconv3x3padding = newMat(left, w + 2 * net.block_mask_generateconv3x3->padding, h + 2 * net.block_mask_generateconv3x3->padding, net.block_mask_postconv3x3->out_channel); //conv3x3
    memset(left, 0, total(generateconv3x3padding) * sizeof(float)); //padding需要先清空内存
    padding(input, generateconv3x3padding, net.block_mask_generateconv3x3->padding, net.block_mask_generateconv3x3->padding, 2, 0);
    Mat generateconv3x3 = newMat(right, input.w, input.h, net.block_mask_generateconv3x3->out_channel);
    weight_f = new_memory(generateconv3x3);
    bias_f = weight_f + net.block_mask_generateconv3x3->len_conv3x3_weight;
    short_to_float(weight_f, net.block_mask_generateconv3x3->conv3x3_weight, net.block_mask_generateconv3x3->len_conv3x3_weight, net.block_mask_generateconv3x3->magnification[0]);
    short_to_float(bias_f, net.block_mask_generateconv3x3->conv3x3_bias, net.block_mask_generateconv3x3->len_conv3x3_bias, net.block_mask_generateconv3x3->magnification[1]);
    conv3x3s1_neon(generateconv3x3padding, generateconv3x3, weight_f, bias_f);
    hsigmoid_neon(generateconv3x3);

    memcpy(src, generateconv3x3.data, input.w * input.h * sizeof(float));
    // delete[] memory;
    free(memory);
    return SL_RET_SUCCESS;
}

//main procedure
int scratch_mask(unsigned char* src, int h, int w, unsigned char* mask, int outh, int outw, int threshold, int* score)
{
    get_version_mask();
#if BENCHMARK
    double start = get_current_time();
#endif
    int ret = 0;

    if (h != 122 || w != 36 || !src || outh < 122 || outw < 36 || !mask || !score) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int hcrop = 118, wcrop = 32;
    float* src_f = (float*)malloc(hcrop * wcrop * sizeof(float));

    if (!src_f) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(src_f, 0, hcrop * wcrop * sizeof(float));
    int r_h = 80;
    int r_w = 24;
    float* dst = (float*)malloc(r_w * r_h * sizeof(float));   // 网络处理图像的尺寸为96x96

    if (!dst) {
        free(src_f);
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(dst, 0, r_w * r_h * sizeof(float));

    int i, j;

    for (i = 0; i < hcrop; i++)
        for (j = 0; j < wcrop; j++) {
            src_f[i * wcrop + j] = (float)src[(i + 2)  * w + j + 2];
        }

    //将图像resize到96x96
    Mat input = newMat(src_f, wcrop, hcrop, 1);
    Mat output = newMat(dst, r_w, r_h, 1);

    //将图片resize到96x96
    ret = bilinear_neon_cnn(input, output, 0);

    if (ret < 0) {
        free(src_f);
        free(dst);
        return SL_RET_FAIL;
    }

    free(src_f);
    src_f = NULL;
    ret = net_forward_mask_short(output.data, r_h, r_w, get_para_mask_v6(), score);

    if (ret < 0) {
        free(dst);
        return SL_RET_FAIL;
    }

    float* resize_dst = (float*)malloc(hcrop * wcrop * sizeof(float));

    if (!resize_dst) {
        //free(src_f);
        free(dst);
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(resize_dst, 0, hcrop * wcrop * sizeof(float));

    Mat resize_output = newMat(resize_dst, wcrop, hcrop, 1);

    //将图片resize到目标尺寸
    ret = bilinear_neon_cnn(output, resize_output, 0);

    if (ret < 0) {
        //free(src_f);
        free(dst);
        free(resize_dst);
        return SL_RET_FAIL;
    }

    float* padding_dst = (float*)malloc(outh * outw * sizeof(float));

    if (!padding_dst) {
        //free(src_f);
        free(dst);
        free(resize_dst);
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(padding_dst, 0, outh * outw * sizeof(float));

    Mat resize_out = newMat(padding_dst, outw, outh, 1);
    padding(resize_output, resize_out, 2, 2, 1, 0);

    for (i = 0; i < outh; i++) {
        for (j = 0; j < outw; j++) {
            mask[i * outw + j] = (unsigned char)(resize_out.data[i * outw + j] * 255 > threshold ? 255 : 0);
        }
    }

    //free(src_f);
    if (padding_dst) {
        free(padding_dst);
    }

    if (resize_dst) {
        free(resize_dst);
    }

    if (dst) {
        free(dst);
    }

#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}

