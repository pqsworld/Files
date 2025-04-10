#include "net_param/net_para_descexp.h"
#include "net_param/para_ori_exp.h"
#include "net_param/net_para_enhance.h"
#include "../net_cnn_common.h"
#include "../alog.h"

#include "net_api.h"
#include "string.h"
#include "stdlib.h"
#include <stdio.h>
#include <stdlib.h>
#include "stdint.h"
#include "SL_Math.h"
// #include "android.h"

typedef struct {
    Block_Conv3x3* featrues_deconv1;
    Block_Conv3x3* featrues_deconv2_1;
    Block_Conv3x3* featrues_deconv2_2;
    Block_Bneck_Dw* featrues_deconv3;
    Block_Ghost* featrues_ghost1;
    Block_Ghost* featrues_ghost2;
    Block_Bneck_Dw* featrues_upconv3;
    Block_Ghost* featrues_ghost3;
    Block_Ghost* featrues_ghost4;
    Block_Conv3x3* featrues_upconv2_3x3;
    Block_Conv1x1* featrues_upconv2_1x1;
    Block_Conv3x3* featrues_upconv1_1;
    Block_Conv3x3* featrues_upconv1_2;
    Block_Conv3x3* featrues_upconv1_3;
    char* ver;

} NetGSRCExtend;
NetGSRCExtend get_param_srcextendG();

NetGSRCExtend get_param_srcextendG()
{
    static char version[28] = "v20230208_ghost1";

    static Block_Conv3x3 deconv1 = {
        src_ext_deconv1_weight,
        src_ext_deconv1_bias,
        2, // stride
        1, // padding
        4, // outchannel
    };
    static Block_Conv3x3 deconv2_1 = {
        src_ext_deconv2_1_weight,
        src_ext_deconv2_1_bias,
        1, // stride
        1, // padding
        4, // outchannel
    };

    static Block_Conv3x3 deconv2_2 = {
        src_ext_deconv2_2_weight,
        src_ext_deconv2_2_bias,
        1, // stride
        2, // padding
        4, // outchannel
    };

    static Block_Bneck_Dw featrues_deconv3 = {
        src_ext_deconv3_1_weight,
        src_ext_deconv3_1_bias,
        src_ext_deconv3_2_weight,
        src_ext_deconv3_2_bias,
        src_ext_deconv3_3_weight,
        src_ext_deconv3_3_bias,
        2,  // stride
        1,  // padding
        16, // outchannel
        16, // expand_channel
        16  // groups
    };

    static Block_Ghost ghost1 = {
        src_ext_ghost1_prim1_weight,
        src_ext_ghost1_prim1_bias,
        src_ext_ghost1_cheap1_weight,
        src_ext_ghost1_cheap1_bias,
        2,
        1,
        3,
        1,
        1,
        32,
    };

    static Block_Ghost ghost2 = {
        src_ext_ghost2_prim1_weight,
        src_ext_ghost2_prim1_bias,
        src_ext_ghost2_cheap1_weight,
        src_ext_ghost2_cheap1_bias,
        2,
        1,
        3,
        1,
        1,
        16,
    };

    static Block_Bneck_Dw featrues_upconv3 = {
        src_ext_upconv3_1_weight,
        src_ext_upconv3_1_bias,
        src_ext_upconv3_2_weight,
        src_ext_upconv3_2_bias,
        src_ext_upconv3_3_weight,
        src_ext_upconv3_3_bias,
        1,  // stride
        1,  // padding
        16, // outchannel
        32, // expand_channel
        32  // groups
    };

    static Block_Ghost ghost3 = {
        src_ext_ghost3_prim1_weight,
        src_ext_ghost3_prim1_bias,
        src_ext_ghost3_cheap1_weight,
        src_ext_ghost3_cheap1_bias,
        2,
        1,
        3,
        1,
        1,
        32,
    };

    static Block_Ghost ghost4 = {
        src_ext_ghost4_prim1_weight,
        src_ext_ghost4_prim1_bias,
        src_ext_ghost4_cheap1_weight,
        src_ext_ghost4_cheap1_bias,
        2,
        1,
        3,
        1,
        1,
        16,
    };

    static Block_Conv3x3 featrues_upconv2_3x3 = {
        src_ext_upconv2_3x3_weight,
        src_ext_upconv2_3x3_bias,
        1, // stride
        1, // padding
        16,
    };
    static Block_Conv1x1 featrues_upconv2_1x1 = {
        src_ext_upconv2_1x1_weight,
        src_ext_upconv2_1x1_bias,
        8,
    };

    static Block_Conv3x3 featrues_upconv1_1 = {
        src_ext_upconv1_1_weight,
        src_ext_upconv1_1_bias,
        1, // stride
        1, // padding
        8, // outchannel
    };

    static Block_Conv3x3 featrues_upconv1_2 = {
        src_ext_upconv1_2_weight,
        src_ext_upconv1_2_bias,
        1, // stride
        1, // padding
        1, // outchannel
    };

    static Block_Conv3x3 featrues_upconv1_3 = {
        src_ext_upconv1_3_weight,
        src_ext_upconv1_3_bias,
        1, // stride
        1, // padding
        1, // outchannel
    };

    NetGSRCExtend net = {
        &deconv1,
        &deconv2_1,
        &deconv2_2,
        &featrues_deconv3,
        &ghost1,
        &ghost2,
        &featrues_upconv3,
        &ghost3,
        &ghost4,
        &featrues_upconv2_3x3,
        &featrues_upconv2_1x1,
        &featrues_upconv1_1,
        &featrues_upconv1_2,
        &featrues_upconv1_3,
        version,
    };
    return net;
}

typedef struct {
    Block_Conv3x3* featrues_deconv1;
    Block_Conv3x3* featrues_deconv2_1;
    Block_Conv3x3* featrues_deconv2_2;
    Block_Conv3x3* featrues_deconv3;
    Block_Sep* featrues_deconv4_1;
    Block_Sep* featrues_deconv4_2;
    Block_Resnet_Dw* featrues_resnet1;
    Block_Conv3x3* featrues_upconv3_3x3;
    Block_Conv1x1* featrues_upconv3_1x1;
    Block_Conv3x3* featrues_upconv3;
    Block_Conv3x3* featrues_upconv2;
    Block_Conv3x3* featrues_upconv2_1;
    Block_Conv3x3* featrues_upconv2_2;
    Block_Conv3x3* featrues_upconv1;
    Block_Conv1x1* featrues_upconv1_1x1;
    Block_Conv3x3* featrues_upconv0;
    char* ver;
} NetExp_small2212;
NetExp_small2212 get_param_exp_small();

NetExp_small2212 get_param_enh_small()
{
    static char version[28] = "6192_2_2_enh20230317";

    static Block_Conv3x3 enh_deconv1 = {
        para_enh_features_deconv1_weight,
        para_enh_features_deconv1_bias,
        2, // stride
        1, // padding
        4, // outchannel
    };

    static Block_Conv3x3 enh_deconv2_1 = {
        para_enh_features_deconv2_1_weight,
        para_enh_features_deconv2_1_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 enh_deconv2_2 = {
        para_enh_features_deconv2_2_weight,
        para_enh_features_deconv2_2_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 enh_deconv3 = {
        para_enh_features_deconv3_weight,
        para_enh_features_deconv3_bias,
        2,
        1,
        16,
    };
    static Block_Sep enh_deconv4_1 = {
        para_enh_features_deconv4_1_weight,
        para_enh_features_deconv4_1_bias,
        para_enh_features_deconv4_11_weight,
        para_enh_features_deconv4_11_bias,
        1,
        1,
        16,
        16,

    };

    static Block_Sep enh_deconv4_2 = {
        para_enh_features_deconv4_2_weight,
        para_enh_features_deconv4_2_bias,
        para_enh_features_deconv4_12_weight,
        para_enh_features_deconv4_12_bias,
        1,
        1,
        16,
        16,
    };

    static Block_Resnet_Dw enh_resnet1 = {
        para_enh_features_resnet1_1_weight,
        para_enh_features_resnet1_1_bias,
        para_enh_features_resnet1_2_weight,
        para_enh_features_resnet1_2_bias,
        para_enh_features_resnet1_3_weight,
        para_enh_features_resnet1_3_bias,
        para_enh_features_resnet1_4_weight,
        para_enh_features_resnet1_4_bias,
        1,
        1,
        32,
        32,
    };

    static Block_Conv3x3 enh_upconv3_3x3 = {
        para_enh_features_upconv3_3x3_weight,
        para_enh_features_upconv3_3x3_bias,
        1,
        1,
        32,
    };
    static Block_Conv1x1 enh_upconv3_1x1 = {
        para_enh_features_upconv3_1x1_weight,
        para_enh_features_upconv3_1x1_bias,
        16,
    };

    static Block_Conv3x3 enh_upconv3 = {
        para_enh_features_upconv3_3x3_1_weight,
        para_enh_features_upconv3_3x3_1_bias,
        1,
        1,
        16,
    };

    static Block_Conv3x3 enh_upconv2 = {
        para_enh_features_upconv2_weight,
        para_enh_features_upconv2_bias,
        1,
        1,
        8,
    };

    static Block_Conv3x3 enh_upconv2_1 = {
        para_enh_features_upconv2_1_weight,
        para_enh_features_upconv2_1_bias,
        1,
        1,
        4,
    };
    static Block_Conv3x3 enh_upconv2_2 = {
        para_enh_features_upconv2_2_weight,
        para_enh_features_upconv2_2_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 enh_upconv1 = {
        para_enh_features_upconv1_3x3_weight,
        para_enh_features_upconv1_3x3_bias,
        1,
        1,
        8,
    };
    static Block_Conv1x1 enh_upconv1_1x1 = {
        para_enh_features_upconv1_1x1_weight,
        para_enh_features_upconv1_1x1_bias,
        4,
    };

    static Block_Conv3x3 enh_upconv0 = {
        para_enh_features_upconv0_weight,
        para_enh_features_upconv0_bias,
        2,
        1,
        1,
    };

    NetExp_small2212 net = {
        &enh_deconv1,
        &enh_deconv2_1,
        &enh_deconv2_2,
        &enh_deconv3,
        &enh_deconv4_1,
        &enh_deconv4_2,
        &enh_resnet1,
        &enh_upconv3_3x3,
        &enh_upconv3_1x1,
        &enh_upconv3,
        &enh_upconv2,
        &enh_upconv2_1,
        &enh_upconv2_2,
        &enh_upconv1,
        &enh_upconv1_1x1,
        &enh_upconv0,
        version,
    };
    return net;
};

NetExp_small2212 get_param_exp_small()
{
    static char version[28] = "v20221012_resnet_small2212";

    static Block_Conv3x3 exp_deconv1 = {
        exp_deconv1_weight,
        exp_deconv1_bias,
        2, // stride
        1, // padding
        4, // outchannel
    };

    static Block_Conv3x3 exp_deconv2_1 = {
        exp_deconv2_1_weight,
        exp_deconv2_1_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 exp_deconv2_2 = {
        exp_deconv2_2_weight,
        exp_deconv2_2_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 exp_deconv3 = {
        exp_deconv3_weight,
        exp_deconv3_bias,
        2,
        1,
        16,
    };
    static Block_Sep exp_deconv4_1 = {
        exp_deconv4_1_weight,
        exp_deconv4_1_bias,
        exp_deconv4_11_weight,
        exp_deconv4_11_bias,
        1,
        1,
        16,
        16,

    };

    static Block_Sep exp_deconv4_2 = {
        exp_deconv4_2_weight,
        exp_deconv4_2_bias,
        exp_deconv4_12_weight,
        exp_deconv4_12_bias,
        1,
        1,
        16,
        16,
    };

    static Block_Resnet_Dw exp_resnet1 = {
        exp_resnet1_1_weight,
        exp_resnet1_1_bias,
        exp_resnet1_2_weight,
        exp_resnet1_2_bias,
        exp_resnet1_3_weight,
        exp_resnet1_3_bias,
        exp_resnet1_4_weight,
        exp_resnet1_4_bias,
        1,
        1,
        32,
        32,
    };

    static Block_Conv3x3 exp_upconv3_3x3 = {
        exp_upconv3_3x3_weight,
        exp_upconv3_3x3_bias,
        1,
        1,
        32,
    };
    static Block_Conv1x1 exp_upconv3_1x1 = {
        exp_upconv3_1x1_weight,
        exp_upconv3_1x1_bias,
        16,
    };

    static Block_Conv3x3 exp_upconv3 = {
        exp_upconv3_3x3_1_weight,
        exp_upconv3_3x3_1_bias,
        1,
        1,
        16,
    };

    static Block_Conv3x3 exp_upconv2 = {
        exp_upconv2_weight,
        exp_upconv2_bias,
        1,
        1,
        8,
    };

    static Block_Conv3x3 exp_upconv2_1 = {
        exp_upconv2_1_weight,
        exp_upconv2_1_bias,
        1,
        1,
        4,
    };
    static Block_Conv3x3 exp_upconv2_2 = {
        exp_upconv2_2_weight,
        exp_upconv2_2_bias,
        1,
        1,
        4,
    };

    static Block_Conv3x3 exp_upconv1 = {
        exp_upconv1_3x3_weight,
        exp_upconv1_3x3_bias,
        1,
        1,
        8,
    };
    static Block_Conv1x1 exp_upconv1_1x1 = {
        exp_upconv1_1x1_weight,
        exp_upconv1_1x1_bias,
        4,
    };

    static Block_Conv3x3 exp_upconv0 = {
        exp_upconv0_weight,
        exp_upconv0_bias,
        2,
        1,
        1,
    };

    NetExp_small2212 net = {
        &exp_deconv1,
        &exp_deconv2_1,
        &exp_deconv2_2,
        &exp_deconv3,
        &exp_deconv4_1,
        &exp_deconv4_2,
        &exp_resnet1,
        &exp_upconv3_3x3,
        &exp_upconv3_1x1,
        &exp_upconv3,
        &exp_upconv2,
        &exp_upconv2_1,
        &exp_upconv2_2,
        &exp_upconv1,
        &exp_upconv1_1x1,
        &exp_upconv0,
        version,
    };
    return net;
};

int conv_block(Mat* input, Block_Conv3x3* block, int kernel, int bool_relu)
{
    int h = input->h, w = input->w, c = input->c, stride = block->stride;

    float* left = input->data;
    float* right = left + alignPtr(input->cstep * MAX(c, block->out_channel) + h, MALLOC_ALIGN);

    Mat mat_convpadding = newMat(right,  w + 2 * block->padding, h + 2 * block->padding, c);

    if (bool_relu == 1) {
        leakyrelu_neon(*input, 0.2);
    }

    if (bool_relu == 2) {
        relu_neon(*input);
    }

    memset(right, 0, total(mat_convpadding) * sizeof(float));
    padding_normal(*input, mat_convpadding, block->padding);

    h = (h - kernel + 2 * block->padding) / stride + 1;
    w = (w - kernel + 2 * block->padding) / stride + 1;
    c = block->out_channel;
    Mat mat_convs = newMat(left, w, h, c);
    memset(left, 0, total(mat_convs) * sizeof(float));

    if (kernel == 3) {
        if (stride == 1) {
            conv3x3s1_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias);
        } else {
            conv3x3s2_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias);
        }
    } else if (kernel == 5) {
        if (stride == 1) {
            conv5x5s1_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias);
        } else {
            conv5x5s2_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias);
        }
    } else {
    }

    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int conv1x1_blocktool(Mat* input, Block_Conv1x1* block, float* right)
{
    int h = input->h, w = input->w, c = block->out_channel;
    Mat mat_convs = newMat(right, w, h, c);
    memset(right, 0, total(mat_convs) * sizeof(float));

    conv1x1s1_neon(*input, mat_convs, block->conv1x1_weight, block->conv1x1_bias);
    copy(mat_convs, *input);
    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);

    return SL_RET_SUCCESS;
}

int conv_sep_block(Mat* input, Block_Sep* block)
{
    int h = input->h, w = input->w, c = input->c;
    // float * left = new_memory(*input);
    // float *right = left + alignPtr(input->cstep*c + h, MALLOC_ALIGN); //h*w*c
    float* right = input->data + alignPtr(input->cstep * c + h, MALLOC_ALIGN); // h*w*c

    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, block->in_channel);
    memset(right, 0, total(mat_conv3x3padding) * sizeof(float));

    padding_normal(*input, mat_conv3x3padding, block->padding);
    float* rightpoint = new_memory(mat_conv3x3padding);
    Mat conv_tmp = newMat(rightpoint, w, h, block->in_channel);
    memset(rightpoint, 0, total(conv_tmp) * sizeof(float));

    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, block->conv3x3_1_weight, block->conv3x3_1_bias);

    /*Mat mat_convs = newMat(input->data, h, w, block->out_channel);
    memset(input->data, 0, total(mat_convs) * sizeof(float));*/
    Mat mat_convs = newMat(right, w, h, block->out_channel);
    memset(right, 0, total(mat_convs) * sizeof(float));
    conv1x1s1_neon(conv_tmp, mat_convs, block->conv3x3_2_weight, block->conv3x3_2_bias);
    copy(mat_convs, *input);

    input->h = h;
    input->w = w;
    input->c = block->out_channel;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int resnet3x3_blockdw(Mat* input, Block_Resnet_Dw* block)
{
    int h = input->h, w = input->w, c = input->c;
    float* left = new_memory(*input);
    float* right = left + alignPtr(input->cstep * c + h, MALLOC_ALIGN); //h*w*c

    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, block->out_channel);
    memset(right, 0, total(mat_conv3x3padding) * sizeof(float));

    padding_normal(*input, mat_conv3x3padding, block->padding);
    float* rightpoint = new_memory(mat_conv3x3padding);
    Mat conv_tmp = newMat(rightpoint, w, h, block->out_channel);
    memset(rightpoint, 0, total(conv_tmp) * sizeof(float));

    Mat mat_conv3x3s = newMat(left, w, h, block->out_channel);
    memset(left, 0, total(mat_conv3x3s) * sizeof(float));
    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, block->conv3x3_1_weight, block->conv3x3_1_bias);

    memset(left, 0, total(mat_conv3x3s) * sizeof(float));
    conv1x1s1_neon(conv_tmp, mat_conv3x3s, block->conv3x3_2_weight, block->conv3x3_2_bias);

    relu_neon(mat_conv3x3s);

    padding_normal(mat_conv3x3s, mat_conv3x3padding, block->padding);

    memset(rightpoint, 0, total(conv_tmp) * sizeof(float));

    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, block->conv3x3_3_weight, block->conv3x3_3_bias);

    memset(left, 0, total(mat_conv3x3s) * sizeof(float));
    conv1x1s1_neon(conv_tmp, mat_conv3x3s, block->conv3x3_4_weight, block->conv3x3_4_bias);
    leakyrelu_neon(mat_conv3x3s, 0.2);
    mat_add_neon_inplace(*input, mat_conv3x3s);
    input->h = h;
    input->w = w;
    input->c = block->in_channel;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int conv_group_single_block(Mat* input, Block_Conv3x3* block, int kernel, int group)
{
    int h = input->h, w = input->w, c = input->c, stride = block->stride;
    float* right = input->data + alignPtr(input->cstep * MAX(c, block->out_channel) + h, MALLOC_ALIGN);
    Mat mat_convpadding = newMat(right,  w + 2 * block->padding, h + 2 * block->padding, c);
    memset(right, 0, total(mat_convpadding) * sizeof(float));

    padding_normal(*input, mat_convpadding, block->padding);

    h = (h - kernel + 2 * block->padding) / stride + 1, w = (w - kernel + 2 * block->padding) / stride + 1, c = block->out_channel;

    Mat mat_convs = newMat(input->data, w, h, c);
    memset(input->data, 0, total(mat_convs) * sizeof(float));

    if (block->stride == 2) {
        groupconv3x3s2_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias, group);
    } else {
        groupconv3x3s1_neon(mat_convpadding, mat_convs, block->conv3x3_weight, block->conv3x3_bias, group);
    }

    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

//depth to space
static void depth2space(Mat input, Mat output, int block_size)
{
    int i, j, k;

    for (k = 0; k < output.c; k++) {
        int koustep = k * output.cstep;
        int kinstep = k * block_size * block_size * input.cstep;

        for (i = 0; i < output.h; i++) {
            int y = i;
            int mody = (y % block_size) * block_size;
            int ydiv2 = y / block_size;
            int yw = y * output.w;
            int yinw = ydiv2 * input.w;

            for (j = 0; j < output.w; j++) {
                int x = j;
                int modx = x % block_size;
                int xdiv2 = x / block_size;
                output.data[yw + x + koustep] = input.data[yinw + xdiv2 + (mody + modx) * input.cstep + kinstep];
            }
        }
    }
}

int ghost_bneck(Mat* input, Block_Ghost* block, int bool_relu)
{
    int h = input->h, w = input->w, c = input->c;
    int init_ch = (block->out_channel) >> 1;// SL_Ceil(block->out_channel / block->ratio);
    // int new_ch = init_ch;// init_ch*(block->ratio - 1);
    // int ksize = block->ksize;//1x1
    // int dwsize = block->dwsize;//3x3

    float* left = input->data;
    float* right = left + alignPtr(input->cstep * MAX(c, init_ch), MALLOC_ALIGN);

    Mat mat_convs = newMat(right, w, h, init_ch);
    memset(right, 0, total(mat_convs) * sizeof(float));

    conv1x1s1_neon(*input, mat_convs, block->conv1x1_weight, block->conv1x1_bias);

    if (bool_relu > 0) {
        relu_neon(mat_convs);
    }

    input->c = init_ch;
    copy(mat_convs, *input);

    right = left + input->cstep * init_ch;
    Mat cheap_opera = newMat(right, w, h, init_ch);
    memset(right, 0, total(cheap_opera) * sizeof(float));

    right = new_memory(cheap_opera);
    Mat mat_convpadding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, init_ch);
    memset(right, 0, total(mat_convpadding) * sizeof(float));
    padding_normal(*input, mat_convpadding, block->padding);

    convdw3x3s1_neon(mat_convpadding, cheap_opera, block->conv3x3_weight, block->conv3x3_bias);

    if (bool_relu > 0) {
        relu_neon(cheap_opera);
    }

    input->c = block->out_channel;

    return SL_RET_SUCCESS;
}

int conv_bottleneck_block(Mat* input, Block_Bneck_Dw* block, int kernel)
{
    int h = input->h, w = input->w, c = input->c, stride = block->stride;

    float* left = input->data;
    float* right = left + alignPtr(input->cstep * MAX(c, block->out_channel), MALLOC_ALIGN);
    float* rightp1 = left + input->cstep * c;

    Mat mat_convs = newMat(rightp1, w, h, block->expand_channel);
    memset(rightp1, 0, total(mat_convs) *sizeof(float));

    conv1x1s1_neon(*input, mat_convs, block->conv1x1_1_weight, block->conv1x1_1_bias);
    hswish_neon(mat_convs);

    float* righttmp = new_memory(mat_convs);
    Mat mat_convpadding = newMat(righttmp, w + 2 * block->padding, h + 2 * block->padding, mat_convs.c);

    memset(righttmp, 0, total(mat_convpadding) *sizeof(float));
    padding_normal(mat_convs, mat_convpadding, block->padding);

    h = (h - kernel + 2 * block->padding) / stride + 1;
    w = (w - kernel + 2 * block->padding) / stride + 1;
    c = block->expand_channel;

    Mat mat_tmp = newMat(right, w, h, c);
    memset(right, 0, total(mat_tmp) * sizeof(float));

    if (stride == 1) {
        convdw3x3s1_neon(mat_convpadding, mat_tmp, block->conv3x3_weight, block->conv3x3_bias);
    } else {
        convdw3x3s2_neon(mat_convpadding, mat_tmp, block->conv3x3_weight, block->conv3x3_bias);
    }

    hswish_neon(mat_tmp);

    h = mat_tmp.h;
    w = mat_tmp.w;
    c = block->out_channel;
    Mat mat_out = newMat(left, w, h, c);
    conv1x1s1_neon(mat_tmp, mat_out, block->conv1x1_2_weight, block->conv1x1_2_bias);

    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

// deconv4*4_block
int deconv4x4_block(Mat* input, Block_Conv3x3* block)
{
    int h = input->h, w = input->w, c = block->out_channel;

    float* left = input->data;
    int address = MAX(input->c, c << 2);
    float* right = left + alignPtr(input->cstep * address + h, MALLOC_ALIGN);

    h = (h - 1) * block->stride + 3, w = (w - 1) * block->stride + 3;

    Mat mat_conv4x4s = newMat(right, w, h, c);
    memset(right, 0, total(mat_conv4x4s) * sizeof(float));
    deconv4x4s2_neon(*input, mat_conv4x4s, block->conv3x3_weight, block->conv3x3_bias);
    Mat deconv_crop = newMat(left, w - 1, h - 1, block->out_channel);
    memset(left, 0, total(deconv_crop) * sizeof(float));
    deconvcrop(mat_conv4x4s, deconv_crop, 1);
    input->h = h - 1;
    input->w = w - 1;
    input->c = block->out_channel;
    input->cstep = alignPtr((w - 1) * (h - 1), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

// desc exp
int pix2pix_resnetblockdw(float* src, short h, short w, NetExp_small2212 net)
{
    int hc = h, wc = w;
    int ori_size = hc * wc;
    // int j = 0;

    const int LEFT_STRIDE = alignPtr(ori_size, MALLOC_ALIGN);
    int size = (int)(11.2 * ori_size);
    float* memory = (float*)sl_aligned_malloc(size * sizeof(float), MALLOC_ALIGN);

    if (NULL == memory) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(memory, 0, size * sizeof(float));

    float* img_diff = (float*)alignPtr((size_t)memory, MALLOC_ALIGN);

    // normalization
    normalize_neon(src, img_diff, 127.5f, 127.5f, ori_size);

    Mat input0 = newMat(img_diff,  wc, hc, 1);

    conv_block(&input0, net.featrues_deconv1, 3, 0);

    leakyrelu_neon(input0, 0.2);
    conv_block(&input0, net.featrues_deconv2_1, 3, 0);
    float* leftp = new_memory(input0);
    Mat deconv2_2 = newMat(leftp, input0.w, input0.h, input0.c);
    memset(leftp, 0, total(deconv2_2) * sizeof(float));
    copy(input0, deconv2_2);
    leakyrelu_neon(deconv2_2, 0.2);

    conv_block(&deconv2_2, net.featrues_deconv2_2, 3, 0);

    Mat inputmp = newMat(img_diff,  deconv2_2.w, deconv2_2.h, deconv2_2.c * 2);

    conv_block(&inputmp, net.featrues_deconv3, 3, 1);

    leakyrelu_neon(inputmp, 0.2);

    conv_sep_block(&inputmp, net.featrues_deconv4_1);
    float* leftp2 = new_memory(inputmp);
    Mat deconv4_2 = newMat(leftp2,  inputmp.w, inputmp.h, inputmp.c);
    memset(leftp2, 0, total(deconv4_2) * sizeof(float));
    copy(inputmp, deconv4_2);
    leakyrelu_neon(deconv4_2, 0.2);
    conv_sep_block(&deconv4_2, net.featrues_deconv4_2);

    Mat inputmp0 = newMat(img_diff, inputmp.w, inputmp.h, inputmp.c * 2);


    float* right = new_memory(inputmp0) + 4 * LEFT_STRIDE;
    resnet3x3_blockdw(&inputmp0, net.featrues_resnet1);

    relu_neon(inputmp0);

    conv_group_single_block(&inputmp0, net.featrues_upconv3_3x3, 3, 4);
    conv1x1_blocktool(&inputmp0, net.featrues_upconv3_1x1, right + LEFT_STRIDE);

    float* tmright = inputmp0.data + inputmp0.cstep * inputmp0.c * 2;
    Mat out_up2 = newMat(tmright, inputmp0.w * 2, inputmp0.h * 2, inputmp.c);
    memset(tmright, 0, total(out_up2) * sizeof(float));
    bilinear_neon_cnn(inputmp0, out_up2, 0);


    conv_block(&out_up2, net.featrues_upconv3, 3, 0);

    relu_neon(out_up2);
    conv_block(&out_up2, net.featrues_upconv2, 3, 0);

    relu_neon(out_up2);

    conv_block(&out_up2, net.featrues_upconv2_1, 3, 0);

    Mat upconv2_2 = newMat(new_memory(out_up2),  out_up2.w, out_up2.h, out_up2.c);
    memset(new_memory(out_up2), 0, total(upconv2_2) * sizeof(float));
    copy(out_up2, upconv2_2);
    relu_neon(upconv2_2);
    conv_block(&upconv2_2, net.featrues_upconv2_2, 3, 0);

    Mat upconv2 = newMat(tmright,  upconv2_2.w, upconv2_2.h, upconv2_2.c * 2);
    relu_neon(upconv2);
    conv_group_single_block(&upconv2, net.featrues_upconv1, 3, 4);
    conv1x1_blocktool(&upconv2, net.featrues_upconv1_1x1, right + 2 * LEFT_STRIDE);

    relu_neon(upconv2);
    deconv4x4_block(&upconv2, net.featrues_upconv0);
    tanh_neon(upconv2);

    int i = 0;


    for (i = 0; i < ori_size; i++) {
        unsigned short tmp = (upconv2.data[i] + 1) * 127.5;

        if (tmp < 0) {
            tmp = 0;
        }

        if (tmp > 255) {
            tmp = 255;
        }

        src[i] = tmp;
    }

    if (memory) {
        sl_aligned_free(memory);
    }

    return SL_RET_SUCCESS;

}


int pix2pix_oriextend(float* src, short h, short w, NetGSRCExtend net)
{
    int hc = h, wc = w;
    int ori_size = hc * wc;
    int i = 0;
    int size = (int)(ori_size * 12.2);
    int total_memory = size * sizeof(float);
    float* memory = (float*)sl_aligned_malloc(total_memory, MALLOC_ALIGN);

    if (NULL == memory) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(memory, 0, total_memory);

    float* img_diff = (float*)alignPtr((size_t)memory, MALLOC_ALIGN);

    // normalization
    normalize_neon(src, img_diff, 127.5f, 127.5f, ori_size);
    Mat input = newMat(img_diff, wc, hc, 1);

    conv_block(&input, net.featrues_deconv1, 3, 0);
    relu_neon(input);

    conv_block(&input, net.featrues_deconv2_1, 3, 1);

    float* leftp = new_memory(input);
    Mat deconv2_2 = newMat(leftp, input.w, input.h, input.c);
    memset(leftp, 0, total(deconv2_2) * sizeof(float));
    copy(input, deconv2_2);

    conv_block(&deconv2_2, net.featrues_deconv2_2, 5, 1);

    Mat inputmp = newMat(img_diff, deconv2_2.w, deconv2_2.h, deconv2_2.c * 2);
    conv_bottleneck_block(&inputmp, net.featrues_deconv3, 3);

    leftp = new_memory(inputmp);
    Mat inputmp1 = newMat(leftp, inputmp.w, inputmp.h, inputmp.c);
    copy(inputmp, inputmp1);

    ghost_bneck(&inputmp1, net.featrues_ghost1, 1);
    ghost_bneck(&inputmp1, net.featrues_ghost2, 0);
    mat_add_neon_inplace(inputmp, inputmp1);

    conv_bottleneck_block(&inputmp, net.featrues_upconv3, 3);

    leftp = new_memory(inputmp);
    Mat input2 = newMat(leftp, inputmp.w, inputmp.h, inputmp.c);
    memset(leftp, 0, total(input2) * sizeof(float));
    copy(inputmp, input2);
    ghost_bneck(&input2, net.featrues_ghost3, 1);
    ghost_bneck(&input2, net.featrues_ghost4, 0);
    mat_add_neon_inplace(inputmp, input2);

    conv_block(&inputmp, net.featrues_upconv2_3x3, 3, 2);

    float* tmright = inputmp.data + inputmp.cstep * inputmp.c * 2;
    Mat out_up = newMat(tmright, inputmp.w * 2, inputmp.h * 2, inputmp.c);
    memset(tmright, 0, total(out_up) * sizeof(float));
    bilinear_neon_cnn(inputmp, out_up, 0);

    float* right = new_memory(out_up);
    conv1x1_blocktool(&out_up, net.featrues_upconv2_1x1, right);

    conv_block(&out_up, net.featrues_upconv1_1, 3, 0);
    tmright = out_up.data + out_up.cstep * out_up.c;
    Mat out_up2 = newMat(tmright, out_up.w * 2, out_up.h * 2, out_up.c / 4);
    memset(tmright, 0, total(out_up2) * sizeof(float));
    depth2space(out_up, out_up2, 2);
    conv_block(&out_up2, net.featrues_upconv1_2, 3, 0);
    relu_neon(out_up2);
    conv_block(&out_up2, net.featrues_upconv1_3, 3, 0);
    tanh_neon(out_up2);

    for (i = 0; i < ori_size; i++) {
        float tou = (out_up2.data[i] + 1) * 127.5;
        unsigned char tmp;

        if (tou >= 0 && tou <= 255) {
            tmp = tou;
        } else {
            if (tou < 0) {
                tmp = 0;
            } else {
                tmp = 255;
            }
        }

        src[i] = tmp;
    }

    if (memory) {
        sl_aligned_free(memory);
    }

    return SL_RET_SUCCESS;

}

/*描述子图扩边*/
int desc_exp(unsigned char* src, const int h, const int w, unsigned short* dst, const int ph, const int pw)
{
    if (h <= 0 || w <= 0 || !src || ph <= 0 || pw <= 0 || !dst) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    get_version_exp();

    int oh = h + ph * 2;
    int ow = w + pw * 2;
    int ori_size = oh * ow;

    float* dst_f = (float*)malloc(ori_size * sizeof(float));

    if (!dst_f) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(dst_f, 0, ori_size * sizeof(float));
    int i = 0, j = 0;

    for (i = 0; i < ori_size; i++) {
        dst_f[i] = 255;
    }

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            dst_f[(i + ph) * ow + j + pw] = (float)src[i * w + j];
        }
    }

    int ret = 0;

    ret = pix2pix_resnetblockdw(dst_f, oh, ow, get_param_exp_small());

    if (ret < 0) {
        free(dst_f);
        return ret;
    }

    for (i = 0; i < oh - 2; i++) {
        for (j = 0; j < ow; j++) {
            unsigned short tmp = dst_f[(i + 1) * ow + j];

            dst[i * ow + j] = tmp << 8;
        }
    }

    //only edge
#if 1

    int left_num = 4, right_num = 4;

    float mergeCoeff_src = 0.4, mergeCoeff_ex = 0.6;

    for (i = 0; i < h; i++) {
        for (j = left_num; j < w - right_num; j++) {
            unsigned short tmp0 = src[i * w + j];
            dst[(i + ph - 1) * ow + j + pw] = tmp0 << 8;
        }

        for (j = 0; j < left_num; j++) {
            unsigned short tmp0 = src[i * w + j];
            unsigned short tmp = dst[(i + ph - 1) * ow + j + pw];

            dst[(i + ph - 1) * ow + j + pw] = (unsigned short)((tmp0 << 8) * mergeCoeff_src + tmp * mergeCoeff_ex);
        }

        for (j = w - right_num; j < w; j++) {
            unsigned short tmp0 = src[i * w + j];
            unsigned short tmp = dst[(i + ph - 1) * ow + j + pw];

            dst[(i + ph - 1) * ow + j + pw] = (unsigned short)((tmp0 << 8) * mergeCoeff_src + tmp * mergeCoeff_ex);
        }
    }

#endif

    free(dst_f);
    return ret;
}

/*  h,w 代表的是输入尺寸，exth,extw分别代表上下左右各扩多少个像素  */
int ori_exp(unsigned char* src, const int h, const int w, unsigned char* dst, const int exth, const int extw)
{

    int oh = h + (exth * 2);
    int ow = w + (extw * 2);
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

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            dst_f[(i + exth) * ow + j + extw] = (float)src[i * w + j];
        }
    }

    int ret = 0;
    ret = pix2pix_oriextend(dst_f, oh, ow, get_param_srcextendG());

    //for (j = 0; j < ori_size; j++) {
    //
    //  unsigned char tmp = dst_f[j];
    //  dst[j] = tmp;
    //}

    // 中心替换
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            dst[(i + exth) * ow + j + extw] = src[i * w + j];
        }
    }

    //left
    for (i = 0; i < oh; i++) {
        for (j = 0; j < extw; j++) {
            unsigned char tmp = dst_f[i * ow + j];
            dst[i * ow + j] = tmp;
        }
    }

    //right
    for (i = 0; i < oh; i++) {
        for (j = w + extw; j < ow; j++) {
            unsigned char tmp = dst_f[i * ow + j];
            dst[i * ow + j] = tmp;
        }
    }

    free(dst_f);
    return ret;
}

//enhance
int net_enhance(unsigned char* src, const int h, const int w, unsigned char* dst)
{
    if (h <= 0 || w <= 0 || !src || !dst) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    get_version_enhance();

    int oh = h;
    int ow = w;
    int ori_size = oh * ow;
    float* dst_f = (float*)malloc(ori_size * sizeof(float));

    if (!dst_f) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    memset(dst_f, 0, ori_size * sizeof(float));

    int i = 0, j = 0;

    for (i = 0; i < ori_size; i++) {
        dst_f[i] = (float)src[i];
    }

    int ret = 0;
    ret = pix2pix_resnetblockdw(dst_f, oh, ow, get_param_enh_small());

    for (i = 0; i < ori_size; i++) {
        unsigned char tmp = dst_f[i];
        dst[i] = tmp;
    }

    free(dst_f);
    return ret;
}