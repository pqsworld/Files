#ifndef __CNN_PARAM_H__
#define __CNN_PARAM_H__

//spoof
typedef struct {
    float* conv1x1_di_weight;
    float* conv1x1_di_bias;
    float* convdw3x3_weight;
    float* convdw3x3_bias;
    float* conv1x1_dd_weight;
    float* conv1x1_dd_bias;
    int stride;
    int padding;
    int out_channel;
    int ratio;
} Block_Bneck;

//ori_exp_6192
typedef struct {
    float* conv1x1_1_weight;
    float* conv1x1_1_bias;
    float* conv3x3_weight;
    float* conv3x3_bias;
    float* conv1x1_2_weight;
    float* conv1x1_2_bias;
    int stride;
    int padding;
    int out_channel;
    int expand_channel;
    int groups;
} Block_Bneck_Dw;

//mistouch spoof
typedef struct {
    short* conv1x1_di_weight;
    short* conv1x1_di_bias;
    short* convdw3x3_weight;
    short* convdw3x3_bias;
    short* conv1x1_dd_weight;
    short* conv1x1_dd_bias;
    int stride;
    int padding;
    int out_channel;
    int ratio;
    int* magnification;
    int len_conv1x1_di_weight;
    int len_conv1x1_di_bias;
    int len_convdw3x3_weight;
    int len_convdw3x3_bias;
    int len_conv1x1_dd_weight;
    int len_conv1x1_dd_bias;
} Block_Bneck_Short;

//mask quality spd
typedef struct {
    short* conv1x1_di_weight;
    short* conv1x1_di_bias;
    short* convdw3x3_weight;
    short* convdw3x3_bias;
    short* conv1x1_dd_weight;
    short* conv1x1_dd_bias;
    short* conv1x1_dd_se_weight;
    short* conv1x1_dd_se_bias;
    short* conv1x1_di_se_weight;
    short* conv1x1_di_se_bias;
    int stride;
    int padding;
    int out_channel;
    int ratio;
    int* magnification;
    int len_conv1x1_di_weight;
    int len_conv1x1_di_bias;
    int len_convdw3x3_weight;
    int len_convdw3x3_bias;
    int len_conv1x1_dd_weight;
    int len_conv1x1_dd_bias;
    int len_conv1x1_dd_se_weight;
    int len_conv1x1_dd_se_bias;
    int len_conv1x1_di_se_weight;
    int len_conv1x1_di_se_bias;
} Block_Bneck_SE_Short;

//patch_6193
typedef struct {
    short* conv1x1_di_weight;
    float* conv1x1_di_bias;
    short* convdw3x3_weight;
    float* convdw3x3_bias;
    short* conv1x1_dd_weight;
    float* conv1x1_dd_bias;
    short* conv1x1_dd_se_weight;
    float* conv1x1_dd_se_bias;
    short* conv1x1_di_se_weight;
    float* conv1x1_di_se_bias;
    int stride;
    int padding;
    int out_channel;
    int ratio;
} Block_Bneck_SE_Short_Desc;

//spoof ori_enh
typedef struct {
    float* conv1x1_weight;
    float* conv1x1_bias;
    int out_channel;
} Block_Conv1x1;

typedef struct {
    short* conv1x1_weight;
    float* conv1x1_bias;
    int out_channel;
} Block_Conv1x1_Short_Desc;

//spoof mistouch enhance exp mask spd quality
typedef struct {
    short* conv1x1_weight;
    short* conv1x1_bias;
    int out_channel;
    int* magnification;
    int len_conv1x1_weight;
    int len_conv1x1_bias;
} Block_Conv1x1_Short;

//ori_enh
typedef struct {
    float* conv1x1_weight;
    float* conv1x1_bias;
    float* conv3x3_weight;
    float* conv3x3_bias;
    int ratio;
    int ksize;
    int dwsize;
    int stride;
    int padding;
    int out_channel;
} Block_Ghost;

//oei_enh
typedef struct {
    float* conv3x3_1_weight;
    float* conv3x3_1_bias;
    float* conv3x3_2_weight;
    float* conv3x3_2_bias;
    int stride;
    int padding;
    int out_channel;
} Block_Resnet;

//ori_enh
typedef struct {
    float* conv3x3_1_weight;
    float* conv3x3_1_bias;
    float* conv3x3_2_weight;
    float* conv3x3_2_bias;
    float* conv3x3_3_weight;
    float* conv3x3_3_bias;
    float* conv3x3_4_weight;
    float* conv3x3_4_bias;
    int stride;
    int padding;
    int in_channel;
    int out_channel;
} Block_Resnet_Dw;

//exp enhance
typedef struct {
    short* conv3x3_1_weight;
    short* conv3x3_1_bias;
    short* conv3x3_2_weight;
    short* conv3x3_2_bias;
    short* conv3x3_3_weight;
    short* conv3x3_3_bias;
    short* conv3x3_4_weight;
    short* conv3x3_4_bias;
    int stride;
    int padding;
    int in_channel;
    int out_channel;
    int* magnification;
    int len_conv3x3_1_weight;
    int len_conv3x3_1_bias;
    int len_conv3x3_2_weight;
    int len_conv3x3_2_bias;
    int len_conv3x3_3_weight;
    int len_conv3x3_3_bias;
    int len_conv3x3_4_weight;
    int len_conv3x3_4_bias;
} Block_Resnet_Short;

//spoof mistouch ori_enh  exp enhance mask spd
typedef struct {
    float* conv3x3_weight;
    float* conv3x3_bias;
    int stride;
    int padding;
    int out_channel;
} Block_Conv3x3;

//ori_enh enhance exp
typedef struct {
    float* conv3x3_1_weight;
    float* conv3x3_1_bias;
    float* conv3x3_2_weight;
    float* conv3x3_2_bias;
    int stride;
    int padding;
    int in_channel;
    int out_channel;
} Block_Sep;

//mistouch spoof enhance  exp mask spd quality
typedef struct {
    short* conv3x3_weight;
    short* conv3x3_bias;
    int stride;
    int padding;
    int out_channel;
    int* magnification;
    int len_conv3x3_weight;
    int len_conv3x3_bias;
} Block_Conv3x3_Short;

//exp enhance
typedef struct {
    short* conv3x3_1_weight;
    short* conv3x3_1_bias;
    short* conv3x3_2_weight;
    short* conv3x3_2_bias;
    int stride;
    int padding;
    int in_channel;
    int out_channel;
    int* magnification;
    int len_conv3x3_1_weight;
    int len_conv3x3_1_bias;
    int len_conv3x3_2_weight;
    int len_conv3x3_2_bias;
} Block_Sep_Short;


#endif
