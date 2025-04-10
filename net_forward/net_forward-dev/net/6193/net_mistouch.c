#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_mistouch.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Conv3x3_Short* preconv3x3s1;
    Block_Bneck_Short* block1;
    Block_Bneck_Short* block2;
    Block_Bneck_Short* block3;
    Block_Bneck_Short* block4;
    Block_Bneck_Short* block5;
    Block_Bneck_Short* block6;
    Block_Conv1x1_Short* postconv1x1;
    char* ver_mistouch;
} NetMistouchShort;
// beta = this.offset.value,
// gamma=this.scale.value,eps=this.epsilon,
// mean=this.trainedmeancache.value,var=this.trainedvariancecache.value
// #define OP_GEN2_ART_RAW_JIANBIAN

static NetMistouchShort get_para_mistouch_v1()
{
    static char* ver_mistouch = para_ver_mistouch;

    static Block_Conv3x3_Short preconv3x3s1 = {
        conv3x3_weight,
        conv3x3_bias,
        1,
        1,
        8,
        magnification_factor,
        sizeof(conv3x3_weight) / 2,
        sizeof(conv3x3_bias) / 2,
    };

    static Block_Bneck_Short block1 = {
        conv1x1s1_di_1_weight,
        conv1x1s1_di_1_bias,
        convdw3x3s1_1_weight,
        convdw3x3s1_1_bias,
        conv1x1s1_dd_1_weight,
        conv1x1s1_dd_1_bias,
        1,
        1,
        8,
        1,
        magnification_factor + 2,
        sizeof(conv1x1s1_di_1_weight) / 2,
        sizeof(conv1x1s1_di_1_bias) / 2,
        sizeof(convdw3x3s1_1_weight) / 2,
        sizeof(convdw3x3s1_1_bias) / 2,
        sizeof(conv1x1s1_dd_1_weight) / 2,
        sizeof(conv1x1s1_dd_1_bias) / 2,
    };

    static Block_Bneck_Short block2 = {
        conv1x1s1_di_2_weight,
        conv1x1s1_di_2_bias,
        convdw3x3s2_2_weight,
        convdw3x3s2_2_bias,
        conv1x1s1_dd_2_weight,
        conv1x1s1_dd_2_bias,
        2,
        1,
        16,
        4,
        magnification_factor + 8,
        sizeof(conv1x1s1_di_2_weight) / 2,
        sizeof(conv1x1s1_di_2_bias) / 2,
        sizeof(convdw3x3s2_2_weight) / 2,
        sizeof(convdw3x3s2_2_bias) / 2,
        sizeof(conv1x1s1_dd_2_weight) / 2,
        sizeof(conv1x1s1_dd_2_bias) / 2,
    };

    static Block_Bneck_Short block3 = {
        conv1x1s1_di_3_weight,
        conv1x1s1_di_3_bias,
        convdw3x3s1_3_weight,
        convdw3x3s1_3_bias,
        conv1x1s1_dd_3_weight,
        conv1x1s1_dd_3_bias,
        1,
        1,
        16,
        4,
        magnification_factor + 14,
        sizeof(conv1x1s1_di_3_weight) / 2,
        sizeof(conv1x1s1_di_3_bias) / 2,
        sizeof(convdw3x3s1_3_weight) / 2,
        sizeof(convdw3x3s1_3_bias) / 2,
        sizeof(conv1x1s1_dd_3_weight) / 2,
        sizeof(conv1x1s1_dd_3_bias) / 2,
    };

    static Block_Bneck_Short block4 = {
        conv1x1s1_di_4_weight,
        conv1x1s1_di_4_bias,
        convdw3x3s2_4_weight,
        convdw3x3s2_4_bias,
        conv1x1s1_dd_4_weight,
        conv1x1s1_dd_4_bias,
        2,
        1,
        24,
        4,
        magnification_factor + 20,
        sizeof(conv1x1s1_di_4_weight) / 2,
        sizeof(conv1x1s1_di_4_bias) / 2,
        sizeof(convdw3x3s2_4_weight) / 2,
        sizeof(convdw3x3s2_4_bias) / 2,
        sizeof(conv1x1s1_dd_4_weight) / 2,
        sizeof(conv1x1s1_dd_4_bias) / 2,
    };

    static Block_Bneck_Short block5 = {
        conv1x1s1_di_5_weight,
        conv1x1s1_di_5_bias,
        convdw3x3s1_5_weight,
        convdw3x3s1_5_bias,
        conv1x1s1_dd_5_weight,
        conv1x1s1_dd_5_bias,
        1,
        1,
        24,
        4,
        magnification_factor + 26,
        sizeof(conv1x1s1_di_5_weight) / 2,
        sizeof(conv1x1s1_di_5_bias) / 2,
        sizeof(convdw3x3s1_5_weight) / 2,
        sizeof(convdw3x3s1_5_bias) / 2,
        sizeof(conv1x1s1_dd_5_weight) / 2,
        sizeof(conv1x1s1_dd_5_bias) / 2,
    };

    static Block_Bneck_Short block6 = {
        conv1x1s1_di_6_weight,
        conv1x1s1_di_6_bias,
        convdw3x3s2_6_weight,
        convdw3x3s2_6_bias,
        conv1x1s1_dd_6_weight,
        conv1x1s1_dd_6_bias,
        2,
        1,
        32,
        4,
        magnification_factor + 32,
        sizeof(conv1x1s1_di_6_weight) / 2,
        sizeof(conv1x1s1_di_6_bias) / 2,
        sizeof(convdw3x3s2_6_weight) / 2,
        sizeof(convdw3x3s2_6_bias) / 2,
        sizeof(conv1x1s1_dd_6_weight) / 2,
        sizeof(conv1x1s1_dd_6_bias) / 2,
    };
    static Block_Conv1x1_Short postconv1x1s1 = {
        conv1x1s1_weight,
        conv1x1s1_bias,
        2,
        magnification_factor + 38,
        sizeof(conv1x1s1_weight) / 2,
        sizeof(conv1x1s1_bias) / 2,
    };

    NetMistouchShort net = {
        &preconv3x3s1,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
        ver_mistouch,
    };
    return net;
}

static int net_forward_mistouch(unsigned char* img, int h, int w, NetMistouchShort net, int* score)
{
    if ( img == NULL || score == NULL || h < 0 || w < 0 ) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int hc, wc;
    hc = h;
    wc = w;
    int size = hc * wc;
    // const int LEFT_STRIDE = alignPtr(6*size, MALLOC_ALIGN);       // = 6*size 对齐16的倍数

    // memory block
    float* memory = ( float* )malloc( 6 * size * sizeof( float ) );

    if ( !memory ) {
        SL_ERR_MALLOC_LOG;
        return SL_ERR_MALLOC;
    }

    // float* left = memory;
    float* tensor = ( float* )alignPtr( ( size_t )memory, MALLOC_ALIGN ); // 指针指的地址也要被16整除
    // printf("tensor = %p, %f", tensor, *tensor);
    char2float( img, tensor, hc, wc); // CenterCrop
    totensor_neon( tensor, tensor, size ); // uint8 -> double [0, 1]
    Mat input = newMat(tensor, wc, hc, 1);
    // pre
    input = preblock_short( input, net.preconv3x3s1 );

    // input = conv3x3_block(input, net.preconv3x3s2, right,1);
    // conv
    input = bottleneck_short( input, net.block1 );

    input = bottleneck_short( input, net.block2 );

    input = bottleneck_short( input, net.block3 );

    input = bottleneck_short( input, net.block4 );

    input = bottleneck_short( input, net.block5 );

    input = bottleneck_short( input, net.block6 );

    // global pooling
    float* right = tensor + alignPtr( input.c * input.cstep, MALLOC_ALIGN );

    Mat mat_avgpool = newMat( right, 1, 1, net.block6->out_channel );

    pooling_global( input, mat_avgpool, PoolMethod_AVE );
    // class
    Mat postconv1x1 = newMat( tensor, 1, 1, net.postconv1x1->out_channel );

    float* weight_f = new_memory(postconv1x1);
    float* bias_f = weight_f + net.postconv1x1->len_conv1x1_weight;
    short_to_float(weight_f, net.postconv1x1->conv1x1_weight, net.postconv1x1->len_conv1x1_weight, net.postconv1x1->magnification[0]);
    short_to_float(bias_f, net.postconv1x1->conv1x1_bias, net.postconv1x1->len_conv1x1_bias, net.postconv1x1->magnification[1]);

    conv1x1s1_neon(mat_avgpool, postconv1x1, weight_f, bias_f);

    flatten( postconv1x1, tensor );

    // 归一化

    float* soft_img = tensor + net.postconv1x1->out_channel;

    soft_max( tensor, net.postconv1x1->out_channel, soft_img );

    *score = 65536 * soft_img[0];

    if ( memory ) {
        free( memory );
    }

    return SL_RET_SUCCESS;
}
int mistouch_check( unsigned char* img, int width, int height, int* score )
{
#if BENCHMARK
    double start = get_current_time();
#endif
    get_version_mistouch();
    SL_LOGE( "mistouch_check93 init, image w = %d,h = %d", width, height );

    if ( NULL == img || width <= 0 || height <= 0 || score == NULL ) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int ret = 0;

    ret = net_forward_mistouch( img, height, width, get_para_mistouch_v1(), score );

    if ( ret < 0 ) {
        SL_LOGE( "mistouch_check93 error ret %d", ret );
    }

    SL_LOGE( "mistouch_check93 out score = %d", *score );
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}
