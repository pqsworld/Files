#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_spoof.h"
#include "../block_function.h"

#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

#define TEMPLATE_FEATURE_NUM 20

typedef struct {
    Block_Conv3x3_Short* preconv3x3s1;
    Block_Bneck_Short* block1;
    Block_Bneck_Short* block2;
    Block_Bneck_Short* block3;
    Block_Bneck_Short* block4;
    Block_Bneck_Short* block5;
    Block_Bneck_Short* block6;
    Block_Conv1x1_Short* postconv1x1;
    char* ver_spoof;
} NetSpoofShort;
// beta = this.offset.value,
// gamma=this.scale.value,eps=this.epsilon,
// mean=this.trainedmeancache.value,var=this.trainedvariancecache.value
// #define OP_GEN2_ART_RAW_JIANBIAN

static NetSpoofShort get_para_spoof_v1()
{
    static char* ver_spoof = para_ver_spoof;
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

    NetSpoofShort net = {
        &preconv3x3s1,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
        ver_spoof,
    };
    return net;
}
static NetSpoofShort get_para_spoof_v1_alipay()
{
    static char* ver_spoof = para_ver_spoof;
    static Block_Conv3x3_Short preconv3x3s1 = {
        conv3x3_weight_alipay,
        conv3x3_bias_alipay,
        1,
        1,
        8,
        magnification_factor_alipay,
        sizeof(conv3x3_weight_alipay) / 2,
        sizeof(conv3x3_bias_alipay) / 2,
    };

    static Block_Bneck_Short block1 = {
        conv1x1s1_di_1_weight_alipay,
        conv1x1s1_di_1_bias_alipay,
        convdw3x3s2_1_weight_alipay,
        convdw3x3s2_1_bias_alipay,
        conv1x1s1_dd_1_weight_alipay,
        conv1x1s1_dd_1_bias_alipay,
        2,
        1,
        16,
        3,
        magnification_factor_alipay + 2,
        sizeof(conv1x1s1_di_1_weight_alipay) / 2,
        sizeof(conv1x1s1_di_1_bias_alipay) / 2,
        sizeof(convdw3x3s2_1_weight_alipay) / 2,
        sizeof(convdw3x3s2_1_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_1_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_1_bias_alipay) / 2,
    };

    static Block_Bneck_Short block2 = {
        conv1x1s1_di_2_weight_alipay,
        conv1x1s1_di_2_bias_alipay,
        convdw3x3s1_2_weight_alipay,
        convdw3x3s1_2_bias_alipay,
        conv1x1s1_dd_2_weight_alipay,
        conv1x1s1_dd_2_bias_alipay,
        1,
        1,
        16,
        3,
        magnification_factor_alipay + 8,
        sizeof(conv1x1s1_di_2_weight_alipay) / 2,
        sizeof(conv1x1s1_di_2_bias_alipay) / 2,
        sizeof(convdw3x3s1_2_weight_alipay) / 2,
        sizeof(convdw3x3s1_2_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_2_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_2_bias_alipay) / 2,
    };

    static Block_Bneck_Short block3 = {
        conv1x1s1_di_3_weight_alipay,
        conv1x1s1_di_3_bias_alipay,
        convdw3x3s2_3_weight_alipay,
        convdw3x3s2_3_bias_alipay,
        conv1x1s1_dd_3_weight_alipay,
        conv1x1s1_dd_3_bias_alipay,
        2,
        1,
        24,
        3,
        magnification_factor_alipay + 14,
        sizeof(conv1x1s1_di_3_weight_alipay) / 2,
        sizeof(conv1x1s1_di_3_bias_alipay) / 2,
        sizeof(convdw3x3s2_3_weight_alipay) / 2,
        sizeof(convdw3x3s2_3_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_3_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_3_bias_alipay) / 2,
    };

    static Block_Bneck_Short block4 = {
        conv1x1s1_di_4_weight_alipay,
        conv1x1s1_di_4_bias_alipay,
        convdw3x3s1_4_weight_alipay,
        convdw3x3s1_4_bias_alipay,
        conv1x1s1_dd_4_weight_alipay,
        conv1x1s1_dd_4_bias_alipay,
        1,
        1,
        24,
        3,
        magnification_factor_alipay + 20,
        sizeof(conv1x1s1_di_4_weight_alipay) / 2,
        sizeof(conv1x1s1_di_4_bias_alipay) / 2,
        sizeof(convdw3x3s1_4_weight_alipay) / 2,
        sizeof(convdw3x3s1_4_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_4_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_4_bias_alipay) / 2,
    };

    static Block_Bneck_Short block5 = {
        conv1x1s1_di_5_weight_alipay,
        conv1x1s1_di_5_bias_alipay,
        convdw3x3s2_5_weight_alipay,
        convdw3x3s2_5_bias_alipay,
        conv1x1s1_dd_5_weight_alipay,
        conv1x1s1_dd_5_bias_alipay,
        2,
        1,
        32,
        3,
        magnification_factor_alipay + 26,
        sizeof(conv1x1s1_di_5_weight_alipay) / 2,
        sizeof(conv1x1s1_di_5_bias_alipay) / 2,
        sizeof(convdw3x3s2_5_weight_alipay) / 2,
        sizeof(convdw3x3s2_5_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_5_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_5_bias_alipay) / 2,
    };

    static Block_Bneck_Short block6 = {
        conv1x1s1_di_6_weight_alipay,
        conv1x1s1_di_6_bias_alipay,
        convdw3x3s1_6_weight_alipay,
        convdw3x3s1_6_bias_alipay,
        conv1x1s1_dd_6_weight_alipay,
        conv1x1s1_dd_6_bias_alipay,
        1,
        1,
        32,
        4,
        magnification_factor_alipay + 32,
        sizeof(conv1x1s1_di_6_weight_alipay) / 2,
        sizeof(conv1x1s1_di_6_bias_alipay) / 2,
        sizeof(convdw3x3s1_6_weight_alipay) / 2,
        sizeof(convdw3x3s1_6_bias_alipay) / 2,
        sizeof(conv1x1s1_dd_6_weight_alipay) / 2,
        sizeof(conv1x1s1_dd_6_bias_alipay) / 2,
    };
    static Block_Conv1x1_Short postconv1x1s1 = {
        conv1x1s1_weight_alipay,
        conv1x1s1_bias_alipay,
        2,
        magnification_factor_alipay + 38,
        sizeof(conv1x1s1_weight_alipay) / 2,
        sizeof(conv1x1s1_bias_alipay) / 2,
    };

    NetSpoofShort net = {
        &preconv3x3s1,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
        ver_spoof,
    };
    return net;
}


static int net_forward_spoof( unsigned char* img, int h, int w, NetSpoofShort net, int* score )
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
static int get_mobilenet_feature( unsigned char* img, int h, int w, NetSpoofShort net, float* feature )
{
    if ( img == NULL || feature == NULL || h < 0 || w < 0 ) {
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

    Mat mat_avgpool = newMat(right, 1, 1, net.block6->out_channel );

    pooling_global( input, mat_avgpool, PoolMethod_AVE );
    // class
    Mat postconv1x1 = newMat( tensor, 1, 1, net.postconv1x1->out_channel );

    float* weight_f = new_memory(postconv1x1);
    float* bias_f = weight_f + net.postconv1x1->len_conv1x1_weight;
    short_to_float(weight_f, net.postconv1x1->conv1x1_weight, net.postconv1x1->len_conv1x1_weight, net.postconv1x1->magnification[0]);
    short_to_float(bias_f, net.postconv1x1->conv1x1_bias, net.postconv1x1->len_conv1x1_bias, net.postconv1x1->magnification[1]);

    conv1x1s1_neon(mat_avgpool, postconv1x1, weight_f, bias_f);

    flatten( postconv1x1, tensor );
    memcpy( feature, tensor, 2 * sizeof( float ) );
    // 归一化

    if ( memory ) {
        free( memory );
    }

    return SL_RET_SUCCESS;
}
static int get_feature_distance( float* feature, float* template_feature, float* distance )
{
    if ( feature == NULL || template_feature == NULL || distance == NULL ) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    float temp1 = 0;
    float temp2 = 0;
    temp1 = ( feature[0] * template_feature[0] ) + ( feature[1] * template_feature[1] );
    temp2 = SL_Sqrt( feature[0] * feature[0] + feature[1] * feature[1] ) * SL_Sqrt( template_feature[0] * template_feature[0] + template_feature[1] * template_feature[1] );

    if ( temp2 != 0 ) {
        *distance = temp1 / temp2;
    } else {
        *distance = 1;
    }

    return SL_RET_SUCCESS;
}
static int spoofcheck_feature_distance( float thre_distance, float* feature, float* template_feature, int* num_exceed_distance )
{
    if (thre_distance < 0 || feature == NULL || template_feature == NULL || num_exceed_distance == NULL) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    int ret = 0;
    int i = 0;
    int num = 0;
    float distance = 0;

    for ( ; i < TEMPLATE_FEATURE_NUM; i++ ) {
        ret = get_feature_distance( feature, template_feature + i * 2, &distance );

        if ( ret < 0 ) {
            return ret;
        }

        SL_LOGI( "tpl_f1=%d,tpl_f2=%d,distance=%d", ( int32_t )( *( template_feature + i * 2 ) * 100 ), ( int32_t )( *( template_feature + i * 2 + 1 ) * 100 ), ( int32_t )( distance * 100 ) );

        if ( distance < thre_distance ) {
            num++;
        }
    }

    SL_LOGI( "feature1=%d,feature2=%d", ( int32_t )( *( feature ) * 100 ), ( int32_t )( *( feature + 1 ) * 100 ) );
    *num_exceed_distance = num;
    return SL_RET_SUCCESS;
}
static int net_forward_spoof_alipay( unsigned char* src, const int width, const int height, NetSpoofShort net, float thre_distance, float* template_feature, int* num_exceed_thre )
{
    if ( src == NULL || template_feature == NULL || height < 0 || width < 0 ) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    float feature[2] = {0.0f};
    int ret = 0;
    ret = get_mobilenet_feature( src, height, width, net, feature );

    if ( ret < 0 ) {
        return ret;
    }

    ret = spoofcheck_feature_distance( thre_distance, feature, template_feature, num_exceed_thre );


    return ret;
}
int spoof_check( unsigned char* src, const int width, const int height, int enable_learn, int enroll_flag, float* template_feature, int* score )
{
    get_version_spoof();
#if BENCHMARK
    double start = get_current_time();
#endif

    int ret = 0;
    int num_exceed_distance = 0;
    SL_LOGE( "spoof_check93 init, image w = %d,h = %d,e_l=%d,e_f=%d", width, height, enable_learn, enroll_flag );

    if ( src == NULL || template_feature == NULL || score == NULL || height < 0 || width < 0 || enable_learn < 0 || enroll_flag < 0 ) {
        SL_ERR_PARAM_LOG;
        return SL_ERR_PARAM;
    }

    float thre_distance = 0.03;
    int ther_score = 50000;
    int thre_num_exceed_distance = 10;

    do {
        if ( enable_learn == 0 ) {
            // SL_LOGE("nolearn\n");
            ret = net_forward_spoof( src, height, width, get_para_spoof_v1(), score );

            if ( ret < 0 ) {
                break;
            }

            if ( enroll_flag == 0 ) {
                if ( *score >= ther_score ) {
                    ret = net_forward_spoof_alipay( src, width, height, get_para_spoof_v1_alipay(), thre_distance, template_feature, &num_exceed_distance );

                    if ( ret < 0 ) {
                        break;
                    }

                    if ( num_exceed_distance > thre_num_exceed_distance ) {
                        *score = 0;
                    }

                    // SL_LOGI("num_exceed_distance=%d,score=%d,lin net\n", num_exceed_distance, *score);
                }
            } else {
                ret = get_mobilenet_feature( src, height, width, get_para_spoof_v1_alipay(), template_feature );

                if ( ret < 0 ) {
                    break;
                }

                // SL_LOGE("nolearn,enroll get feature\n");
            }
        } else {
            ret = net_forward_spoof( src, height, width, get_para_spoof_v1(), score );

            if ( ret < 0 ) {
                break;
            }

            // SL_LOGE("openlearn,wang net\n");
        }
    } while ( 0 );

    SL_LOGE( "spoof_check93 score=%d,num_exceed_distance=%d", *score, num_exceed_distance );
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}
