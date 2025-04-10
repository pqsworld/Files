#include "net_api.h"
#include "net_param/para_spoof.h"
#include "../net_struct_common.h"
#include "../net_cnn_common.h"

#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "../alog.h"
#include "SL_Math.h"
#define TEMPLATE_FEATURE_NUM 20

typedef struct {
    Block_Conv3x3* preconv3x3s1;
    Block_Bn* block1;
    Block_Bn* block2;
    Block_Bn* block3;
    Block_Bn* block4;
    Block_Bn* block5;
    Block_Bn* block6;
    Block_Conv1x1* postconv1x1;
} NetSpoof;
// beta = this.offset.value,
// gamma=this.scale.value,eps=this.epsilon,
// mean=this.trainedmeancache.value,var=this.trainedvariancecache.value
//#define OP_GEN2_ART_RAW_JIANBIAN

static NetSpoof get_para_spoof_v1()
{
    static Block_Conv3x3 preconv3x3s1 = {
        conv3x3_weight,
        conv3x3_bias,
        1,
        1,
        8,
    };

    static Block_Bn block1 = {
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
    };

    static Block_Bn block2 = {
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
    };

    static Block_Bn block3 = {
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
    };

    static Block_Bn block4 = {
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
    };

    static Block_Bn block5 = {
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
    };

    static Block_Bn block6 = {
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
    };
    static Block_Conv1x1 postconv1x1s1 = {
        conv1x1s1_weight,
        conv1x1s1_bias,
        2,
    };

    NetSpoof net = {
        &preconv3x3s1,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
    };
    return net;
}
static NetSpoof get_para_spoof_v1_alipay()
{
    static Block_Conv3x3 preconv3x3s1 = {
        conv3x3_weight_alipay,
        conv3x3_bias_alipay,
        1,
        1,
        8,
    };

    static Block_Bn block1 = {
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
    };

    static Block_Bn block2 = {
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
    };

    static Block_Bn block3 = {
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
    };

    static Block_Bn block4 = {
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
    };

    static Block_Bn block5 = {
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
    };

    static Block_Bn block6 = {
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
    };
    static Block_Conv1x1 postconv1x1s1 = {
        conv1x1s1_weight_alipay,
        conv1x1s1_bias_alipay,
        2,
    };

    NetSpoof net = {
        &preconv3x3s1,
        &block1,
        &block2,
        &block3,
        &block4,
        &block5,
        &block6,
        &postconv1x1s1,
    };
    return net;
}

/*Mobile Net Spoof*/
static Mat bottleneck( Mat input, Block_Bn* block )
{
    // init
    const int t = block->ratio; // ratio
    int temp = 0;
    // conv1x1
    // conv1x1s1_neon
    int h = input.h, w = input.w, c = input.c * t, stride = block->stride;

    Mat mat_conv1x1_di = newMat(new_memory(input), w, h, c);
    temp += total( input );
    conv1x1s1_neon( input, mat_conv1x1_di, block->conv1x1_di_weight, block->conv1x1_di_bias );
    hswish_neon( mat_conv1x1_di );

    // conv3x3
    // padding
    // convdw3x3s1_neon or convdw3x3s2_neon
    Mat mat_conv3x3padding = newMat(new_memory(mat_conv1x1_di), w + 2 * block->padding, h + 2 * block->padding, c);
    // Mat mat_conv3x3padding = newMat(mat_conv1x1_di.data, h + 2 * block->padding, w + 2 * block->padding, c);

    if ( stride == 2 ) {
        mat_conv3x3padding.data = input.data;
        int jump = mat_conv3x3padding.cstep * mat_conv3x3padding.c - mat_conv1x1_di.cstep * mat_conv1x1_di.c + mat_conv1x1_di.cstep;
        float* data = input.data + alignPtr( jump, MALLOC_ALIGN );
        memmove( data, mat_conv1x1_di.data, mat_conv1x1_di.cstep * c * sizeof( float ) );
        mat_conv1x1_di.data = data;
    }

    temp += total( mat_conv1x1_di );
    // memset(mat_conv3x3padding.data, 0, total(mat_conv3x3padding) * sizeof(float));
    padding( mat_conv1x1_di, mat_conv3x3padding, block->padding, block->padding, 0, 0 );
    w = ( w + 2 * block->padding - 3 ) / stride + 1, h = ( h + 2 * block->padding - 3 ) / stride + 1;
    Mat mat_conv3x3 = newMat(mat_conv1x1_di.data, w, h, c); //回到mat_conv1x1_di.data指针，节省内存,因为输出通道数是一样的

    // Mat mat_conv3x3 = newMat(new_memory(mat_conv3x3padding), h, w, c); //回到mat_conv1x1_di.data指针，节省内存,因为输出通道数是一样的
    if ( stride == 2 ) {
        mat_conv3x3.data = new_memory( mat_conv3x3padding );
    }

    if ( stride == 1 ) {
        convdw3x3s1_neon( mat_conv3x3padding, mat_conv3x3, block->convdw3x3_weight, block->convdw3x3_bias );
    } else if ( stride == 2 ) {
        convdw3x3s2_neon( mat_conv3x3padding, mat_conv3x3, block->convdw3x3_weight, block->convdw3x3_bias );
    }

    hswish_neon( mat_conv3x3 );

    // conv1x1
    // conv1x1s1_neon
    c = block->out_channel;
    Mat mat_conv1x1_dd = newMat(new_memory(mat_conv3x3), w, h, c);

    // Mat mat_conv1x1_dd = newMat(input.data, h, w, c);
    if ( stride == 2 ) {
        mat_conv1x1_dd.data = input.data;
    }

    temp += total( mat_conv3x3 );
    conv1x1s1_neon( mat_conv3x3, mat_conv1x1_dd, block->conv1x1_dd_weight, block->conv1x1_dd_bias );

    // se
    // pooling_global
    // conv1x1s1_neon
    // conv1x1s1_neon
    /*
        Mat mat_global_pooling = newMat(new_memory(mat_conv1x1_dd), 1, 1, c);
        temp += total(mat_conv1x1_dd);
        pooling_global(mat_conv1x1_dd, mat_global_pooling, PoolMethod_AVE);
        c /= 4;
        Mat mat_conv1x1dd_se = newMat(new_memory(mat_global_pooling), 1, 1, c);
        temp += total(mat_global_pooling);
        conv1x1s1_neon(mat_global_pooling, mat_conv1x1dd_se, block->conv1x1_dd_se_weight, block->conv1x1_dd_se_bias);
        relu_neon(mat_conv1x1dd_se);
        c = block->out_channel;
        Mat mat_conv1x1di_se = newMat(new_memory(mat_conv1x1dd_se), 1, 1, c);
        temp += total(mat_conv1x1dd_se);
        //printf("\ntemp:%d\n",temp);
        conv1x1s1_neon(mat_conv1x1dd_se, mat_conv1x1di_se, block->conv1x1_di_se_weight, block->conv1x1_di_se_bias);
        hsigmoid_neon(mat_conv1x1di_se);

        mat_scale_neon_inplace(mat_conv1x1_dd, mat_conv1x1di_se);
    */
    // shortcut
    /*
        if (stride == 1)
        {
            mat_add_neon_inplace(input, mat_conv1x1_dd);
            return input;
        }
    */

    Mat output = newMat(input.data, w, h, c);
    memcpy( output.data, mat_conv1x1_dd.data, mat_conv1x1_dd.cstep * c * sizeof( float ) );
    return output;
}
static int char2float( unsigned char* src, float* dst, int h, int w, const int cut ) // int2float and cut
{
    int i, j;

    for ( i = 0; i < h; i++ ) {
        for ( j = 0; j < w; j++ ) {
            dst[i * w + j] = ( float )src[( i + cut ) * ( w + 2 * cut ) + j + cut];
        }

        // printf("dst[%d]=%d\n", i*w + 90, dst[i*w + 90]);
    }

    return 0;
}
static Mat preblock( Mat input, Block_Conv3x3* block )
{
    int w = input.w, h = input.h, c = input.c; //, stride = block->stride;
    float* left = input.data;
    int outh = ( int )( ( h + 2 * block->padding - 3 ) / block->stride ) + 1;
    int outw = ( int )( ( w + 2 * block->padding - 3 ) / block->stride ) + 1;
    int out_size = block->out_channel * alignPtr( outh * outw, MALLOC_ALIGN );
    int right_stride = total( input );

    if ( out_size > right_stride ) {
        // if stride is equal 2,out_size may be less than total(input)
        right_stride = out_size;
    }

    float* right = left + right_stride;
    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, c);
    memset( right, 0, total( mat_conv3x3padding ) * sizeof( float ) );
    padding( input, mat_conv3x3padding, block->padding, block->padding, 0, 0 );
    c = block->out_channel;
    Mat mat_conv3x3s1 = newMat(left, outw, outh, c);
    conv3x3s1_neon( mat_conv3x3padding, mat_conv3x3s1, block->conv3x3_weight, block->conv3x3_bias );
    relu_neon( mat_conv3x3s1 );
    // leakyrelu_neon(mat_conv3x3s2,0.1);
    w /= 2, h /= 2;
    Mat output = newMat(left, w, h, c);
    pooling2x2s2_max_neon( mat_conv3x3s1, output );
    return output;
}

static int net_forward_spoof( unsigned char* img, int h, int w, NetSpoof net, int* score )
{
    if ( img == NULL || score == NULL || h < 0 || w < 0 ) {
        return -1;
    }

    const int cut = 0;
    int hc, wc;
    hc = h - 2 * cut;
    wc = w - 2 * cut;
    int size = hc * wc;
    // const int LEFT_STRIDE = alignPtr(6*size, MALLOC_ALIGN);       // = 6*size 对齐16的倍数

    // memory block
    float* memory = ( float* )malloc( 12 * size * sizeof( float ) );

    if ( !memory ) {
        return -1;
    }

    // float* left = memory;
    float* tensor = ( float* )alignPtr( ( size_t )memory, MALLOC_ALIGN ); // 指针指的地址也要被16整除
    // printf("tensor = %p, %f", tensor, *tensor);
    char2float( img, tensor, hc, wc, cut ); // CenterCrop
    totensor_neon( tensor, tensor, size ); // uint8 -> double [0, 1]
    Mat input = newMat(tensor, wc, hc, 1);
    // pre
    input = preblock( input, net.preconv3x3s1 );

    // input = conv3x3_block(input, net.preconv3x3s2, right,1);
    // conv
    input = bottleneck( input, net.block1 );

    input = bottleneck( input, net.block2 );

    input = bottleneck( input, net.block3 );

    input = bottleneck( input, net.block4 );

    input = bottleneck( input, net.block5 );

    input = bottleneck( input, net.block6 );

    // global pooling
    float* right = tensor + alignPtr( input.c * input.cstep, MALLOC_ALIGN );

    Mat mat_avgpool = newMat( right, 1, 1, net.block6->out_channel );

    pooling_global( input, mat_avgpool, PoolMethod_AVE );
    // class
    Mat postconv1x1 = newMat( tensor, 1, 1, net.postconv1x1->out_channel );

    conv1x1s1_neon( mat_avgpool, postconv1x1, net.postconv1x1->conv1x1_weight, net.postconv1x1->conv1x1_bias );

    flatten( postconv1x1, tensor );

    //归一化

    float* soft_img = tensor + net.postconv1x1->out_channel;

    soft_max( tensor, net.postconv1x1->out_channel, soft_img );

    *score = 65536 * soft_img[0];

    if ( memory ) {
        free( memory );
    }

    return 0;
}
static int get_mobilenet_feature( unsigned char* img, int h, int w, NetSpoof net, float* feature )
{
    if ( img == NULL || feature == NULL || h < 0 || w < 0 ) {
        return -1;
    }

    const int cut = 0;
    int hc, wc;
    hc = h - 2 * cut;
    wc = w - 2 * cut;
    int size = hc * wc;
    // const int LEFT_STRIDE = alignPtr(6*size, MALLOC_ALIGN);       // = 6*size 对齐16的倍数

    // memory block
    float* memory = ( float* )malloc( 10 * size * sizeof( float ) );

    if ( !memory ) {
        return -1;
    }

    // float* left = memory;
    float* tensor = ( float* )alignPtr( ( size_t )memory, MALLOC_ALIGN ); // 指针指的地址也要被16整除
    // printf("tensor = %p, %f", tensor, *tensor);
    char2float( img, tensor, hc, wc, cut ); // CenterCrop
    totensor_neon( tensor, tensor, size ); // uint8 -> double [0, 1]
    Mat input = newMat(tensor, wc, hc, 1);
    // pre
    input = preblock( input, net.preconv3x3s1 );

    // input = conv3x3_block(input, net.preconv3x3s2, right,1);
    // conv
    input = bottleneck( input, net.block1 );

    input = bottleneck( input, net.block2 );

    input = bottleneck( input, net.block3 );

    input = bottleneck( input, net.block4 );

    input = bottleneck( input, net.block5 );

    input = bottleneck( input, net.block6 );

    // global pooling
    float* right = tensor + alignPtr( input.c * input.cstep, MALLOC_ALIGN );

    Mat mat_avgpool = newMat( right, 1, 1, net.block6->out_channel );

    pooling_global( input, mat_avgpool, PoolMethod_AVE );
    // class
    Mat postconv1x1 = newMat( tensor, 1, 1, net.postconv1x1->out_channel );

    conv1x1s1_neon( mat_avgpool, postconv1x1, net.postconv1x1->conv1x1_weight, net.postconv1x1->conv1x1_bias );

    flatten( postconv1x1, tensor );
    memcpy( feature, tensor, 2 * sizeof( float ) );
    //归一化

    if ( memory ) {
        free( memory );
    }

    return 0;
}
static int get_feature_distance( float* feature, float* template_feature, float* distance )
{
    if ( feature == NULL || template_feature == NULL || distance == NULL ) {
        return -1;
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

    return 0;
}
static int spoofcheck_feature_distance( float thre_distance, float* feature, float* template_feature, int* num_exceed_distance )
{
    if ( thre_distance < 0 || feature == NULL || template_feature == NULL || num_exceed_distance == NULL ) {
        return -1;
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
    return 0;
}
static int net_forward_spoof_alipay( unsigned char* src, const int width, const int height, NetSpoof net, float thre_distance, float* template_feature, int* num_exceed_thre )
{
    if ( src == NULL || template_feature == NULL || height < 0 || width < 0 ) {
        return -1;
    }

    float feature[2] = {0.0f};
    int ret = 0;
    ret = get_mobilenet_feature( src, height, width, net, feature );

    if ( ret < 0 ) {
        return ret;
    }

    ret = spoofcheck_feature_distance( thre_distance, feature, template_feature, num_exceed_thre );

    if ( ret < 0 ) {
        return ret;
    }

    return ret;
}
int spoof_check( unsigned char* src, const int width, const int height, int enable_learn, int enroll_flag, float* template_feature, int* score )
{

    get_version_spoof();

    int ret = 0;
    int num_exceed_distance = 0;
    SL_LOGE( "spoof_check91 init, image w = %d,h = %d,e_l=%d,e_f=%d", width, height, enable_learn, enroll_flag );

    if ( src == NULL || template_feature == NULL || score == NULL || height < 0 || width < 0 || enable_learn < 0 || enroll_flag < 0 ) {
        SL_LOGE( "spoof_check91 param error" );
        return -1;
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

    SL_LOGE( "spoof_check91 score=%d,num_exceed_distance=%d", *score, num_exceed_distance );
    return ret;
}
