#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "SL_Math.h"

#include "net_api.h"
#include "net_param/para_patch.h"


#include "../net_cnn_common.h"
#include "../net_struct_common.h"
#include "../alog.h"

typedef struct {
    Block_Bneck_SE_Short_Desc* block_patch_block1;
    Block_Bneck_SE_Short_Desc* block_patch_block2;
    Block_Bneck_SE_Short_Desc* block_patch_block3;
    Block_Bneck_SE_Short_Desc* block_patch_block4;
    Block_Bneck_SE_Short_Desc* block_patch_block5;
    Block_Bneck_SE_Short_Desc* block_patch_block6;
    Block_Conv1x1_Short_Desc* block_patch_postconv1x1;
    int* scale_patch;
    char* ver_patch;
} NetPatch;


static NetPatch get_para_patch_v1()
{
    static char* ver_patch = para_ver_patch;

    static Block_Bneck_SE_Short_Desc block_patch_block1 = {
        para_patch_conv1x1s1_di_1_weight,
        para_patch_conv1x1s1_di_1_bias,
        para_patch_convdw3x3s2_1_weight,
        para_patch_convdw3x3s2_1_bias,
        para_patch_conv1x1s1_dd_1_weight,
        para_patch_conv1x1s1_dd_1_bias,
        para_patch_conv1x1s1_dd_se_1_weight,
        para_patch_conv1x1s1_dd_se_1_bias,
        para_patch_conv1x1s1_di_se_1_weight,
        para_patch_conv1x1s1_di_se_1_bias,
        2,
        1,
        8,
        8,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block2 = {
        para_patch_conv1x1s1_di_2_weight,
        para_patch_conv1x1s1_di_2_bias,
        para_patch_convdw3x3s2_2_weight,
        para_patch_convdw3x3s2_2_bias,
        para_patch_conv1x1s1_dd_2_weight,
        para_patch_conv1x1s1_dd_2_bias,
        para_patch_conv1x1s1_dd_se_2_weight,
        para_patch_conv1x1s1_dd_se_2_bias,
        para_patch_conv1x1s1_di_se_2_weight,
        para_patch_conv1x1s1_di_se_2_bias,
        2,
        1,
        16,
        2,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block3 = {
        para_patch_conv1x1s1_di_3_weight,
        para_patch_conv1x1s1_di_3_bias,
        para_patch_convdw3x3s1_3_weight,
        para_patch_convdw3x3s1_3_bias,
        para_patch_conv1x1s1_dd_3_weight,
        para_patch_conv1x1s1_dd_3_bias,
        para_patch_conv1x1s1_dd_se_3_weight,
        para_patch_conv1x1s1_dd_se_3_bias,
        para_patch_conv1x1s1_di_se_3_weight,
        para_patch_conv1x1s1_di_se_3_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block4 = {
        para_patch_conv1x1s1_di_4_weight,
        para_patch_conv1x1s1_di_4_bias,
        para_patch_convdw3x3s1_4_weight,
        para_patch_convdw3x3s1_4_bias,
        para_patch_conv1x1s1_dd_4_weight,
        para_patch_conv1x1s1_dd_4_bias,
        para_patch_conv1x1s1_dd_se_4_weight,
        para_patch_conv1x1s1_dd_se_4_bias,
        para_patch_conv1x1s1_di_se_4_weight,
        para_patch_conv1x1s1_di_se_4_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block5 = {
        para_patch_conv1x1s1_di_5_weight,
        para_patch_conv1x1s1_di_5_bias,
        para_patch_convdw3x3s1_5_weight,
        para_patch_convdw3x3s1_5_bias,
        para_patch_conv1x1s1_dd_5_weight,
        para_patch_conv1x1s1_dd_5_bias,
        para_patch_conv1x1s1_dd_se_5_weight,
        para_patch_conv1x1s1_dd_se_5_bias,
        para_patch_conv1x1s1_di_se_5_weight,
        para_patch_conv1x1s1_di_se_5_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block6 = {
        para_patch_conv1x1s1_di_6_weight,
        para_patch_conv1x1s1_di_6_bias,
        para_patch_convdw3x3s1_6_weight,
        para_patch_convdw3x3s1_6_bias,
        para_patch_conv1x1s1_dd_6_weight,
        para_patch_conv1x1s1_dd_6_bias,
        para_patch_conv1x1s1_dd_se_6_weight,
        para_patch_conv1x1s1_dd_se_6_bias,
        para_patch_conv1x1s1_di_se_6_weight,
        para_patch_conv1x1s1_di_se_6_bias,
        1,
        1,
        32,
        2,
    };

    static Block_Conv1x1_Short_Desc block_patch_postconv1x1 = {
        para_patch_conv1x1_post_weight,
        para_patch_conv1x1_post_bias,
        16,
    };

    NetPatch net = {
        &block_patch_block1,
        &block_patch_block2,
        &block_patch_block3,
        &block_patch_block4,
        &block_patch_block5,
        &block_patch_block6,
        &block_patch_postconv1x1,
        para_patch_weight_scale,
        ver_patch,
    };
    return net;
}

static NetPatch get_para_patch_rect_v1()
{
    static char* ver_patch = para_ver_patch;

    static Block_Bneck_SE_Short_Desc block_patch_block1 = {
        rectDes_conv1x1s1_di_1_weight,
        rectDes_conv1x1s1_di_1_bias,
        rectDes_convdw3x3s2_1_weight,
        rectDes_convdw3x3s2_1_bias,
        rectDes_conv1x1s1_dd_1_weight,
        rectDes_conv1x1s1_dd_1_bias,
        rectDes_conv1x1s1_dd_se_1_weight,
        rectDes_conv1x1s1_dd_se_1_bias,
        rectDes_conv1x1s1_di_se_1_weight,
        rectDes_conv1x1s1_di_se_1_bias,
        2,
        1,
        8,
        8,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block2 = {
        rectDes_conv1x1s1_di_2_weight,
        rectDes_conv1x1s1_di_2_bias,
        rectDes_convdw3x3s2_2_weight,
        rectDes_convdw3x3s2_2_bias,
        rectDes_conv1x1s1_dd_2_weight,
        rectDes_conv1x1s1_dd_2_bias,
        rectDes_conv1x1s1_dd_se_2_weight,
        rectDes_conv1x1s1_dd_se_2_bias,
        rectDes_conv1x1s1_di_se_2_weight,
        rectDes_conv1x1s1_di_se_2_bias,
        2,
        1,
        16,
        2,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block3 = {
        rectDes_conv1x1s1_di_3_weight,
        rectDes_conv1x1s1_di_3_bias,
        rectDes_convdw3x3s1_3_weight,
        rectDes_convdw3x3s1_3_bias,
        rectDes_conv1x1s1_dd_3_weight,
        rectDes_conv1x1s1_dd_3_bias,
        rectDes_conv1x1s1_dd_se_3_weight,
        rectDes_conv1x1s1_dd_se_3_bias,
        rectDes_conv1x1s1_di_se_3_weight,
        rectDes_conv1x1s1_di_se_3_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block4 = {
        rectDes_conv1x1s1_di_4_weight,
        rectDes_conv1x1s1_di_4_bias,
        rectDes_convdw3x3s1_4_weight,
        rectDes_convdw3x3s1_4_bias,
        rectDes_conv1x1s1_dd_4_weight,
        rectDes_conv1x1s1_dd_4_bias,
        rectDes_conv1x1s1_dd_se_4_weight,
        rectDes_conv1x1s1_dd_se_4_bias,
        rectDes_conv1x1s1_di_se_4_weight,
        rectDes_conv1x1s1_di_se_4_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block5 = {
        rectDes_conv1x1s1_di_5_weight,
        rectDes_conv1x1s1_di_5_bias,
        rectDes_convdw3x3s1_5_weight,
        rectDes_convdw3x3s1_5_bias,
        rectDes_conv1x1s1_dd_5_weight,
        rectDes_conv1x1s1_dd_5_bias,
        rectDes_conv1x1s1_dd_se_5_weight,
        rectDes_conv1x1s1_dd_se_5_bias,
        rectDes_conv1x1s1_di_se_5_weight,
        rectDes_conv1x1s1_di_se_5_bias,
        1,
        1,
        16,
        1,
    };

    static Block_Bneck_SE_Short_Desc block_patch_block6 = {
        rectDes_conv1x1s1_di_6_weight,
        rectDes_conv1x1s1_di_6_bias,
        rectDes_convdw3x3s1_6_weight,
        rectDes_convdw3x3s1_6_bias,
        rectDes_conv1x1s1_dd_6_weight,
        rectDes_conv1x1s1_dd_6_bias,
        rectDes_conv1x1s1_dd_se_6_weight,
        rectDes_conv1x1s1_dd_se_6_bias,
        rectDes_conv1x1s1_di_se_6_weight,
        rectDes_conv1x1s1_di_se_6_bias,
        1,
        1,
        32,
        2,
    };

    static Block_Conv1x1_Short_Desc block_patch_postconv1x1 = {
        rectDes_conv1x1_post_weight,
        rectDes_conv1x1_post_bias,
        8,
    };

    NetPatch net = {
        &block_patch_block1,
        &block_patch_block2,
        &block_patch_block3,
        &block_patch_block4,
        &block_patch_block5,
        &block_patch_block6,
        &block_patch_postconv1x1,
        rectDes_weight_scale,
        ver_patch,
    };
    return net;
}


static void padding_onlycpy(const Mat bottom_blob, Mat top_blob, int top, int left, int type, float v)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int bottom = top;
    int outh = h + top + bottom;
    int q, i;

    for (q = 0; q < channels; q++) {
        // const Mat m = bottom_blob.channel(q);
        // Mat borderm = top_blob.channel(q);
        for (i = 0; i < h; i++) {
            float* bottom_src = bottom_blob.data + q * bottom_blob.cstep + i * w;
            float* top_src = top_blob.data + q * top_blob.cstep + (i + top) * top_blob.w + left;

            memcpy(top_src, bottom_src, w * sizeof(float));
        }
    }
}
// net desc patch
//  BottleneckSeBlock
Mat bottleneck_desc_short(Mat input, Block_Bneck_SE_Short_Desc* block, int count, float** weight_p, int* scale)
{
    /*
    memmory consuming:
    input: x, expand ratio: t
    memory = x + t*x + t*x + y (y is tiny)
    */
    // init
    const int t = block->ratio; // ratio
    // int t = 4;
    // conv1x1
    // conv1x1s1_neon
    int h = input.h, w = input.w, c = input.c * t, stride = block->stride;

    Mat mat_conv1x1_di = newMat(new_memory(input), w, h, c);

    int weight_len;
    weight_len = c * input.c;

    if (count == 0) {
        short_to_float(*weight_p, block->conv1x1_di_weight, weight_len, scale[0]);
    }

    conv1x1s1_neon(input, mat_conv1x1_di, *weight_p, block->conv1x1_di_bias);
    hswish_neon(mat_conv1x1_di);

    // conv3x3
    // padding
    // convdw3x3s1_neon or convdw3x3s2_neon
    Mat mat_conv3x3padding = newMat(new_memory(mat_conv1x1_di), w + 2 * block->padding, h + 2 * block->padding, c);
    memset(mat_conv3x3padding.data, 0, total(mat_conv3x3padding) * sizeof(float));
    padding_onlycpy(mat_conv1x1_di, mat_conv3x3padding, block->padding, block->padding, 0, 0);
    w = (int)((w + 2 * block->padding - 3) / stride) + 1, h = (int)((h + 2 * block->padding - 3) / stride) + 1;
    Mat mat_conv3x3 = newMat(mat_conv1x1_di.data, w, h, c); // 回到mat_conv1x1_di.data指针，节省内存

    *weight_p += weight_len;
    weight_len = c * 9;

    if (count == 0) {
        short_to_float(*weight_p, block->convdw3x3_weight, weight_len, scale[1]);
    }

    if (stride == 1) {
        convdw3x3s1_neon(mat_conv3x3padding, mat_conv3x3, *weight_p, block->convdw3x3_bias);
    }

    if (stride == 2) {
        convdw3x3s2_neon(mat_conv3x3padding, mat_conv3x3, *weight_p, block->convdw3x3_bias);
    }

    hswish_neon(mat_conv3x3);

    // conv1x1
    // conv1x1s1_neon
    c = block->out_channel;
    Mat mat_conv1x1_dd = newMat(new_memory(mat_conv3x3), w, h, c);

    *weight_p += weight_len;
    weight_len = c * mat_conv3x3.c;

    if (count == 0) {
        short_to_float(*weight_p, block->conv1x1_dd_weight, weight_len, scale[2]);
    }

    conv1x1s1_neon(mat_conv3x3, mat_conv1x1_dd, *weight_p, block->conv1x1_dd_bias);

    // se
    // pooling_global
    // conv1x1s1_neon
    // conv1x1s1_neon
    Mat mat_global_pooling = newMat(new_memory(mat_conv1x1_dd), 1, 1, c);
    pooling_global(mat_conv1x1_dd, mat_global_pooling, PoolMethod_AVE);
    c /= 4;
    Mat mat_conv1x1dd_se = newMat(new_memory(mat_global_pooling), 1, 1, c);

    *weight_p += weight_len;
    weight_len = c * mat_global_pooling.c;

    if (count == 0) {
        short_to_float(*weight_p, block->conv1x1_dd_se_weight, weight_len, scale[3]);
    }

    conv1x1s1_neon(mat_global_pooling, mat_conv1x1dd_se, *weight_p, block->conv1x1_dd_se_bias);
    relu_neon(mat_conv1x1dd_se);
    c = block->out_channel;
    Mat mat_conv1x1di_se = newMat(new_memory(mat_conv1x1dd_se), 1, 1, c);

    *weight_p += weight_len;
    weight_len = c * mat_conv1x1dd_se.c;

    if (count == 0) {
        short_to_float(*weight_p, block->conv1x1_di_se_weight, weight_len, scale[4]);
    }

    conv1x1s1_neon(mat_conv1x1dd_se, mat_conv1x1di_se, *weight_p, block->conv1x1_di_se_bias);
    hsigmoid_neon(mat_conv1x1di_se);

    mat_scale_neon_inplace(mat_conv1x1_dd, mat_conv1x1di_se);

    *weight_p += weight_len;
    // shortcut

    if (stride == 1 && input.c == mat_conv1x1_dd.c) {
        mat_add_neon_inplace(input, mat_conv1x1_dd);
        return input;
    }

    Mat output = newMat(input.data, w, h, c);
    memcpy(output.data, mat_conv1x1_dd.data, mat_conv1x1_dd.cstep * c * sizeof(float));

    return output;
}

//static void transpose_flatten(Mat mat, float* data)
//{
//    int inch = mat.c;
//    int w = mat.w;
//    int h = mat.h;
//    int size = w * h;
//    int i, j;
//
//    for (i = 0; i < size; i++) {
//        for (j = 0; j < inch; j++) {
//            data[i * inch + j] = mat.data[j * mat.cstep + i];
//        }
//    }
//}
static void transpose_flatten(Mat mat, float* data)
{
	int inch = mat.c;
	int w = mat.w;
	int h = mat.h;
	int size = w * h;
	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < inch; j++)
		{
			data[i * inch + j] = mat.data[j * mat.cstep + i];
		}
	}
}

static void transpose_flatten_256(Mat mat, float* data)
{
    //inch必须为16
    int inch = mat.c / 2;
    int w = mat.w;
    int h = mat.h;
    int size = w * h;
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < inch; j++) {
            data[i * inch + j] = mat.data[j * mat.cstep + i];
            data[i * inch + j + inch * size] = mat.data[(j + inch) * mat.cstep + i];
        }
    }
}

// forward
static int net_forward_patch_short(float* patch, int patch_size, uint32_t* desc, NetPatch net, int* call_count_weight)
{
    int size = patch_size * patch_size;
    // float mean, std;
    int descriptor_dim = 256;
    totensor_neon(patch, patch, size);

    ////normalization
    // mean = mean_neon(patch, size);
    // std = std_neon(patch, mean, size);
    // std = sqrt(std);
    // normalize_neon(patch, mean, std, size);

    Mat input = newMat(patch, patch_size, patch_size, 1);

    int call_count = call_count_weight[0];
    float* weight_f = (float*)(call_count_weight + 1);
    ////pre
    // input = conv3x3_block(input, net.block_patch_preconv3x3, new_memory(input), 0);


    //bottleneck
    input = bottleneck_desc_short(input, net.block_patch_block1, call_count, &weight_f, net.scale_patch);
    input = bottleneck_desc_short(input, net.block_patch_block2, call_count, &weight_f, net.scale_patch + 5);
    input = bottleneck_desc_short(input, net.block_patch_block3, call_count, &weight_f, net.scale_patch + 10);
    input = bottleneck_desc_short(input, net.block_patch_block4, call_count, &weight_f, net.scale_patch + 15);
    input = bottleneck_desc_short(input, net.block_patch_block5, call_count, &weight_f, net.scale_patch + 20);
    input = bottleneck_desc_short(input, net.block_patch_block6, call_count, &weight_f, net.scale_patch + 25);

    ////global pooling
    // Mat mat_avgpool = newMat(new_memory(input), 1, 1, net.block_patch_block3->out_channel);
    // pooling_global(input, mat_avgpool, PoolMethod_AVE);

    // post
    // Mat postconv1x1 = newMat(new_memory(mat_avgpool), 1, 1, net.block_patch_postconv1x1->out_channel);
    // conv1x1s1_neon(mat_avgpool, postconv1x1, net.block_patch_postconv1x1->conv1x1_weight, net.block_patch_postconv1x1->conv1x1_bias);

    Mat postconv1x1 = newMat(new_memory(input), input.w, input.h, net.block_patch_postconv1x1->out_channel);

    if (call_count == 0) {
        short_to_float(weight_f, net.block_patch_postconv1x1->conv1x1_weight, (postconv1x1.c * input.c), net.scale_patch[30]);
    }

    conv1x1s1_neon(input, postconv1x1, weight_f, net.block_patch_postconv1x1->conv1x1_bias);

    transpose_flatten_256(postconv1x1, patch);

    // L2 norm
    float L2_norm = std_neon(patch, 0, descriptor_dim) * (descriptor_dim - 1);
    L2_norm = SL_sqrt(L2_norm);
    normalize_neon(patch, patch, 0, L2_norm, descriptor_dim);

    ////binarization
    // descriptor_Hamming(patch, desc, descriptor_dim);

    memcpy(desc, patch, descriptor_dim * sizeof(float));
    // memcpy(desc + descriptor_dim, patch, descriptor_dim * sizeof(float));  //输出128维描述子，故复制两份

    return SL_RET_SUCCESS;
}

static int net_forward_patch_short_rect(float* patch, int *patch_size, uint32_t* desc, NetPatch net, int* call_count_weight)
{
    int size = patch_size[0] * patch_size[1];
    // float mean, std;
    int descriptor_dim = 128;
    totensor_neon(patch, patch, size);

    ////normalization
    // mean = mean_neon(patch, size);
    // std = std_neon(patch, mean, size);
    // std = sqrt(std);
    // normalize_neon(patch, mean, std, size);

    Mat input = newMat(patch, patch_size[1], patch_size[0], 1);

    int call_count = call_count_weight[0];
    float* weight_f = (float*)(call_count_weight + 1);
    ////pre
    // input = conv3x3_block(input, net.block_patch_preconv3x3, new_memory(input), 0);


    //bottleneck
    input = bottleneck_desc_short(input, net.block_patch_block1, call_count, &weight_f, net.scale_patch);
    input = bottleneck_desc_short(input, net.block_patch_block2, call_count, &weight_f, net.scale_patch + 5);
    input = bottleneck_desc_short(input, net.block_patch_block3, call_count, &weight_f, net.scale_patch + 10);
    input = bottleneck_desc_short(input, net.block_patch_block4, call_count, &weight_f, net.scale_patch + 15);
    input = bottleneck_desc_short(input, net.block_patch_block5, call_count, &weight_f, net.scale_patch + 20);
    input = bottleneck_desc_short(input, net.block_patch_block6, call_count, &weight_f, net.scale_patch + 25);

    ////global pooling
    // Mat mat_avgpool = newMat(new_memory(input), 1, 1, net.block_patch_block3->out_channel);
    // pooling_global(input, mat_avgpool, PoolMethod_AVE);

    // post
    // Mat postconv1x1 = newMat(new_memory(mat_avgpool), 1, 1, net.block_patch_postconv1x1->out_channel);
    // conv1x1s1_neon(mat_avgpool, postconv1x1, net.block_patch_postconv1x1->conv1x1_weight, net.block_patch_postconv1x1->conv1x1_bias);

    Mat postconv1x1 = newMat(new_memory(input), input.w, input.h, net.block_patch_postconv1x1->out_channel);

    if (call_count == 0) {
        short_to_float(weight_f, net.block_patch_postconv1x1->conv1x1_weight, (postconv1x1.c * input.c), net.scale_patch[30]);
    }

    conv1x1s1_neon(input, postconv1x1, weight_f, net.block_patch_postconv1x1->conv1x1_bias);

    transpose_flatten(postconv1x1, patch);

    // L2 norm
    float L2_norm = std_neon(patch, 0, descriptor_dim) * (descriptor_dim - 1);
    L2_norm = SL_sqrt(L2_norm);
    normalize_neon(patch, patch, 0, L2_norm, descriptor_dim);

    ////binarization
    // descriptor_Hamming(patch, desc, descriptor_dim);

    memcpy(desc, patch, descriptor_dim * sizeof(float));
    // memcpy(desc + descriptor_dim, patch, descriptor_dim * sizeof(float));  //输出128维描述子，故复制两份

    return SL_RET_SUCCESS;
}

int descriptor_net_patch_short(unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int patch_size, int sample_size, uint32_t* desc, void* net_memory)
{
#if BENCHMARK
    double start = get_current_time();
#endif
    int ret = 0;

    // int i, j;
    // int size;
    // size = (32 + 64 + 64 + 25) * patch_size * patch_size;
    ////memory block
    // void* memory = (void*)malloc(size * sizeof(float));
    // if (!memory)
    //    return -1;

    int* call_count_weight = (int*)net_memory;

    //18664 save float weight
    float* tensor = (float*)alignPtr((size_t)((float*)net_memory + 18664 + 2), MALLOC_ALIGN); //16位对齐

    MatImg image_warp = newMat(ucImage, width, height, 1);
    Mat patch_warp = newMat(tensor, 16, 16, 1);

    float theta = (float)ori / 4096;
    theta = theta - 1.570796327; // theta-pi/2 梯度方向变为指纹方向

    bilinear_warp_neon(image_warp, patch_warp, theta, (float)nX, (float)nY, patch_size, sample_size);

    ret = net_forward_patch_short(tensor, patch_size, desc, get_para_patch_v1(), call_count_weight);
    call_count_weight[0] += 1;

    // free(memory);
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}

int descriptor_net_patch_short_rect(unsigned char* ucImage, int height, int width, int32_t nX, int32_t nY, int16_t ori, int *patch_size, int *sample_size, uint32_t* desc, void* net_memory)
{
#if BENCHMARK
    double start = get_current_time();
#endif
    int ret = 0;

    // int i, j;
    // int size;
    // size = (32 + 64 + 64 + 25) * patch_size * patch_size;
    ////memory block
    // void* memory = (void*)malloc(size * sizeof(float));
    // if (!memory)
    //    return -1;

    int* call_count_weight = (int*)net_memory;

    //18664 save float weight
    float* tensor = (float*)alignPtr((size_t)((float*)net_memory + 18664 + 2), MALLOC_ALIGN); //16位对齐

    MatImg image_warp = newMat(ucImage, width, height, 1);
    Mat patch_warp = newMat(tensor, 16, 32, 1);

    float theta = (float)ori / 4096;
    theta = theta - 1.570796327; // theta-pi/2 梯度方向变为指纹方向

    bilinear_warp_neon_rect(image_warp, patch_warp, theta, (float)nX, (float)nY, patch_size, sample_size);


    ret = net_forward_patch_short_rect(tensor, patch_size, desc, get_para_patch_rect_v1(), call_count_weight);
    call_count_weight[0] += 1;

    // free(memory);
#if BENCHMARK
    double end = get_current_time();
    benchmark(__func__, start, end);
#endif
    return ret;
}

int descriptor_net_init_patch_short(int patch_size, void** net_memory)
{
    get_version_patch();

    int size;
    size = (22 + 76) * patch_size * patch_size;
    // //memory block
    void* memory = (int*)malloc(size * sizeof(float));

    if (memory != NULL) {
        *net_memory = memory;
        int* call_count_weight = (int*)memory;
        call_count_weight[0] = 0; //初始化，此时还未运行

        return SL_RET_SUCCESS;
    }

    return SL_RET_FAIL;
}

int descriptor_net_init_patch_short_rect(int *patch_size, void** net_memory)
{
    get_version_patch();

    int size;
    size = (22 + 76) * patch_size[0] * patch_size[1];
    // //memory block
    void* memory = (int*)malloc(size * sizeof(float));

    if (memory != NULL) {
        *net_memory = memory;
        int* call_count_weight = (int*)memory;
        call_count_weight[0] = 0; //初始化，此时还未运行

        return SL_RET_SUCCESS;
    }

    return SL_RET_FAIL;
}

int descriptor_net_deinit_patch(void* net_memory)
{
    if (net_memory != NULL) {
        free(net_memory);
    }

    return SL_RET_SUCCESS;
}
