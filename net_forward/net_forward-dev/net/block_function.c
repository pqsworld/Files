#include "block_function.h"

/*Mobile Net Spoof*/
Mat bottleneck_short(Mat input, Block_Bneck_Short* block) // 第四次优化内存   conv1x1s1_neon一次循环生成四个
{
    // init
    // ratio
    // int temp = 0;
    // conv1x1
    // conv1x1s1_neon
    int once_out_channel = 4;
    int h = input.h, w = input.w, c = input.c * block->ratio, stride = block->stride, input_channel = input.c;

    int t = c / once_out_channel;

    int jump = alignPtr( ( w + 2 * block->padding ) * ( h + 2 * block->padding ), MALLOC_ALIGN ) * once_out_channel - alignPtr( ( w ) * ( h ), MALLOC_ALIGN ) * once_out_channel;
    jump = alignPtr( jump, MALLOC_ALIGN );
    Mat mat_conv1x1_di = newMat(new_memory(input) + jump, w, h, once_out_channel);
    Mat mat_conv3x3padding = newMat(new_memory(input), w + 2 * block->padding, h + 2 * block->padding, once_out_channel);
    w = ( w + 2 * block->padding - 3 ) / stride + 1, h = ( h + 2 * block->padding - 3 ) / stride + 1;
    Mat mat_conv3x3 = newMat(new_memory(mat_conv3x3padding), w, h, once_out_channel);
    float* conv3x3_begin = mat_conv3x3.data;
    jump = total( mat_conv3x3 );
    int conv3x3_len = jump * t;
    int i = 0;

    float* weight_di_f = mat_conv3x3.data + mat_conv3x3.cstep * c;
    float* bias_di_f = weight_di_f + block->len_conv1x1_di_weight;
    float* weight_dw_f = bias_di_f + block->len_conv1x1_di_bias;
    float* bias_dw_f = weight_dw_f + block->len_convdw3x3_weight;
    short_to_float(weight_di_f, block->conv1x1_di_weight, block->len_conv1x1_di_weight, block->magnification[0]);
    short_to_float(bias_di_f, block->conv1x1_di_bias, block->len_conv1x1_di_bias, block->magnification[1]);
    short_to_float(weight_dw_f, block->convdw3x3_weight, block->len_convdw3x3_weight, block->magnification[2]);
    short_to_float(bias_dw_f, block->convdw3x3_bias, block->len_convdw3x3_bias, block->magnification[3]);


    for ( ; i < t; i++ ) {
        conv1x1s1_neon(input, mat_conv1x1_di, weight_di_f + i * input_channel * once_out_channel, bias_di_f + i * once_out_channel);
        hswish_neon( mat_conv1x1_di );
        padding( mat_conv1x1_di, mat_conv3x3padding, block->padding, block->padding, 0, 0 );

        if ( stride == 1 ) {
            convdw3x3s1_neon(mat_conv3x3padding, mat_conv3x3, weight_dw_f + i * once_out_channel * 9, bias_dw_f + i * once_out_channel);
        } else if ( stride == 2 ) {
            convdw3x3s2_neon(mat_conv3x3padding, mat_conv3x3, weight_dw_f + i * once_out_channel * 9, bias_dw_f + i * once_out_channel);
        }

        hswish_neon( mat_conv3x3 );
        mat_conv3x3.data += jump;
    }

    memcpy( input.data, conv3x3_begin, conv3x3_len * sizeof( float ) );
    mat_conv3x3.data = input.data;
    mat_conv3x3.c = c;
    // conv1x1
    // conv1x1s1_neon
    c = block->out_channel;
    Mat mat_conv1x1_dd = newMat(new_memory(mat_conv3x3), w, h, c);

    float* weight_dd_f = new_memory(mat_conv1x1_dd);
    float* bias_dd_f = weight_dd_f + block->len_conv1x1_dd_weight;
    short_to_float(weight_dd_f, block->conv1x1_dd_weight, block->len_conv1x1_dd_weight, block->magnification[4]);
    short_to_float(bias_dd_f, block->conv1x1_dd_bias, block->len_conv1x1_dd_bias, block->magnification[5]);
    // Mat mat_conv1x1_dd = newMat(input.data, h, w, c);
    conv1x1s1_neon(mat_conv3x3, mat_conv1x1_dd, weight_dd_f, bias_dd_f);

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
int char2float( unsigned char* src, float* dst, int h, int w ) // int2float and cut
{
    int i, j;

    for ( i = 0; i < h; i++ ) {
        for ( j = 0; j < w; j++ ) {
            dst[i * w + j] = ( float )src[ i  *  w  + j ];
        }

        // printf("dst[%d]=%d\n", i*w + 90, dst[i*w + 90]);
    }

    return SL_RET_SUCCESS;
}
Mat preblock_short(Mat input, Block_Conv3x3_Short* block) // 第二次优化内存
{
    int w = input.w, h = input.h, c = input.c; //, stride = block->stride;
    int t = 4;
    int i = 0;
    int outh = ( int )( ( h + 2 * block->padding - 3 ) / block->stride ) + 1;
    int outw = ( int )( ( w + 2 * block->padding - 3 ) / block->stride ) + 1;
    // int out_size = block->out_channel * alignPtr(outh * outw, MALLOC_ALIGN);
    int right_stride = total( input );
    // if (out_size > right_stride) { //if stride is equal 2,out_size may be less than total(input)
    //     right_stride = out_size;
    // }

    // float* left = input.data;
    float* right = input.data + right_stride;

    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, c);
    memset( right, 0, total( mat_conv3x3padding ) * sizeof( float ) );
    padding( input, mat_conv3x3padding, block->padding, block->padding, 0, 0 );

    memmove( input.data, mat_conv3x3padding.data, total( mat_conv3x3padding ) * sizeof( float ) );
    mat_conv3x3padding.data = input.data;

    int out_c = block->out_channel;
    Mat mat_conv3x3s1 = newMat(new_memory(mat_conv3x3padding), outw, outh, out_c / t);
    w /= 2, h /= 2;
    Mat output = newMat(new_memory(mat_conv3x3s1), w, h, out_c / t);
    float* output_begin = output.data;
    int jump = total( output );
    int output_len = jump * t;

    float* weight_f = output.data + output.cstep * out_c;
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);

    for ( ; i < t; i++ ) {
        conv3x3s1_neon(mat_conv3x3padding, mat_conv3x3s1, weight_f + i * c * out_c / t * 9, bias_f + i * out_c / t);
        relu_neon( mat_conv3x3s1 );
        // leakyrelu_neon(mat_conv3x3s2,0.1);
        pooling2x2s2_max_neon( mat_conv3x3s1, output );
        output.data += jump;
    }

    memcpy( input.data, output_begin, output_len * sizeof( float ) );
    output.data = input.data;
    output.c = out_c;
    return output;
}
void pooling2x1s2x1_max_neon(Mat bottom_blob, Mat top_blob)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w;

    //#pragma omp parallel for num_threads(opt.num_threads)
    int q;
    int i;

    for (q = 0; q < inch; q++) {
        const float* img0 = channel(bottom_blob, q);
        float* outptr = channel(top_blob, q);

        const float* r0 = img0;
        const float* r1 = img0 + w;

        for (i = 0; i < outh; i++) {

            int remain = outw;

            for (; remain > 0; remain--) {

                *outptr = MAX(r0[0], r1[0]);

                r0 += 1;
                r1 += 1;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
//conv3x3_block
Mat conv3x3_block_short(Mat input, Block_Conv3x3_Short* block, float* rightpointer, int pooling, int pad_type) // �������ݱ���λ��left_memory
{
    int h = input.h, w = input.w, c = input.c, stride = block->stride;


    float* left = input.data;
    /*float * right = left + alignPtr(block->out_channel * alignPtr(((h + 2 * (block->padding - 1))) * ((w + 2 * (block->padding - 1))), MALLOC_ALIGN), MALLOC_ALIGN);*/
    float* right = rightpointer;
    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, c);
    memset(right, 0, total(mat_conv3x3padding) * sizeof(float)); //padding��Ҫ������ڴ�
    padding(input, mat_conv3x3padding, block->padding, block->padding, pad_type, 0);

    h = (h + 2 * (block->padding - 1)) / stride, w = (w + 2 * (block->padding - 1)) / stride, c = block->out_channel;
    Mat mat_conv3x3 = newMat(left, w, h, c);
    float* weight_f = new_memory(mat_conv3x3);
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);

    if (stride == 1) {
        conv3x3s1_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    }

    if (stride == 2) {
        conv3x3s2_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    }

    relu_neon(mat_conv3x3);

    if (pooling) {
        h = h >> 1, w = w >> 1;
        Mat mat_pooling_max = newMat(left, w, h, c);
        pooling2x2s2_max_neon(mat_conv3x3, mat_pooling_max);
        return mat_pooling_max;
    }

    return mat_conv3x3;
}
Mat bottleneck1_short(Mat input, Block_Bneck_SE_Short* block)
{
    //conv1x1
    //conv1x1s1_neon
    int h = input.h, w = input.w, c = block->ratio, stride = block->stride;

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
    padding(mat_conv1x1_di, mat_conv3x3padding, block->padding, block->padding, 2, 0);
    w = (w - 3 + 2 * block->padding) / stride + 1;
    h = (h - 3 + 2 * block->padding) / stride + 1;
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
    //Mat mat_global_pooling = newMat(new_memory(mat_conv1x1_dd), 1, 1, c);
    //pooling_global(mat_conv1x1_dd, mat_global_pooling, PoolMethod_AVE);
    //c /= 4;
    //Mat mat_conv1x1dd_se = newMat(new_memory(mat_global_pooling), 1, 1, c);
    //conv1x1s1_cnn_neon(mat_global_pooling, mat_conv1x1dd_se, block->conv1x1_dd_se_weight, block->conv1x1_dd_se_bias);
    //relu_neon(mat_conv1x1dd_se);
    //c = block->out_channel;
    //Mat mat_conv1x1di_se = newMat(new_memory(mat_conv1x1dd_se), 1, 1, c);
    //conv1x1s1_cnn_neon(mat_conv1x1dd_se, mat_conv1x1di_se, block->conv1x1_di_se_weight, block->conv1x1_di_se_bias);
    //hsigmoid_neon(mat_conv1x1di_se);
    //
    //mat_scale_neon_inplace(mat_conv1x1_dd, mat_conv1x1di_se);

    //shortcut

    if (stride == 1) {
        mat_add_neon_inplace(input, mat_conv1x1_dd);
        return input;
    }

    Mat output = newMat(input.data, w, h, c);
    memcpy(output.data, mat_conv1x1_dd.data, mat_conv1x1_dd.cstep * c * sizeof(float));

    return output;
}
Mat bottleneck2_short(Mat input, Block_Bneck_SE_Short* block)
{
    //conv1x1
    //conv1x1s1_neon
    int h = input.h, w = input.w, c = block->ratio, stride = block->stride;

    //Mat mat_conv1x1_di = newMat(new_memory(input), w, h, c);

    //conv1x1s1_cnn_neon(input, mat_conv1x1_di, block->conv1x1_di_weight, block->conv1x1_di_bias);
    //hswish_neon(mat_conv1x1_di);

    //conv3x3
    //padding
    //convdw3x3s1_neon or convdw3x3s2_neon
    Mat mat_conv3x3padding = newMat(new_memory(input), w + 2 * block->padding, h + 2 * block->padding, c);
    memset(mat_conv3x3padding.data, 0, total(mat_conv3x3padding) * sizeof(float));
    padding(input, mat_conv3x3padding, block->padding, block->padding, 2, 0);
    w = (w - 3 + 2 * block->padding) / stride + 1;
    h = (h - 3 + 2 * block->padding) / stride + 1;
    Mat mat_conv3x3 = newMat(new_memory(mat_conv3x3padding), w, h, c); //回到mat_conv1x1_di.data指针，节省内存
    float* weight_f = new_memory(mat_conv3x3);
    float* bias_f = weight_f + block->len_convdw3x3_weight;
    short_to_float(weight_f, block->convdw3x3_weight, block->len_convdw3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->convdw3x3_bias, block->len_convdw3x3_bias, block->magnification[1]);

    if (stride == 1) {
        convdw3x3s1_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    } else if (stride == 2) {
        convdw3x3s2_neon(mat_conv3x3padding, mat_conv3x3, weight_f, bias_f);
    }

    hswish_neon(mat_conv3x3);

    //conv1x1
    //conv1x1s1_neon
    c = block->out_channel;
    Mat mat_conv1x1_dd = newMat(mat_conv3x3padding.data, w, h, c);
    weight_f = new_memory(mat_conv3x3);
    bias_f = weight_f + block->len_conv1x1_dd_weight;
    short_to_float(weight_f, block->conv1x1_dd_weight, block->len_conv1x1_dd_weight, block->magnification[2]);
    short_to_float(bias_f, block->conv1x1_dd_bias, block->len_conv1x1_dd_bias, block->magnification[3]);
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
    short_to_float(weight_f, block->conv1x1_dd_se_weight, block->len_conv1x1_dd_se_weight, block->magnification[4]);
    short_to_float(bias_f, block->conv1x1_dd_se_bias, block->len_conv1x1_dd_se_bias, block->magnification[5]);
    conv1x1s1_neon(mat_global_pooling, mat_conv1x1dd_se, weight_f, bias_f);
    relu_neon(mat_conv1x1dd_se);
    c = block->out_channel;
    Mat mat_conv1x1di_se = newMat(new_memory(mat_conv1x1dd_se), 1, 1, c);
    weight_f = new_memory(mat_conv1x1di_se);
    bias_f = weight_f + block->len_conv1x1_di_se_weight;
    short_to_float(weight_f, block->conv1x1_di_se_weight, block->len_conv1x1_di_se_weight, block->magnification[6]);
    short_to_float(bias_f, block->conv1x1_di_se_bias, block->len_conv1x1_di_se_bias, block->magnification[7]);
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



int conv_block_s(Mat* input, Block_Conv3x3_Short* block, int kernel, int bool_relu)
{
    int h = input->h, w = input->w, c = input->c, stride = block->stride;

    float* left = input->data;
    float* right = left + alignPtr(input->cstep * MAX(c, block->out_channel) + h, MALLOC_ALIGN);

    Mat mat_convpadding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, c);

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
    float* weight_f = new_memory(mat_convpadding);
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);

    if (kernel == 3) {
        if (stride == 1) {
            conv3x3s1_neon(mat_convpadding, mat_convs, weight_f, bias_f);
        } else {
            conv3x3s2_neon(mat_convpadding, mat_convs, weight_f, bias_f);
        }
    }
    //else if (kernel == 5) {
    //    if (stride == 1) {
    //        conv5x5s1_neon(mat_convpadding, mat_convs, weight_f, bias_f);
    //    }
    //    else {
    //        conv5x5s2_neon(mat_convpadding, mat_convs, weight_f, bias_f);
    //    }
    //}
    //else if (kernel == 7) {
    //    if (stride == 1) {
    //        conv7x7s1_neon(mat_convpadding, mat_convs, weight_f, bias_f);
    //    }
    //    else {
    //        conv7x7s2_neon(mat_convpadding, mat_convs, weight_f, bias_f);
    //    }
    //}
    else {

    }

    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int conv1x1_blocktool_s(Mat* input, Block_Conv1x1_Short* block, float* right)
{
    int h = input->h, w = input->w, c = block->out_channel;
    Mat mat_convs = newMat(right, w, h, c);
    memset(right, 0, total(mat_convs) * sizeof(float));

    float* weight_f = new_memory(mat_convs);
    float* bias_f = weight_f + block->len_conv1x1_weight;
    short_to_float(weight_f, block->conv1x1_weight, block->len_conv1x1_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv1x1_bias, block->len_conv1x1_bias, block->magnification[1]);

    conv1x1s1_neon(*input, mat_convs, weight_f, bias_f);
    copy(mat_convs, *input);
    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);

    return SL_RET_SUCCESS;
}

int conv_group_single_block_s(Mat* input, Block_Conv3x3_Short* block, int kernel, int group)
{
    int h = input->h, w = input->w, c = input->c, stride = block->stride;
    float* right = input->data + alignPtr(input->cstep * MAX(c, block->out_channel) + h, MALLOC_ALIGN);
    Mat mat_convpadding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, c);
    memset(right, 0, total(mat_convpadding) * sizeof(float));

    padding_normal(*input, mat_convpadding, block->padding);

    h = (h - kernel + 2 * block->padding) / stride + 1, w = (w - kernel + 2 * block->padding) / stride + 1, c = block->out_channel;

    Mat mat_convs = newMat(input->data, w, h, c);
    memset(input->data, 0, total(mat_convs) * sizeof(float));

    float* weight_f = new_memory(mat_convpadding);
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);

    if (block->stride == 2) {
        groupconv3x3s2_neon(mat_convpadding, mat_convs, weight_f, bias_f, group);
    } else {
        groupconv3x3s1_neon(mat_convpadding, mat_convs, weight_f, bias_f, group);
    }

    input->h = h;
    input->w = w;
    input->c = c;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int resnet3x3_blockdw_s(Mat* input, Block_Resnet_Short* block)
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
    float* weight_f = new_memory(conv_tmp);
    float* bias_f = weight_f + block->len_conv3x3_1_weight;
    short_to_float(weight_f, block->conv3x3_1_weight, block->len_conv3x3_1_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_1_bias, block->len_conv3x3_1_bias, block->magnification[1]);
    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, weight_f, bias_f);

    memset(left, 0, total(mat_conv3x3s) * sizeof(float));
    weight_f = new_memory(conv_tmp);
    bias_f = weight_f + block->len_conv3x3_2_weight;
    short_to_float(weight_f, block->conv3x3_2_weight, block->len_conv3x3_2_weight, block->magnification[2]);
    short_to_float(bias_f, block->conv3x3_2_bias, block->len_conv3x3_2_bias, block->magnification[3]);
    conv1x1s1_neon(conv_tmp, mat_conv3x3s, weight_f, bias_f);

    relu_neon(mat_conv3x3s);

    padding_normal(mat_conv3x3s, mat_conv3x3padding, block->padding);

    memset(rightpoint, 0, total(conv_tmp) * sizeof(float));
    weight_f = new_memory(conv_tmp);
    bias_f = weight_f + block->len_conv3x3_3_weight;
    short_to_float(weight_f, block->conv3x3_3_weight, block->len_conv3x3_3_weight, block->magnification[4]);
    short_to_float(bias_f, block->conv3x3_3_bias, block->len_conv3x3_3_bias, block->magnification[5]);
    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, weight_f, bias_f);

    memset(left, 0, total(mat_conv3x3s) * sizeof(float));
    weight_f = new_memory(conv_tmp);
    bias_f = weight_f + block->len_conv3x3_4_weight;
    short_to_float(weight_f, block->conv3x3_4_weight, block->len_conv3x3_4_weight, block->magnification[6]);
    short_to_float(bias_f, block->conv3x3_4_bias, block->len_conv3x3_4_bias, block->magnification[7]);
    conv1x1s1_neon(conv_tmp, mat_conv3x3s, weight_f, bias_f);
    leakyrelu_neon(mat_conv3x3s, 0.2);
    mat_add_neon_inplace(*input, mat_conv3x3s);
    input->h = h;
    input->w = w;
    input->c = block->in_channel;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int conv_sep_block_s(Mat* input, Block_Sep_Short* block)
{
    int h = input->h, w = input->w, c = input->c;
    //float * left = new_memory(*input);
    //float *right = left + alignPtr(input->cstep*c + h, MALLOC_ALIGN); //h*w*c
    float* right = input->data + alignPtr(input->cstep * c + h, MALLOC_ALIGN); //h*w*c

    Mat mat_conv3x3padding = newMat(right, w + 2 * block->padding, h + 2 * block->padding, block->in_channel);
    memset(right, 0, total(mat_conv3x3padding) * sizeof(float));

    padding_normal(*input, mat_conv3x3padding, block->padding);
    float* rightpoint = new_memory(mat_conv3x3padding);
    Mat conv_tmp = newMat(rightpoint, w, h, block->in_channel);
    memset(rightpoint, 0, total(conv_tmp) * sizeof(float));

    float* weight_f = new_memory(conv_tmp);
    float* bias_f = weight_f + block->len_conv3x3_1_weight;
    short_to_float(weight_f, block->conv3x3_1_weight, block->len_conv3x3_1_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_1_bias, block->len_conv3x3_1_bias, block->magnification[1]);
    convdw3x3s1_neon(mat_conv3x3padding, conv_tmp, weight_f, bias_f);

    Mat mat_convs = newMat(right, w, h, block->out_channel);
    memset(right, 0, total(mat_convs) * sizeof(float));
    weight_f = new_memory(conv_tmp);
    bias_f = weight_f + block->len_conv3x3_2_weight;
    short_to_float(weight_f, block->conv3x3_2_weight, block->len_conv3x3_2_weight, block->magnification[2]);
    short_to_float(bias_f, block->conv3x3_2_bias, block->len_conv3x3_2_bias, block->magnification[3]);
    conv1x1s1_neon(conv_tmp, mat_convs, weight_f, bias_f);
    copy(mat_convs, *input);

    input->h = h;
    input->w = w;
    input->c = block->out_channel;
    input->cstep = alignPtr((w) * (h), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}

int deconv4x4_block_s(Mat* input, Block_Conv3x3_Short* block)
{
    int h = input->h, w = input->w, c = block->out_channel, stride = block->stride;

    float* left = input->data;
    int address = MAX(input->c, c << 2);
    float* right = left + alignPtr(input->cstep * address + h, MALLOC_ALIGN);

    relu_neon(*input);

    h = (h - 1) * stride + 3, w = (w - 1) * stride + 3;

    Mat mat_conv4x4s = newMat(right, w, h, c);
    memset(right, 0, total(mat_conv4x4s) * sizeof(float));
    float* weight_f = new_memory(mat_conv4x4s) + h;
    float* bias_f = weight_f + block->len_conv3x3_weight;
    short_to_float(weight_f, block->conv3x3_weight, block->len_conv3x3_weight, block->magnification[0]);
    short_to_float(bias_f, block->conv3x3_bias, block->len_conv3x3_bias, block->magnification[1]);
    deconv4x4s2_neon(*input, mat_conv4x4s, weight_f, bias_f);
    Mat deconv_crop = newMat(left, w - 1, h - 1, block->out_channel);
    memset(left, 0, total(deconv_crop) * sizeof(float));
    deconvcrop(mat_conv4x4s, deconv_crop, 1);
    input->h = h - 1;
    input->w = w - 1;
    input->c = block->out_channel;
    input->cstep = alignPtr((w - 1) * (h - 1), MALLOC_ALIGN);
    return SL_RET_SUCCESS;
}


