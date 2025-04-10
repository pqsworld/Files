#include "net_api.h"
#include "net_param/para_mistouch.h"
#include "../net_cnn_common.h"

#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "../alog.h"


#if RUN_TST
#include "SL_Math.h"
// #include "math.h"
//  #define SL_Sqrt sqrt
//  #define exp_appro exp
#define SL_LOGD printf
#define SL_LOGE printf
#else
#include "SL_Math.h"
#endif
// beta,gamma,epsilon,mean,var

#ifndef NULL
#define NULL (void *)0
#endif

static const float epsilon = 1e-5;
// beta = this.offset.value,
// gamma=this.scale.value,eps=this.epsilon,
// mean=this.trainedmeancache.value,var=this.trainedvariancecache.value
// #define OP_GEN2_ART_RAW_JIANBIAN

static short saturate_cast_short( int data )
{
    if ( data < -32768 ) {
        data = -32768;
    } else if ( data > 32767 ) {
        data = 32767;
    }

    return data;
}

static void silead_resize_bilinear( unsigned char* src, int w_in, int h_in, int w_out, int h_out, unsigned char* dst )
{
    // IplImage *matSrc, *matDst1;
    // matSrc = (IplImage*)src;
    // matDst1 = (IplImage*)dst;

    double scale_x = ( double )w_in / w_out;
    double scale_y = ( double )h_in / h_out;

    unsigned char* dataDst = dst;
    int stepDst = w_out;          // matDst1->width;
    unsigned char* dataSrc = src; // matSrc->imageData;//data;
    int stepSrc = w_in;           // matSrc->width;//step;
    int iWidthSrc = w_in;         // matSrc->width;//.cols;
    int iHiehgtSrc = h_in;        // matSrc->height;//.rows;
    int j = 0;
    int i = 0;
    int k = 0;

    for ( j = 0; j < h_out; ++j ) {
        // matDst1->height
        float fy = ( float )( ( j + 0.5 ) * scale_y - 0.5 );
        int sy = SL_Floor( fy );
        fy -= sy;
        sy = MIN( sy, iHiehgtSrc - 2 );
        sy = MAX( 0, sy );

        short cbufy[2];
        cbufy[0] = saturate_cast_short( ( 1.f - fy ) * 2048 );
        cbufy[1] = 2048 - cbufy[0];

        for ( i = 0; i < w_out; ++i ) {
            // matDst1->width
            float fx = ( float )( ( i + 0.5 ) * scale_x - 0.5 );
            int sx = SL_Floor( fx );
            fx -= sx;

            if ( sx < 0 ) {
                fx = 0, sx = 0;
            }

            if ( sx >= iWidthSrc - 1 ) {
                fx = 0, sx = iWidthSrc - 2;
            }

            short cbufx[2];
            cbufx[0] = saturate_cast_short( ( 1.f - fx ) * 2048 );
            cbufx[1] = 2048 - cbufx[0];

            for ( k = 0; k < 1; ++k ) {
                // matSrc->nChannels
                *( dataDst + j * stepDst + 1 * i + k ) = ( *( dataSrc + sy * stepSrc + 1 * sx + k ) * cbufx[0] * cbufy[0] +
                        * ( dataSrc + ( sy + 1 ) * stepSrc + 1 * sx + k ) * cbufx[0] * cbufy[1] +
                        * ( dataSrc + sy * stepSrc + 1 * ( sx + 1 ) + k ) * cbufx[1] * cbufy[0] +
                        * ( dataSrc + ( sy + 1 ) * stepSrc + 1 * ( sx + 1 ) + k ) * cbufx[1] * cbufy[1] ) >>
                        22;
            }
        }
    }
}

#if 0
void print_txt( float* dst, int rows, int cols )
{
    FILE* fp = fopen( "data.txt", "w" );
    int i, j;

    for ( i = 0; i < rows; i++ ) {
        for ( j = 0; j < cols; j++ ) {
            fprintf( fp, "%10.4f", *( dst + cols * i + j ) );
        }

        fprintf( fp, "" );
    }

    fclose( fp );
}
#endif
static void inputImageAverage( const float* src, short h, short w, float* dst )
{
    int j, i;
    int row = 0;

    for ( j = 0; j < h; j++ ) {
        for ( i = 0; i < w; i++ ) {
            *(dst + row + i) = *(src + row + i) - *(para_hw_input_Average + row + i);
        }

        row += w;
    }
}

static void ReLU( const float* src, float* dst, int length )
{
    int i;

    for ( i = 0; i < length; i++ ) {
        *( dst + i ) = *( src + i ) > 0.0f ? *( src + i ) : 0;
        //;//printf("%f,",*(dst + i));
        // if((i+1)%(int)(sqrt(length))==0)
        //;//printf("");
    }
}

static void maxpooling_stride2_size2( const float* src, short h, short w, float* dst )
{
    int i, j, i2;
    float mx;
    // int outsize = h>>1;
    int row = 0;
    int row2 = 0;

    for ( j = 0; j < h / 2; j++ ) {
        for ( i = 0; i < w / 2; i++ ) {
            i2 = i * 2;
            mx = *( src + row2 + i2 );
            mx = *( src + row2 + i2 + 1 ) > mx ? *( src + row2 + i2 + 1 ) : mx;
            mx = *( src + row2 + w + i2 ) > mx ? *( src + row2 + w + i2 ) : mx;
            mx = *( src + row2 + w + i2 + 1 ) > mx ? *( src + row2 + w + i2 + 1 ) : mx;
            *( dst + i + row ) = mx;
            //;//printf("%f,",mx);
        }

        //;//printf("");
        row += w / 2; // outsize;
        row2 += w * 2;
    }
}
static void softmax( const float* src, int channel, float* dst )
{
    // exponents = X - MAX(X,[],3);
    // expX = exp(exponents);
    // Z = expX./sum(expX,3);
    int i;
    float exp_sum = 0;
    float mx = *src;
    ; // printf("func:%s,%f,%f,",__func__,*src,*(src+1));

    for ( i = 1; i < channel; i++ )
        if ( *( src + i ) > mx ) {
            mx = *( src + i );
        }

    for ( i = 0; i < channel; i++ ) {
        *( dst + i ) = *( src + i ) - mx;
        *( dst + i ) = SL_exp_appro( *( dst + i ) );
        //;//printf("%f,",*(dst+i));
    }

    for ( i = 0; i < channel; i++ ) {
        exp_sum += ( *( dst + i ) );
    }

    //;//printf("%f,",exp_sum);
    for ( i = 0; i < channel; i++ ) {
        *( dst + i ) /= ( exp_sum );
    }
}
/*
static void averagepooling(const float*src, short h, short w, float *dst)
{
  int i,j;
  int row = 0;
  *dst = 0;

  for(j = 0; j < h; j++)
  {
    for(i = 0; i < w; i++)
    {
      *dst += *(src+i+row);
    }
    row += w;
  }
  *dst = *dst/h/w;
  //;//printf("%f,",*dst);
}
*/
// fullyconnect(relu_memory, hc, wc, fc_input_channel, class_number, hw_fc_weights, hw_fc_bias, pool_memory);
static void fullyconnect( const float* src, short h, short w, short channel_in, short channel_out, float* weights, float* bias, float* dst )
{
    int i, k;
    int weights_size = h * w * channel_in;

    //    int data_size= h*w;
    for ( i = 0; i < channel_out; i++ ) {
        dst[i] = 0;

        for ( k = 0; k < weights_size; k++ ) {
            dst[i] += src[k] * weights[k + i * weights_size];
        }

        // printf("%f,",dst[i]);
        dst[i] += bias[i];
    }
}
static int batchnormalization( float* src, short h, short w, float* dst, float beta, float gamma, float epsilon, float mean, float var )
{
    int i, j;
    float* lpsrc = src, *lpdst = dst;
    float scale = gamma / SL_Sqrt( var + epsilon );
    float offset = beta - gamma * mean / SL_Sqrt( var + epsilon );

    if ( src == NULL || dst == NULL ) {
        return SL_RET_FAIL;
    }

    for ( i = 0; i < h; i++ ) {
        for ( j = 0; j < w; j++, lpsrc++, lpdst++ ) {
            *lpdst = *lpsrc * scale + offset;
            //;//printf("%f,",*lpdst);
        }

        //;//printf("");
    }

    return SL_RET_SUCCESS;
}

// n: ksize, can only be odd number
#if NEON
/*
NEON support only for image whose size can be divided by 4.
*/
static int conv_nxn_sub_same_neon( float* I, float* kernel, int n, float* O, int rows, int cols )
{
    int x, y, i;
    int new_rows = rows + n - 1;
    int new_cols = cols + n - 1;
    int kk = ( n - 1 ) / 2;

    if ( I == NULL || O == NULL || kernel == NULL ) {
        return 1;
    }

    // float val;
    float* lpSrc = NULL;
    float* lpDst = NULL;
    float* temp = ( float* )malloc( new_rows * new_cols * sizeof( float ) );

    if ( NULL == temp ) {
        return -2;
    }

    memset( temp, 0x00, new_rows * new_cols * sizeof( float ) );

    if ( n % 2 != 1 ) {
        free( temp ); // printf("please input an odd number.");
        return 1;
    }

    // copy ori image
    for ( y = 0; y < rows; y++ ) {
        lpSrc = I + y * cols;
        lpDst = temp + kk + kk * new_cols + y * new_cols;
        memcpy( lpDst, lpSrc, cols * sizeof( float ) );
    }

    // temp 本来不就是memset 0了？这里还需要memset吗
    /*
      lpSrc = temp;
        for(i = 0; i < kk; i++)
          memset(lpSrc+i*new_cols, 0x00, new_cols*sizeof(float));

        lpSrc = temp + (rows+kk-1)*new_cols;
        for(i = 1; i <= kk; i++)
          memset(lpSrc+i*new_cols, 0x00, new_cols*sizeof(float));

        for (y=0; y<new_rows; y++)
        {
        lpSrc = temp + y*new_cols;
        for(i = 0; i < kk; i++)
          *(lpSrc+i) = 0;

        lpSrc = temp + cols+ kk-1 + y*new_cols;
        for(i = 1; i <= kk; i++)
          *(lpSrc+i) = 0;
        }
    */
    // 以上是否需要？
    //       kernel_sum = 0;
    //       for(i = 0; i < n*n; i++)
    //           kernel_sum += *(kernel+i);
    // #if NEON
    //     float32x4_t f_neon;
    float* S0, *S1, *S2, *D0;
    float32x4_t s4_neon_0, s4_neon_1, s4_neon_2;

    for ( y = kk; y < new_rows - kk; y++ ) {
        // for (x=kk; x<new_cols-kk; x++)
        for ( x = kk; x <= new_cols - kk - 4; x += 4 ) {
            lpSrc = temp + ( y - kk ) * new_cols + ( x - kk );
            lpDst = O + ( y - kk ) * cols + ( x - kk );
            D0 = lpDst;
            S0 = lpSrc;
            S1 = lpSrc + new_cols;
            S2 = lpSrc + new_cols + new_cols;
            s4_neon_0 = vmulq_f32( vdupq_n_f32( kernel[0] ), vld1q_f32( S0 ) );
            s4_neon_1 = vmulq_f32( vdupq_n_f32( kernel[n] ), vld1q_f32( S1 ) );
            s4_neon_2 = vmulq_f32( vdupq_n_f32( kernel[n << 1] ), vld1q_f32( S2 ) );

            for ( i = 1; i < n; i++ ) {
                S0++;
                S1++;
                S2++;
                s4_neon_0 = vmlaq_f32( s4_neon_0, vdupq_n_f32( kernel[0 + i] ), vld1q_f32( S0 ) );
                s4_neon_1 = vmlaq_f32( s4_neon_1, vdupq_n_f32( kernel[n + i] ), vld1q_f32( S1 ) );
                s4_neon_2 = vmlaq_f32( s4_neon_2, vdupq_n_f32( kernel[( n << 1 ) + i] ), vld1q_f32( S2 ) );
            }

            s4_neon_0 = vaddq_f32( s4_neon_0, s4_neon_1 );
            s4_neon_0 = vaddq_f32( s4_neon_0, s4_neon_2 );
            vst1q_f32( D0, s4_neon_0 );
        }
    }

    /*#else
        for (y=kk; y<new_rows-kk; y++)
        {
          for (x=kk; x<new_cols-kk; x++)
          {
            val = 0;
            k = 0;
            lpSrc = temp + (y-kk)*new_cols + (x-kk);
            lpDst = O + (y-kk)*cols + (x-kk);
            for(i = 0; i < n; i++)
            {
              for(j = 0; j < n; j++)
              {
                  val += *(lpSrc+j) * *(kernel + k++);
              }
              lpSrc += new_cols;
            }
            *lpDst = val;
          }
        }
    #endif
    */
    //      //print_txt(O, rows, cols);
    if ( temp ) {
        free( temp );
    }

    return 0;
}

static int conv_nxn_single_bias_neon( float* src, float* kernel, float bias, int n, int prev, float* dst, int* h, int* w, int padding )
{
    int i = 0, j = 0, hp = 0, wp = 0;
    float* lpkernel = NULL;

    if ( src == NULL || kernel == NULL || dst == NULL ) {
        return 1;
    }

    if ( padding == 0 ) {
        hp = *h;
        wp = *w;
    }

    float* temp = ( float* )malloc( hp * wp * sizeof( float ) );

    if ( NULL == temp ) {
        return -2;
    }

    memset( dst, 0x00, hp * wp * sizeof( float ) );

    for ( i = 0; i < prev; i++ ) {
        lpkernel = kernel + i * n * n;

        if ( padding == 0 ) {
            conv_nxn_sub_same_neon( src + i * ( *h ) * ( *w ), lpkernel, n, temp, hp, wp );
        }

        for ( j = 0; j < hp * wp; j++ ) {
            //            ;//printf("%f", *(dst+j));
            *( dst + j ) += *( temp + j );
        }

        // print_txt(dst, hp, wp);
    }

    for ( i = 0; i < hp * wp; i++ ) {
        *( dst + i ) += bias;
        //;//printf("%f, ", *(dst+i));
        // if((i+1)%hp==0)
        //;//printf("");
    }

    // print_txt(dst, hp, wp);

    *h = hp;
    *w = wp;

    if ( temp ) {
        free( temp );
    }

    return 0;
}

#endif
static int conv_nxn_sub_same( float* I, float* kernel, int n, float* O, int rows, int cols )
{
    int x, y, i, j, k;
    int new_rows = rows + n - 1;
    int new_cols = cols + n - 1;
    int kk = ( n - 1 ) / 2;

    if ( I == NULL || O == NULL || kernel == NULL ) {
        return 1;
    }

    float val;
    float* lpSrc = NULL;
    float* lpDst = NULL;
    float* temp = ( float* )malloc( new_rows * new_cols * sizeof( float ) );

    if ( NULL == temp ) {
        return -2;
    }

    memset( temp, 0.00, new_rows * new_cols * sizeof( float ) );

    if ( n % 2 != 1 ) {
        free( temp ); // printf("please input an odd number.");
        return 1;
    }

    // copy ori image
    for ( y = 0; y < rows; y++ ) {
        lpSrc = I + y * cols;
        lpDst = temp + kk + kk * new_cols + y * new_cols;
        memcpy( lpDst, lpSrc, cols * sizeof( float ) );
    }

    /*
        lpSrc = temp;
        for(i = 0; i < kk; i++)
          memset(lpSrc+i*new_cols, 0x00, new_cols*sizeof(float));

        lpSrc = temp + (rows+kk-1)*new_cols;
        for(i = 1; i <= kk; i++)
          memset(lpSrc+i*new_cols, 0x00, new_cols*sizeof(float));

        for (y=0; y<new_rows; y++)
        {
        lpSrc = temp + y*new_cols;
        for(i = 0; i < kk; i++)
          *(lpSrc+i) = 0;

        lpSrc = temp + cols+ kk-1 + y*new_cols;
        for(i = 1; i <= kk; i++)
          *(lpSrc+i) = 0;
        }
    */
    //      kernel_sum = 0;
    //      for(i = 0; i < n*n; i++)
    //          kernel_sum += *(kernel+i);

    for ( y = kk; y < new_rows - kk; y++ ) {
        for ( x = kk; x < new_cols - kk; x++ ) {
            val = 0;
            k = 0;
            lpSrc = temp + ( y - kk ) * new_cols + ( x - kk );
            lpDst = O + ( y - kk ) * cols + ( x - kk );

            for ( i = 0; i < n; i++ ) {
                for ( j = 0; j < n; j++ ) {
                    val += *( lpSrc + j ) * *( kernel + k++ );
                }

                lpSrc += new_cols;
            }

            *lpDst = val;
        }
    }

    //      //print_txt(O, rows, cols);

    if ( temp ) {
        free( temp );
    }

    return 0;
}

static int conv_nxn_single_bias( float* src, float* kernel, float bias, int n, int prev, float* dst, int* h, int* w, int padding )
{
    int i = 0, j = 0, hp = 0, wp = 0;
    float* lpkernel = NULL;

    if ( src == NULL || kernel == NULL || dst == NULL ) {
        return 1;
    }

    if ( padding == 0 ) {
        hp = *h;
        wp = *w;
    }

    float* temp = ( float* )malloc( hp * wp * sizeof( float ) );

    if ( NULL == temp ) {
        return -2;
    }

    memset( dst, 0x00, hp * wp * sizeof( float ) );

    for ( i = 0; i < prev; i++ ) {
        lpkernel = kernel + i * n * n;

        if ( padding == 0 ) {
            conv_nxn_sub_same( src + i * ( *h ) * ( *w ), lpkernel, n, temp, hp, wp );
        }

        for ( j = 0; j < hp * wp; j++ ) {
            //            ;//printf("%f", *(dst+j));
            *( dst + j ) += *( temp + j );
        }
    }

    for ( i = 0; i < hp * wp; i++ ) {
        *( dst + i ) += bias;

        //;//printf("%f, ", *(dst+i));
        // if((i+1)%hp==0)
        //;//printf("");
    }

    // print_txt(dst, hp, wp);

    *h = hp;
    *w = wp;

    if ( temp ) {
        free( temp );
    }

    return 0;
}

static int resize( unsigned char* input, int height, int width, float* output )
{
    int i = 0;
    int j = 0;
    int m = 0, n = 0;

    if ( NULL == input || NULL == output || height <= 0 || width <= 0 ) {
        return -1;
    }

    for ( i = 0; i < height; i += 2 ) {
        n = 0;

        for ( j = 0; j < width; j++ ) {
            output[( height / 2 - 1 - m ) * width + n] = input[( height - 1 - i ) * width + j] * 1.0; // 可简化，这里和SL_WriteBmp保持一致
            n++;
        }

        m++;
    }

    return 0;
}
/*******************************************************************
Function: sl_fingerprint_op_gen2_spoof_post
parameter@src: input image data in float format
parameter@h: input image height
parameter@w: input image width
parameter@score: output image classification score, image will be
        classified as fingerprints if score > threshold(11141)
version: pre test 20190810
********************************************************************/
static int sl_fingerprint_hw_spoof_0813( float* src, short h, short w, int* score )
{
    int ret = 0;
    int size = 100000;
    int hc = h, wc = w;
    float* memory = ( float* )malloc( size );

    if ( NULL == memory ) {
        // not enough memory
        SL_LOGE( "sl_fingerprint_op_gen2_fc_jianbian_v0416 mslloc failed!" );
        return -2;
    }

    float* layer1_image_out = memory;
    inputImageAverage( src, h, w, layer1_image_out );

    // print_txt(layer1_image_out, hc, wc);
    //    ;//printf("%f,",*layer1_image_out);
    // conv1
    //    static int conv_nxn_single(float *src, float *kernel, float bias, int n, int prev, float *dst, int h, int w, int padding)
    int i, imsize1, imsize2, input_channel, output_channel, ksize;
    float* conv_memory = NULL, *relu_memory = NULL, *pool_memory = NULL, *new_memory = NULL, *keep_memory = NULL, *bn_memory = NULL;

    new_memory = memory + h * w; // 160*160;//48*48;
    //--------------------layer2,conv_1------------------//
    imsize1 = hc * wc;
    imsize2 = ( hc / 2 ) * ( wc / 2 );
    input_channel = 1;
    output_channel = 8;
    ksize = 3;
    conv_memory = new_memory;
    bn_memory = conv_memory + imsize1;
    relu_memory = bn_memory + imsize1;
    //    relu_memory = conv_memory + output_channel*imsize1;
    pool_memory = relu_memory + imsize1;
    new_memory = pool_memory + output_channel * imsize2;
    //    keep_memory = pool_memory;

    for ( i = 0; i < output_channel; i++ ) {
        // conv+bias
        //;//printf("\n\nline:%d conv1\n",__LINE__);
#if NEON
        conv_nxn_single_bias_neon(layer1_image_out, para_hw_conv1_weights + i * input_channel * ksize * ksize,
                                  para_hw_conv1_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
#else
        conv_nxn_single_bias(layer1_image_out, para_hw_conv1_weights + i * input_channel * ksize * ksize,
                             para_hw_conv1_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
#endif
        // printf("conv1");
        //      print_txt(conv_memory + i*imsize1, hc, wc);//same
        // batch-normalization
        //;//printf("\n\nline:%d batch1\n",__LINE__);
        batchnormalization( conv_memory, hc, wc, bn_memory,
                            *(para_hw_norm1 + i), *(para_hw_norm1 + output_channel + i), epsilon, *(para_hw_norm1 + output_channel * 2 + i), *(para_hw_norm1 + output_channel * 3 + i));
        //  printf("bn1");
        //;//printf("\n\nline:%d relu1\n",__LINE__);
        // print_txt(bn_memory + i*imsize1, hc, wc);
        // relu
        ReLU( bn_memory, relu_memory, hc * wc );
        //      ReLU(conv_memory + i*imsize1, relu_memory + i*imsize1, hc*wc);
        // print_txt(relu_memory + i*imsize1, hc, wc);//ok
        // maxpooling
        //;printf("\n\nline:%d pool1\n",__LINE__);
        maxpooling_stride2_size2( relu_memory, hc, wc, pool_memory + i * imsize2 );
        // print_txt(pool_memory + i*imsize2, hc/2, wc/2);//not same
    }

    hc /= 2; // 80x80
    wc /= 2;

    //--------------------layer3,conv_2------------------//
    imsize1 = hc * wc;
    imsize2 = ( hc / 2 ) * ( wc / 2 );
    input_channel = 8;
    output_channel = 16; // 8;
    ksize = 3;
    conv_memory = new_memory;
    keep_memory = pool_memory;
    bn_memory = conv_memory + imsize1;
    relu_memory = bn_memory + imsize1;
    //    relu_memory = conv_memory + output_channel*imsize1;
    pool_memory = relu_memory + imsize1;
    new_memory = pool_memory + output_channel * imsize2;

    for ( i = 0; i < output_channel; i++ ) {
        // conv+bias
        //;//printf("\n\nline:%d conv2\n",__LINE__);
        // #if NEON
        //      conv_nxn_single_bias_neon(keep_memory, hw_conv2_weights + i * input_channel * ksize * ksize,
        //              hw_conv2_bias[i], ksize, input_channel, conv_memory + i * imsize1, &hc, &wc, 0);
        // #else
        conv_nxn_single_bias(keep_memory, para_hw_conv2_weights + i * input_channel * ksize * ksize,
                             para_hw_conv2_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
        // #endif
        //  printf("conv2");//not same

        // batch-normalization
        //;//printf("\n\nline:%d batch2,%f,%f,%f,%f,%f\n",__LINE__,*(fc_norm2+i), *(fc_norm2+output_channel+i), epsilon, *(fc_norm2+output_channel*2+i), *(fc_norm2+output_channel*3+i));
        batchnormalization( conv_memory, hc, wc, bn_memory,
                            *(para_hw_norm2 + i), *(para_hw_norm2 + output_channel + i), epsilon, *(para_hw_norm2 + output_channel * 2 + i), *(para_hw_norm2 + output_channel * 3 + i));
        //;//printf("\n\nline:%d relu2\n",__LINE__);
        // relu
        ReLU( bn_memory, relu_memory, hc * wc );
        // print_txt(relu_memory + i*imsize1, hc, wc);
        // maxpooling
        //;//printf("\n\nline:%d poool2\n",__LINE__);
        maxpooling_stride2_size2( relu_memory, hc, wc, pool_memory + i * imsize2 );
        // print_txt(pool_memory + i*imsize2, hc/2, wc/2);
    }

    hc /= 2; // 40x40
    wc /= 2;
    //--------------------layer4,conv_3------------------//
    imsize1 = hc * wc;
    imsize2 = ( hc / 2 ) * ( wc / 2 );
    input_channel = 16;  // 8;
    output_channel = 32; // 16;
    ksize = 3;
    conv_memory = new_memory;
    keep_memory = pool_memory;
    bn_memory = conv_memory + imsize1;
    relu_memory = bn_memory + imsize1;
    //    relu_memory = conv_memory + output_channel*imsize1;
    pool_memory = relu_memory + imsize1;
    new_memory = pool_memory + output_channel * imsize2;

    for ( i = 0; i < output_channel; i++ ) {
        // conv+bias
        //;//printf("\n\nline:%d conv3\n",__LINE__);
        // #if NEON
        //      conv_nxn_single_bias_neon(keep_memory, hw_conv3_weights + i * input_channel * ksize * ksize,
        //              hw_conv3_bias[i], ksize, input_channel, conv_memory + i * imsize1, &hc, &wc, 0);
        // #else
        conv_nxn_single_bias(keep_memory, para_hw_conv3_weights + i * input_channel * ksize * ksize,
                             para_hw_conv3_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
        // #endif
        //   printf("conv3");
        //  print_txt(conv_memory + i*imsize1, hc, wc); // v
        //  batch-normalization
        //;//printf("\n\nline:%d batch3\n",__LINE__);
        batchnormalization( conv_memory, hc, wc, bn_memory,
                            *(para_hw_norm3 + i), *(para_hw_norm3 + output_channel + i), epsilon, *(para_hw_norm3 + output_channel * 2 + i), *(para_hw_norm3 + output_channel * 3 + i));
        // relu
        //;//printf("\n\nline:%d relu3\n",__LINE__);
        ReLU( bn_memory, relu_memory, hc * wc );
        // print_txt(relu_memory + i*imsize1, hc, wc);
        // maxpooling
        //;//printf("\n\nline:%d pool3\n",__LINE__);
        maxpooling_stride2_size2( relu_memory, hc, wc, pool_memory + i * imsize2 );
        // print_txt(pool_memory + i*imsize2, hc/2, wc/2);
    }

    hc /= 2; // 20x20
    wc /= 2;
    //--------------------layer4,conv_4------------------//
    imsize1 = hc * wc;
    imsize2 = ( hc / 2 ) * ( wc / 2 );
    input_channel = 32;  // 16;
    output_channel = 64; // 32;
    ksize = 3;
    conv_memory = new_memory;
    keep_memory = pool_memory;
    bn_memory = conv_memory + imsize1;
    relu_memory = bn_memory + imsize1;
    //    relu_memory = conv_memory + output_channel*imsize1;
    pool_memory = relu_memory + imsize1;
    new_memory = pool_memory + output_channel * imsize2;

    for ( i = 0; i < output_channel; i++ ) {
        // conv+bias
        //;//printf("\n\nline:%d conv3\n",__line__);
        // #if neon
        //      conv_nxn_single_bias_neon(keep_memory, fc_hw_conv4_weights + i * input_channel * ksize * ksize,
        //              hw_conv4_bias[i], ksize, input_channel, conv_memory + i * imsize1, &hc, &wc, 0);
        // #else
        conv_nxn_single_bias(keep_memory, para_hw_conv4_weights + i * input_channel * ksize * ksize,
                             para_hw_conv4_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
        // #endif
        //   printf("conv3");
        //  print_txt(conv_memory + i*imsize1, hc, wc);// ok
        //  batch-normalization
        //;//printf("\n\nline:%d batch3\n",__line__);
        batchnormalization( conv_memory, hc, wc, bn_memory,
                            *(para_hw_norm4 + i), *(para_hw_norm4 + output_channel + i), epsilon, *(para_hw_norm4 + output_channel * 2 + i), *(para_hw_norm4 + output_channel * 3 + i));
        // relu
        //;//printf("\n\nline:%d relu3\n",__line__);
        ReLU( bn_memory, relu_memory, hc * wc );
        // print_txt(relu_memory + i*imsize1, hc, wc);
        // maxpooling
        //;//printf("\n\nline:%d pool3\n",__line__);
        maxpooling_stride2_size2( relu_memory, hc, wc, pool_memory + i * imsize2 );
        // print_txt(pool_memory + i*imsize2, hc/2, wc/2);
    }

    hc /= 2; // 10*10
    wc /= 2;

    //--------------------layer5,conv_5------------------//
    imsize1 = hc * wc;
    imsize2 = 1;
    input_channel = 64; // 32;
    output_channel = 3; // 2;//6;//5;//2;
    ksize = 3;
    conv_memory = new_memory;
    keep_memory = pool_memory;
    bn_memory = conv_memory + imsize1;
    relu_memory = bn_memory + imsize1;
    // relu_memory = conv_memory + output_channel*imsize1;
    pool_memory = relu_memory + output_channel * imsize1;
    new_memory = pool_memory + output_channel * imsize2;

    for ( i = 0; i < output_channel; i++ ) {
        // conv+bias
        //;//printf("\n\nline:%d conv4\n",__LINE__);
        conv_nxn_single_bias(keep_memory, para_hw_conv5_weights + i * input_channel * ksize * ksize,
                             para_hw_conv5_bias[i], ksize, input_channel, conv_memory, &hc, &wc, 0);
        // batch-normalization

        // printf("conv4");
        ; // printf("\n\nline:%d batch4\n",__LINE__);
        batchnormalization( conv_memory, hc, wc, bn_memory,
                            *(para_hw_norm5 + i), *(para_hw_norm5 + output_channel + i), epsilon, *(para_hw_norm5 + output_channel * 2 + i), *(para_hw_norm5 + output_channel * 3 + i));
        // relu
        //;//printf("\n\nline:%d relu4\n",__LINE__);
        ReLU( bn_memory, relu_memory + i * imsize1, hc * wc );
        // global average pooling
        //;//printf("\n\nline:%d pool4\n",__LINE__);
        // print_txt(relu_memory + i*imsize1, hc, wc);// pass
        // averagepooling(relu_memory + i * imsize1, hc, wc, pool_memory + i *imsize2);
        //      //;//printf("%f,", *(pool_memory + i *imsize2));
    }

    short fc_input_channel = 3;
    short class_number = 3;
    // print_txt(relu_memory, hc, wc);
    fullyconnect(relu_memory, hc, wc, fc_input_channel, class_number, para_hw_fc_weights, para_hw_fc_bias, pool_memory);

    // for(i=0;i<class_number;i++)
    //{
    // SL_LOGD("fc: s%d=%f",i,*(pool_memory+i));
    //}
    float* soft_memory = new_memory + class_number;
    softmax( pool_memory, class_number, soft_memory );
    // softmax(pool_memory, 2, soft_memory);
    //;//printf("\nfinal:\n%f\n%f", *soft_memory, *(soft_memory+1));
    //--------------------softmax------------------//

    //--------------------classify------------------//
    //*score = (int)(*(soft_memory+1)*65536);
    // if(*(soft_memory+2) >= *soft_memory)
    //{
    //*soft_memory = *(soft_memory+2);
    //}
    for ( i = 0; i < class_number; i++ ) {
        SL_LOGD( "s%d=%d", i, ( int )( soft_memory[i] * 100 ) );
    } // printf("s%d=%f",i,*(soft_memory+i));

    /***********
    class 0 : finger, 1 : mistouch,
    threshold =
    ************/

    *score = ( int )( ( *( soft_memory ) ) * 65536 );

    if ( memory ) {
        free( memory );
        memory = NULL;
    }

    return ret;
}

static int ImageExpansion( unsigned char* image, int* h_t, int* w_t, int dropUp, int dropDown, int dropLeft, int dropRight )
{
    if ( image == NULL || *h_t < 0 || ( *w_t < 0 ) || dropUp < 0 || dropLeft < 0 || dropRight < 0 ) {
        return -1;
    }

    int i;
    int j;
    unsigned char* temp;
    int h, w;
    w = *w_t + dropLeft + dropRight;
    h = *h_t + dropUp + dropDown;
    temp = ( unsigned char* )malloc( sizeof( unsigned char ) * h * w );

    if ( temp == NULL ) {
        SL_LOGE( "malloc fail." );
        return -1;
    }

    for ( j = 0; j < h; j++ ) {
        for ( i = 0; i < w; i++ ) {
            if ( ( i < dropLeft ) || ( i > ( w - dropRight - 1 ) ) ) {
                temp[j * w + i] = 255;
            } else {
                temp[j * w + i] = image[j * ( *w_t ) + i - dropLeft];
            }
        }
    }

    *w_t = w;
    *h_t = h;
    memcpy( image, temp, ( h ) * ( w ) );

    if ( temp ) {
        free( temp );
        temp = NULL;
    }

    return 0;
}

int mistouch_check( unsigned char* img, int width, int height, int* score )
{
    get_version_mistouch();

    // SL_LOGD("mistouch_check into");
    SL_LOGD( "mistouch_check image w=%d,h=%d", width, height ); // width = 36  height = 160

    if ( NULL == img || width <= 0 || height <= 0 || score == NULL ) {
        SL_LOGE( "mistouch_check param error" );
        return -1;
    }

    int ret;

    if ( width == 32 && height == 160 ) {
        ret = ImageExpansion( img, &height, &width, 0, 0, 2, 2 );

        if ( ret < 0 ) {
            SL_LOGE( "ImageExpansion fail" );
            return -1;
        }
    }

    float* out_resize = ( float* )malloc( width * ( height / 2 ) * sizeof( float ) );

    if ( NULL == out_resize ) {
        SL_LOGE( "mistouch_check malloc fail" );
        return -1;
    }

    // int i;
    // int j;
    ret = resize( img, height, width, out_resize );

    if ( ret < 0 ) {
        SL_LOGE( "resize param error" );
    }

    ret = sl_fingerprint_hw_spoof_0813( out_resize, 80, 36, score );

    if ( ret < 0 ) {
        SL_LOGE( "sl_fingerprint_hwcap6156skin_fc_v0612_1 error" );

        if ( out_resize ) {
            free( out_resize );
            out_resize = NULL;
        }

        return -1;
    }

    if ( out_resize ) {
        free( out_resize );
        out_resize = NULL;
    }

    SL_LOGE( "mistouch_check out score = %d", *score );
    return 0;
}
