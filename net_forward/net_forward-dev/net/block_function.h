#ifndef __BLOCK_FUN_H__
#define __BLOCK_FUN_H__

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "net_api.h"
#include "../alog.h"
#include "../SL_Math.h"
#include "../net_cnn_common.h"
#include "../net_struct_common.h"

/*spoof、mistouch*/
int char2float( unsigned char* src, float* dst, int h, int w);
Mat bottleneck_short(Mat input, Block_Bneck_Short* block);
Mat preblock_short(Mat input, Block_Conv3x3_Short* block);
/*mask、quality*/
void pooling2x1s2x1_max_neon(Mat bottom_blob, Mat top_blob);
Mat conv3x3_block_short(Mat input, Block_Conv3x3_Short* block, float* rightpointer, int pooling, int pad_type);
Mat bottleneck1_short(Mat input, Block_Bneck_SE_Short* block);
Mat bottleneck2_short(Mat input, Block_Bneck_SE_Short* block);
/*enhance、exp*/
int conv_block_s(Mat* input, Block_Conv3x3_Short* block, int kernel, int bool_relu);
int conv1x1_blocktool_s(Mat* input, Block_Conv1x1_Short* block, float* right);
int conv_group_single_block_s(Mat* input, Block_Conv3x3_Short* block, int kernel, int group);
int resnet3x3_blockdw_s(Mat* input, Block_Resnet_Short* block);
int conv_sep_block_s(Mat* input, Block_Sep_Short* block);
int deconv4x4_block_s(Mat* input, Block_Conv3x3_Short* block);

#endif