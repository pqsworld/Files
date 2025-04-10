#include"net.h"
#include"net_api.h"
#include"layer.h"
#include"parameters_float_alike_pnt.h"
Extractor* net_ex_init(){
  int layer_count = 60;
  Layer** layers = (Layer**)malloc(layer_count * sizeof(Layer*));
  for (int i = 0;i < layer_count; i++)
      layers[i] = (Layer*)malloc(sizeof(Layer));
  int blob_count = 61;
  Blob* blobs = (Blob*)malloc(blob_count * sizeof(Blob));
  DEFINE_LAYER(layers[0],Padding,padding_forward,NULL,1,0,0,list(0,),1,list(1,),1,list(1,1,0,0));
  DEFINE_LAYER(layers[1],Conv,conv3x3s1_forward,NULL,1,0,1,list(1,),1,list(2,),1,list(param_alike_pnt+0,param_alike_pnt+72,72,8,6031,45228,NULL,NULL,1));
  DEFINE_LAYER(layers[2],bool,NULL,relu_forward_inplace,1,1,2,list(2,),1,list(3,),1,list(0));
  DEFINE_LAYER(layers[3],Padding,padding_forward,NULL,1,0,3,list(3,),1,list(4,),1,list(1,1,0,0));
  DEFINE_LAYER(layers[4],Conv,conv3x3s1_forward,NULL,1,0,4,list(4,),1,list(5,),1,list(param_alike_pnt+80,param_alike_pnt+656,576,8,20646,20775,NULL,NULL,1));
  DEFINE_LAYER(layers[5],bool,NULL,relu_forward_inplace,1,1,5,list(5,),1,list(6,),1,list(0));
  DEFINE_LAYER(layers[6],Pool,max_pool2d_forward,NULL,1,0,6,list(6,),1,list(7,),1,list(2,2,2,2));
  DEFINE_LAYER(layers[7],Conv,conv1x1s1_forward,NULL,1,0,7,list(7,),1,list(8,),1,list(param_alike_pnt+664,param_alike_pnt+792,128,16,14684,20104,NULL,NULL,1));
  DEFINE_LAYER(layers[8],bool,NULL,hardswish_forward_inplace,1,1,8,list(8,),1,list(9,),1,list(0));
  DEFINE_LAYER(layers[9],Padding,padding_forward,NULL,1,0,9,list(9,),1,list(10,),1,list(1,1,0,0));
  DEFINE_LAYER(layers[10],Conv,convdw3x3s1_forward,NULL,1,0,10,list(10,),1,list(11,),1,list(param_alike_pnt+808,param_alike_pnt+952,144,16,2227,20885,NULL,NULL,16));
  DEFINE_LAYER(layers[11],bool,NULL,hardswish_forward_inplace,1,1,11,list(11,),1,list(12,),1,list(0));
  DEFINE_LAYER(layers[12],Conv,conv1x1s1_forward,NULL,1,0,12,list(12,),1,list(13,),1,list(param_alike_pnt+968,param_alike_pnt+1224,256,16,11337,6013,NULL,NULL,1));
  DEFINE_LAYER(layers[13],bool,adaptive_avg_pool2d_forward,NULL,1,0,13,list(13,),1,list(14,),1,list(0));
  DEFINE_LAYER(layers[14],Conv,conv1x1s1_forward,NULL,1,0,14,list(14,),1,list(15,),1,list(param_alike_pnt+1240,param_alike_pnt+1304,64,4,21430,11305,NULL,NULL,1));
  DEFINE_LAYER(layers[15],bool,NULL,relu_forward_inplace,1,1,15,list(15,),1,list(16,),1,list(0));
  DEFINE_LAYER(layers[16],Conv,conv1x1s1_forward,NULL,1,0,16,list(16,),1,list(17,),1,list(param_alike_pnt+1308,param_alike_pnt+1372,64,16,34343,21193,NULL,NULL,1));
  DEFINE_LAYER(layers[17],bool,NULL,hardsigmoid_forward_inplace,1,1,17,list(17,),1,list(18,),1,list(0));
  DEFINE_LAYER(layers[18],bool,NULL,mul_forward_inplace,0,1,18,list(13, 18),2,list(19,),1,list(0));
  DEFINE_LAYER(layers[19],Pool,max_pool2d_forward,NULL,1,0,19,list(19,),1,list(20,),1,list(2,2,2,2));
  DEFINE_LAYER(layers[20],Conv,conv1x1s1_forward,NULL,1,0,20,list(20,),1,list(21,),1,list(param_alike_pnt+1388,param_alike_pnt+1900,512,32,19387,28451,NULL,NULL,1));
  DEFINE_LAYER(layers[21],bool,NULL,hardswish_forward_inplace,1,1,21,list(21,),1,list(22,),1,list(0));
  DEFINE_LAYER(layers[22],Padding,padding_forward,NULL,1,0,22,list(22,),1,list(23,),1,list(1,1,0,0));
  DEFINE_LAYER(layers[23],Conv,convdw3x3s1_forward,NULL,1,0,23,list(23,),1,list(24,),1,list(param_alike_pnt+1932,param_alike_pnt+2220,288,32,3668,6359,NULL,NULL,32));
  DEFINE_LAYER(layers[24],bool,NULL,hardswish_forward_inplace,1,1,24,list(24,),1,list(25,),1,list(0));
  DEFINE_LAYER(layers[25],Conv,conv1x1s1_forward,NULL,1,0,25,list(25,),1,list(26,),1,list(param_alike_pnt+2252,param_alike_pnt+3276,1024,32,19412,21519,NULL,NULL,1));
  DEFINE_LAYER(layers[26],bool,adaptive_avg_pool2d_forward,NULL,1,0,26,list(26,),1,list(27,),1,list(0));
  DEFINE_LAYER(layers[27],Conv,conv1x1s1_forward,NULL,1,0,27,list(27,),1,list(28,),1,list(param_alike_pnt+3308,param_alike_pnt+3564,256,8,53841,26530,NULL,NULL,1));
  DEFINE_LAYER(layers[28],bool,NULL,relu_forward_inplace,1,1,28,list(28,),1,list(29,),1,list(0));
  DEFINE_LAYER(layers[29],Conv,conv1x1s1_forward,NULL,1,0,29,list(29,),1,list(30,),1,list(param_alike_pnt+3572,param_alike_pnt+3828,256,32,23791,13587,NULL,NULL,1));
  DEFINE_LAYER(layers[30],bool,NULL,hardsigmoid_forward_inplace,1,1,30,list(30,),1,list(31,),1,list(0));
  DEFINE_LAYER(layers[31],bool,NULL,mul_forward_inplace,0,1,31,list(26, 31),2,list(32,),1,list(0));
  DEFINE_LAYER(layers[32],Pool,max_pool2d_forward,NULL,1,0,32,list(32,),1,list(33,),1,list(2,2,2,2));
  DEFINE_LAYER(layers[33],Conv,conv1x1s1_forward,NULL,1,0,33,list(33,),1,list(34,),1,list(param_alike_pnt+3860,param_alike_pnt+5908,2048,64,21571,16301,NULL,NULL,1));
  DEFINE_LAYER(layers[34],bool,NULL,hardswish_forward_inplace,1,1,34,list(34,),1,list(35,),1,list(0));
  DEFINE_LAYER(layers[35],Padding,padding_forward,NULL,1,0,35,list(35,),1,list(36,),1,list(1,1,0,0));
  DEFINE_LAYER(layers[36],Conv,convdw3x3s1_forward,NULL,1,0,36,list(36,),1,list(37,),1,list(param_alike_pnt+5972,param_alike_pnt+6548,576,64,1201,10672,NULL,NULL,64));
  DEFINE_LAYER(layers[37],bool,NULL,hardswish_forward_inplace,1,1,37,list(37,),1,list(38,),1,list(0));
  DEFINE_LAYER(layers[38],Conv,conv1x1s1_forward,NULL,1,0,38,list(38,),1,list(39,),1,list(param_alike_pnt+6612,param_alike_pnt+10708,4096,64,24556,22956,NULL,NULL,1));
  DEFINE_LAYER(layers[39],bool,adaptive_avg_pool2d_forward,NULL,1,0,39,list(39,),1,list(40,),1,list(0));
  DEFINE_LAYER(layers[40],Conv,conv1x1s1_forward,NULL,1,0,40,list(40,),1,list(41,),1,list(param_alike_pnt+10772,param_alike_pnt+11796,1024,16,61933,45971,NULL,NULL,1));
  DEFINE_LAYER(layers[41],bool,NULL,relu_forward_inplace,1,1,41,list(41,),1,list(42,),1,list(0));
  DEFINE_LAYER(layers[42],Conv,conv1x1s1_forward,NULL,1,0,42,list(42,),1,list(43,),1,list(param_alike_pnt+11812,param_alike_pnt+12836,1024,64,10486,7019,NULL,NULL,1));
  DEFINE_LAYER(layers[43],bool,NULL,hardsigmoid_forward_inplace,1,1,43,list(43,),1,list(44,),1,list(0));
  DEFINE_LAYER(layers[44],bool,NULL,mul_forward_inplace,0,1,44,list(39, 44),2,list(45,),1,list(0));
  DEFINE_LAYER(layers[45],Conv,conv1x1s1_forward,NULL,1,0,45,list(6,),1,list(46,),1,list(param_alike_pnt+12900,param_alike_pnt+12964,64,8,19960,28743,NULL,NULL,1));
  DEFINE_LAYER(layers[46],bool,NULL,relu_forward_inplace,1,1,46,list(46,),1,list(47,),1,list(0));
  DEFINE_LAYER(layers[47],Conv,conv1x1s1_forward,NULL,1,0,47,list(19,),1,list(48,),1,list(param_alike_pnt+12972,param_alike_pnt+13100,128,8,16786,28396,NULL,NULL,1));
  DEFINE_LAYER(layers[48],bool,NULL,relu_forward_inplace,1,1,48,list(48,),1,list(49,),1,list(0));
  DEFINE_LAYER(layers[49],Conv,conv1x1s1_forward,NULL,1,0,49,list(32,),1,list(50,),1,list(param_alike_pnt+13108,param_alike_pnt+13364,256,8,29111,35519,NULL,NULL,1));
  DEFINE_LAYER(layers[50],bool,NULL,relu_forward_inplace,1,1,50,list(50,),1,list(51,),1,list(0));
  DEFINE_LAYER(layers[51],Conv,conv1x1s1_forward,NULL,1,0,51,list(45,),1,list(52,),1,list(param_alike_pnt+13372,param_alike_pnt+13884,512,8,24247,48863,NULL,NULL,1));
  DEFINE_LAYER(layers[52],bool,NULL,relu_forward_inplace,1,1,52,list(52,),1,list(53,),1,list(0));
  DEFINE_LAYER(layers[53],bool,upsample_bilinear2d_forward,NULL,1,0,53,list(49,),1,list(54,),1,list(0));
  DEFINE_LAYER(layers[54],bool,upsample_bilinear2d_forward,NULL,1,0,54,list(51,),1,list(55,),1,list(0));
  DEFINE_LAYER(layers[55],bool,upsample_bilinear2d_forward,NULL,1,0,55,list(53,),1,list(56,),1,list(0));
  DEFINE_LAYER(layers[56],Concat,cat_forward,NULL,0,0,56,list(47, 54, 55, 56),4,list(57,),1,list(1));
  DEFINE_LAYER(layers[57],Conv,conv1x1s1_forward,NULL,1,0,57,list(57,),1,list(58,),1,list(param_alike_pnt+13892,param_alike_pnt+13988,96,3,15841,74845,NULL,NULL,1));
  DEFINE_LAYER(layers[58],bool,NULL,sigmoid_forward_inplace,1,1,58,list(58,),1,list(59,),1,list(0));
  DEFINE_BLOB(&blobs[0],-1,list(0,),322560,list(NULL,1,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[1],0,list(1,),0,list(NULL,1,130,42,5472,0,4,0,0));
  DEFINE_BLOB(&blobs[2],1,list(2,),286720,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[3],2,list(3,),-1,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[4],3,list(4,),0,list(NULL,8,130,42,5472,0,4,0,0));
  DEFINE_BLOB(&blobs[5],4,list(5,),286720,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[6],5,list(6, 45),-1,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[7],6,list(7,),0,list(NULL,8,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[8],7,list(8,),266240,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[9],8,list(9,),-1,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[10],9,list(10,),0,list(NULL,16,66,22,1456,0,4,0,0));
  DEFINE_BLOB(&blobs[11],10,list(11,),266240,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[12],11,list(12,),-1,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[13],12,list(13, 18),56320,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[14],13,list(14,),286464,list(NULL,16,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[15],14,list(15,),0,list(NULL,4,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[16],15,list(16,),-1,list(NULL,4,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[17],16,list(17,),286464,list(NULL,16,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[18],17,list(18,),-1,list(NULL,16,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[19],18,list(19, 47),-1,list(NULL,16,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[20],19,list(20,),0,list(NULL,16,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[21],20,list(21,),276480,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[22],21,list(22,),-1,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[23],22,list(23,),0,list(NULL,32,34,12,416,0,4,0,0));
  DEFINE_BLOB(&blobs[24],23,list(24,),276480,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[25],24,list(25,),-1,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[26],25,list(26, 31),46080,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[27],26,list(27,),286208,list(NULL,32,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[28],27,list(28,),0,list(NULL,8,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[29],28,list(29,),-1,list(NULL,8,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[30],29,list(30,),286208,list(NULL,32,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[31],30,list(31,),-1,list(NULL,32,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[32],31,list(32, 49),-1,list(NULL,32,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[33],32,list(33,),0,list(NULL,32,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[34],33,list(34,),281600,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[35],34,list(35,),-1,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[36],35,list(36,),0,list(NULL,64,18,7,128,0,4,0,0));
  DEFINE_BLOB(&blobs[37],36,list(37,),281600,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[38],37,list(38,),-1,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[39],38,list(39, 44),40960,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[40],39,list(40,),285696,list(NULL,64,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[41],40,list(41,),0,list(NULL,16,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[42],41,list(42,),-1,list(NULL,16,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[43],42,list(43,),285696,list(NULL,64,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[44],43,list(44,),-1,list(NULL,64,1,1,16,0,4,0,0));
  DEFINE_BLOB(&blobs[45],44,list(51,),-1,list(NULL,64,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[46],45,list(46,),0,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[47],46,list(56,),-1,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[48],47,list(48,),314240,list(NULL,8,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[49],48,list(53,),-1,list(NULL,8,64,20,1280,0,4,0,0));
  DEFINE_BLOB(&blobs[50],49,list(50,),324480,list(NULL,8,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[51],50,list(54,),-1,list(NULL,8,32,10,320,0,4,0,0));
  DEFINE_BLOB(&blobs[52],51,list(52,),327040,list(NULL,8,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[53],52,list(55,),-1,list(NULL,8,16,5,80,0,4,0,0));
  DEFINE_BLOB(&blobs[54],53,list(56,),40960,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[55],54,list(56,),81920,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[56],55,list(56,),122880,list(NULL,8,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[57],56,list(57,),163840,list(NULL,32,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[58],57,list(58,),0,list(NULL,3,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[59],58,list(59,),-1,list(NULL,3,128,40,5120,0,4,0,0));
  DEFINE_BLOB(&blobs[60],59,list(59,),-1,list(NULL,0,0,0,0,0,4,0,0));
  Option opt = { 1,0,0,0,NULL };
  Net _net = { forward_layer, opt, layers, blobs, layer_count , blob_count};
  Net* net = (Net*)malloc(sizeof(Net));
  *net = _net;
  Extractor* ex = (Extractor*)malloc(sizeof(Extractor));
  extractor(ex, net, net->blob_count);
  return ex;
}
int detect_alike_pnt(const Mat **bgr, Mat **out,float* data,Extractor* ex)
{
#if NCNN_VULKAN
  mobilenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
  int blob_number=60;
for (int i = 0; i < blob_number; i++)

    {

        if (ex->net->blobs[i].mat_addressoffect != -1)

            ex->blob_mats[i].data = data + ex->net->blobs[i].mat_addressoffect;

        else
            ex->blob_mats[i].data = -1;

    }
  int input_no[1]={0};
  int input_num=sizeof(input_no)/sizeof(int);
  for(int i=0;i<input_num;i++)
      ex->input(ex, i, *bgr[i]);
  for (int i = 1; i <blob_number; i++)
{
    if (ex->net->blobs[i].producer==-1)
        continue;
    ex->extract(ex, i);
}
  for (int i = 0; i < ex->net->blobs[blob_number].consumer_num; i++)
{
  memcpy((*out[i]).data, (float*)(ex->blob_mats[ex->net->blobs[blob_number].consumers[i]].data), total(*out[i])*(*out[i]).elemsize);
}
  return 0;
}
void main()//名称及参数需要改成与原先接口一致
{
     float* data = (float*)malloc(327680 * sizeof(float));
Extractor* ex=net_ex_init();
Mat **input= (Mat**)malloc(() * sizeof(Mat*));//TODO  需要填写输入数目
Mat **output= (Mat**)malloc(() * sizeof(Mat*));/TODO  需要填写输出数目
//TODO  从图像到MAT类型的预处理部分
ret=detect_alike_pnt(input, output,data,ex);
//TODO 从MAT转到输出的后面处理部分 
     free(data);
}
