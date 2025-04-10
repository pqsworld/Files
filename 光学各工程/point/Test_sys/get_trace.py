#from MobileNet import *
from pathlib import Path
from collections import deque
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

import argparse
import torch
import os
import torch.nn as nn
from tensorboard.compat.proto.graph_pb2 import *
from tensorboardX import writer,SummaryWriter
from torch.utils.tensorboard._pytorch_graph import *
#from par import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#from models.mobilenetv3 import *
#from models.networks import *
#from models.zsn import *
#import tensorflow as tf
import copy

import networkx as nx
import matplotlib.pyplot as plt


padding_dict={"zeros":0,"replicate":1,"reflect":2,"circular":3}
upsample_dict={"nearest":0,"linear":1,"bilinear":2,"bicubic":3,"trilinear":4,"area":5}
basic_module=[nn.Conv2d,nn.BatchNorm2d,nn.ConvTranspose2d,nn.Sigmoid,nn.ReLU,nn.LeakyReLU,nn.Hardswish,nn.Hardsigmoid,nn.Tanh,nn.AdaptiveMaxPool2d,nn.AdaptiveAvgPool2d,nn.AvgPool2d,nn.MaxPool2d,nn.Dropout2d,nn.Upsample]

def layer_input_count(node,list_in,dict_of_node,list_input,list_hyper_param):
    if not node.input:
        value_input=node.attr["attr"].s.decode('utf-8')
        value_input=value_input.replace("{","").replace("}","").replace(" ","")
        if len(value_input)>0:
            #print(value_input.split(":")[1])
            list_hyper_param.append(float(value_input.split(":")[1]))
        else:
             list_hyper_param.append("NULL")
    else:
        for input_name in node.input:
            #print("input_start")
            if input_name in list_in:
                #print("input_yes")
                list_input.append(list_in.index(input_name))
            else:
                
                if input_name in dict_of_node.keys():
                    layer_input_count(dict_of_node[input_name],list_in,dict_of_node,list_input,list_hyper_param)
                else:
                    list_hyper_param.append("NULL")
                    continue

def blob_input_count(node,list_in,dict_of_node,list_input):
    if node.name in list_in:
        list_input.append(list_in.index(node.name))
    else:
        for input_name in node.input:
            if input_name in list_in:
                list_input.append(list_in.index(input_name))
            else:
                blob_input_count(dict_of_node[input_name],list_in,dict_of_node,list_input)


def get_trace_graph(net,x):
    """获取各层之间的关联性，即网络结构图，输出的是所有层的名称list，以及Graph的邻接矩阵"""
    list_of_layer=[]
    list_of_blob=[]
    dict_of_layer={}
    dict_of_node={}
    dict_of_blob={}
    with torch.onnx.select_model_mode_for_export(net, torch.onnx.TrainingMode.EVAL):  # TODO: move outside of torch.onnx?
        try:
            trace = torch.jit.trace(net,x)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e

    list_of_nodes = parse(graph, trace, x)
    # print(list_of_nodes)
    # exit()
    for node in list_of_nodes:
        dict_of_node.update({node.name:node})
        if "_output_shapes" in node.attr.keys() and "output/output" not in node.name:
            if(node.attr["_output_shapes"].list.shape[0].dim):
                list_of_blob.append(node.name)
        elif "output/output" in node.name:
            list_output=node.name
        if "aten::" in node.op and "_output_shapes" in node.attr.keys():
            list_of_layer.append(node.name)
    list_of_blob.append(list_output)
    list_of_layer.append(list_output)
    # for i in range(len(list_of_blob)):
    #     print(i, list_of_blob[i])
    # exit()
    layer_len=len(list_of_layer)
    blob_len=len(list_of_blob)
    blob_matrix_layer=np.zeros((blob_len,layer_len),dtype=np.uint8)
    layer_matrix_blob=np.zeros((layer_len,blob_len),dtype=np.uint8)

    for name in list_of_layer:
        list_input_layer=[]
        list_hyper_param_layer=[]
        node=dict_of_node[name]
        layer_input_count(node,list_of_blob,dict_of_node,list_input_layer,list_hyper_param_layer)
        
        dict_of_layer.update({name:list_hyper_param_layer})
        for input_i in list_input_layer:
            blob_matrix_layer[input_i,list_of_layer.index(name)]=list_input_layer.index(input_i)+1
    
    for name in list_of_blob:
        
        list_input_blob=[]
        node=dict_of_node[name]
        
        if not "output/output" in name:
            out_dim=np.ones(4)
            
            for dim_i in range(len(node.attr["_output_shapes"].list.shape[0].dim)):
                out_dim[dim_i]=node.attr["_output_shapes"].list.shape[0].dim[dim_i].size
          
            blob_input_count(node,list_of_layer,dict_of_node,list_input_blob)
            if ("aten::mul" == node.op or "aten::add" == node.op or "aten::sub" == node.op or "aten::leaky_relu"==node.op or
                "aten::leaky_relu_"==node.op or "aten::relu" == node.op or "aten::tanh" == node.op or "aten::hardswish" == node.op 
                or "aten::hardsigmoid" == node.op or"aten::sigmoid" == node.op or  "aten::relu_" == node.op or "aten::tanh_" == node.op
                 or "aten::hardswish_" == node.op or "aten::hardsigmoid_" == node.op or"aten::sigmoid_" == node.op or "aten::softmax"== node.op):
                dict_of_blob.update({name:[out_dim[0],out_dim[1],out_dim[2],out_dim[3],1]})
            else:
                dict_of_blob.update({name:[out_dim[0],out_dim[1],out_dim[2],out_dim[3],0]})
        elif "output/output" in name:
            blob_input_count(node,list_of_layer,dict_of_node,list_input_blob)
      
        for input_i in list_input_blob:
           
            layer_matrix_blob[input_i,list_of_blob.index(name)]=list_input_blob.index(input_i)+1
   
    for name in reversed(list_of_layer):
        node=dict_of_node[name]
        if node.op=="aten::_convolution":
            node_hyperparam=dict_of_layer[name]
           
            if node_hyperparam[8]==0 and node_hyperparam[4]!=0 and node_hyperparam[5]!=0:##针对卷积padding的拆分

                index_layer=list_of_layer.index(name)
                layer_input=np.where(blob_matrix_layer[:,index_layer]==1)[0][0]
                list_of_layer.insert(index_layer,name+"/Padding")
                list_of_blob.insert(layer_input+1,list_of_blob[layer_input]+"/Padding")
                blob_matrix_layer[layer_input,index_layer]=0
                node_input=dict_of_node[list_of_blob[layer_input]]
                dict_of_layer.update({name+"/Padding":[node_hyperparam[4],node_hyperparam[5]]})

                blob_c=node_input.attr["_output_shapes"].list.shape[0].dim[1].size
                blob_h=node_input.attr["_output_shapes"].list.shape[0].dim[2].size+2*node_hyperparam[4]
                blob_w=node_input.attr["_output_shapes"].list.shape[0].dim[3].size+2*node_hyperparam[5]
                dict_of_blob.update({list_of_blob[layer_input]+"/Padding":[1,blob_c,blob_h,blob_w,0]})

                y_matrix=np.zeros(blob_matrix_layer.shape[0],dtype=np.uint8)
                blob_matrix_layer=np.insert(blob_matrix_layer,index_layer,y_matrix,axis=1)
                x_matrix=np.zeros(blob_matrix_layer.shape[1],dtype=np.uint8)
                blob_matrix_layer=np.insert(blob_matrix_layer,layer_input+1,x_matrix,axis=0)
                blob_matrix_layer[layer_input+1,index_layer+1]=1
                blob_matrix_layer[layer_input,index_layer]=1

                y_matrix=np.zeros(layer_matrix_blob.shape[0],dtype=np.uint8)
                layer_matrix_blob=np.insert(layer_matrix_blob,layer_input+1,y_matrix,axis=1)
                x_matrix=np.zeros(layer_matrix_blob.shape[1],dtype=np.uint8)
                layer_matrix_blob=np.insert(layer_matrix_blob,index_layer,x_matrix,axis=0)
                layer_matrix_blob[index_layer,layer_input+1]=1

                
            elif  node_hyperparam[8]==1:##针对反卷积deconvcrop的拆分
                index_layer=list_of_layer.index(name)
                layer_output=np.where(layer_matrix_blob[index_layer,:]==1)[0][0]
                # np.set_printoptions(threshold=np.inf)
                # print(layer_output)
                node_output=dict_of_node[list_of_blob[layer_output]]
                blob_c=node_output .attr["_output_shapes"].list.shape[0].dim[1].size
                blob_h=node_output.attr["_output_shapes"].list.shape[0].dim[2].size
                blob_w=node_output.attr["_output_shapes"].list.shape[0].dim[3].size
                dict_of_blob.update({name:[1,blob_c,blob_h,blob_w,0]})
                dict_of_layer.update({name+"/Deconvcrop":[node_hyperparam[4],node_hyperparam[5],node_hyperparam[9],node_hyperparam[10]]})
                blob_h=blob_h+2*node_hyperparam[4]
                blob_w=blob_w+2*node_hyperparam[5]
                dict_of_blob.update({list_of_blob[layer_output]+"/Deconvcrop":[1,blob_c,blob_h,blob_w,0]})
                list_of_layer.insert(index_layer+1,name+"/Deconvcrop")
                list_of_blob.insert(layer_output,list_of_blob[layer_output]+"/Deconvcrop")
                layer_matrix_blob[index_layer,layer_output]=0
              
                y_matrix=np.zeros(layer_matrix_blob.shape[0],dtype=np.uint8)
                layer_matrix_blob=np.insert(layer_matrix_blob,layer_output,y_matrix,axis=1)
                x_matrix=np.zeros(layer_matrix_blob.shape[1],dtype=np.uint8)
                layer_matrix_blob=np.insert(layer_matrix_blob,index_layer+1,x_matrix,axis=0)
                layer_matrix_blob[index_layer,layer_output]=1
                layer_matrix_blob[index_layer+1,layer_output+1]=1

                y_matrix=np.zeros(blob_matrix_layer.shape[0],dtype=np.uint8)
                blob_matrix_layer=np.insert(blob_matrix_layer,index_layer+1,y_matrix,axis=1)
                x_matrix=np.zeros(blob_matrix_layer.shape[1],dtype=np.uint8)
                blob_matrix_layer=np.insert(blob_matrix_layer,layer_output,x_matrix,axis=0)
                blob_matrix_layer[layer_output,index_layer+1]=1
        if "dropout" in node.op or node.op=="aten::batch_norm" or node.op=="aten::contiguous":#针对dropout和batchnorm的删除
            index_layer=list_of_layer.index(name)
            layer_input=np.where(blob_matrix_layer[:,index_layer]==1)[0][0]
            list_of_layer.pop(index_layer)
            list_of_blob.pop(layer_input)
            layer_matrix_blob=np.delete(layer_matrix_blob,index_layer,axis=0)
            layer_matrix_blob=np.delete(layer_matrix_blob,layer_input,axis=1)
            layer_matrix_blob[index_layer-1,layer_input]=1
            blob_matrix_layer=np.delete(blob_matrix_layer,index_layer,axis=1)
            blob_matrix_layer=np.delete(blob_matrix_layer,layer_input,axis=0)
    return list_of_layer, dict_of_layer,list_of_blob,dict_of_blob,blob_matrix_layer,layer_matrix_blob,dict_of_node



class Layers(object):
    class Padding(object):
        def __init__(self,pad_top=0,pad_left=0,pad_type=0,pad_v=0):
            self.top=pad_top
            self.left=pad_left
            self.type=pad_type
            self.v=pad_v

    class Conv(object):
        def __init__(self,conv_weight="NULL",conv_bias="NULL",conv_weight_data_size=0,
        conv_bias_data_size=0,conv_weight_short_scale=0,conv_bias_short_scale=0,conv_weight_f="NULL",conv_bias_f="NULL",conv_group=1):
            self.weight=conv_weight
            self.bias=conv_bias
            self.weight_data_size=conv_weight_data_size
            self.bias_data_size=conv_bias_data_size
            self.weight_short_scale=conv_weight_short_scale
            self.bias_short_scale=conv_bias_short_scale
            self.weight_f=conv_weight_f
            self.bias_f=conv_bias_f
            self.group=conv_group
    class leakyrelu(object):
        def __init__(self,leakyrelu_slope=0):
            self.slope=leakyrelu_slope
    class pooling(object):
        def __init__(self,pooling_type):
            self.type=pooling_type
    class Blob(object):
        def __init__(
            self,
            blob_producer:int=0,
            blob_consumers:list=None,
            blob_consumers_count:int=0,
            blob_mat_data:str="NULL",
            blob_mat_w:int=0,
            blob_mat_h:int=0,
            blob_mat_c:int=0,
            blob_mat_cstep:int=0,
            blob_mat_refcount:int=0,
            blob_mat_elemsize:int=4,
            blob_mat_elempack:int=0,
            blob_mat_dims:int=0,
        ):
            self.producer=blob_producer
            self.consumers=blob_consumers
            self.consumers_count=blob_consumers_count
            self.mat_data=blob_mat_data
            self.mat_w=blob_mat_w
            self.mat_h=blob_mat_h
            self.mat_c=blob_mat_c
            self.mat_cstep=blob_mat_cstep
            self.mat_refcount=blob_mat_refcount
            self.mat_elemsize=blob_mat_elemsize
            self.mat_elempack=blob_mat_elempack
            self.mat_dims=blob_mat_dims
    class Upsample(object):
        def __init__(self,upsample_scale_factor_h=0,upsample_scale_factor_w=0,upsample_size_h=0,upsample_size_w=0,upsample_mode=2,upsample_align_corners=False):
            self.scale_factor_h=upsample_scale_factor_h
            self.scale_factor_w=upsample_scale_factor_w
            self.size_h=upsample_size_h
            self.size_w=upsample_size_w
            self.mode=upsample_mode
            self.align_corners=upsample_align_corners
    class  Concat(object):
        def __init__(self,cat_axis):
            self.axis=cat_axis
    def __init__(
        self,
        forward:str="NULL",
        forward_inplace:str="NULL",
        one_blob_only:bool=True,
        support_inplace:bool=False,
        typeindex:int=0,
        bottoms:list=None,
        bottoms_count:int=0,
        tops:list=None,
        tops_count:int=0,
        pad_top:int=0,
        pad_left:int=0,
        pad_type:int=0,
        pad_v:float=0,
        conv_weight:str="NULL",
        conv_bias:str="NULL",
        conv_weight_data_size:int=0,
        conv_bias_data_size:int=0,
        conv_weight_short_scale:int=0,
        conv_bias_short_scale:int=0,
        conv_weight_f:str="NULL",
        conv_bias_f:str="NULL",
        conv_group:int=1,
        leakyrelu_slope:float=0,
        pooling_type:int=0,
        blob_producer:int=0,
        blob_consumers:list=None,
        blob_consumers_count:int=0,
        blob_mat_data:str="NULL",
        blob_mat_w:int=0,
        blob_mat_h:int=0,
        blob_mat_c:int=0,
        blob_mat_cstep:int=0,
        blob_mat_refcount:int=0,
        blob_mat_elemsize:int=4,
        blob_mat_elempack:int=0,
        blob_mat_dims:int=0,
        upsample_scale_factor_h:float=0,
        upsample_scale_factor_w:float=0,
        upsample_size_h:int=0,
        upsample_size_w:int=0,
        upsample_mode:int=2,
        upsample_align_corners:bool=False,
        cat_axis:int=0
    ):
        self.forward=forward
        self.forward_inplace=forward_inplace
        self.one_blob_only=one_blob_only
        self.support_inplace=support_inplace
        self.typeindex=typeindex
        self.bottoms=bottoms
        self.bottoms_count=bottoms_count
        self.tops=tops
        self.tops_count=tops_count
        self.Padding=Layers.Padding(pad_top,pad_left,pad_type,pad_v)
        self.Conv=Layers.Conv( conv_weight,conv_bias,conv_weight_data_size,conv_bias_data_size,conv_weight_short_scale,conv_bias_short_scale,conv_weight_f,conv_bias_f,conv_group)
        self.leakyrelu=Layers.leakyrelu(leakyrelu_slope)
        self.pooling=Layers.pooling(pooling_type)
        self.Blob=Layers.Blob(blob_producer,blob_consumers,blob_consumers_count,blob_mat_data,blob_mat_w,blob_mat_h,blob_mat_c,blob_mat_cstep,blob_mat_refcount,
                                blob_mat_elemsize,blob_mat_elempack,blob_mat_dims)
        self.Upsample=Layers.Upsample(upsample_scale_factor_h,upsample_scale_factor_w,upsample_size_h,upsample_size_w,upsample_mode,upsample_align_corners)
        self.Concat=Layers.Concat(cat_axis)

def list_layers(layer,layer_name,name_node):
    """递归函数,如果非基本模块则递归,否则输出layer结果"""
    layers = []
    name_layers=[]
    flag=0
    name_node=name_node+'/'+str(type(layer)).replace(">","").replace("<","").replace("'","").split('.')[-1]+"["+str(layer_name)+']'
    for i in basic_module:
        if isinstance(layer,i):
            layers.append(layer)
            name_layers.append(name_node)
            flag=1
            break
    if (flag==0):
        for name,i in layer.named_children():
            layers.extend(list_layers(i,name,name_node)[0])
            name_layers.extend(list_layers(i,name,name_node)[1])
    return layers,name_layers

def get_parameters_layer(net,name_node):
    """遍历网络层并输出list"""
    layers = []
    name_layers=[]
    queue = deque()
    name_queue=deque()
    for name,i in net.named_children():
        queue.append(i)
        name_queue.append(name)
    while len(queue):
        root = queue.popleft()
        name_root=name_queue.popleft()
        layers.extend(list_layers(root,name_root,name_node)[0])
        name_layers.extend(list_layers(root,name_root,name_node)[1])
    return layers,name_layers

def get_layer_param(layers,name_layers,param_name):
    """获取每一层类型及超参数，参数等信息，并通过字典形式返回名称和对应信息"""
    net_dict={}
    address_header=0
    parameters_float=np.zeros(0)
    parameters_short=np.zeros(0)
    Max_param=0
    for i in range(len(layers)):
        list_net=[]
        layer=layers[i]
        name_layer=name_layers[i]
        weight_bias_f_space=0
        if isinstance(layers[i],nn.Conv2d):
            kargs={"pad_top":layer.padding[0],"pad_left":layer.padding[1],"pad_type":padding_dict[layer.padding_mode],"pad_v":0}
            Layers_layer=Layers("padding_forward","NULL",True,False,**kargs)
            net_dict.update({"{}/Padding".format(name_layer):Layers_layer})

            if layer.groups==1:            
                forward="conv{}x{}s{}_forward".format(layer.kernel_size[0],layer.kernel_size[1],layer.stride[0]) 
            elif layer.groups==layer.in_channels:
                forward="convdw{}x{}s{}_forward".format(layer.kernel_size[0],layer.kernel_size[1],layer.stride[0])
            else:
                forward="groupconv{}x{}s{}_forward".format(layer.kernel_size[0],layer.kernel_size[1],layer.stride[0])
            if i+1<len(layers) and isinstance(layers[i+1],nn.BatchNorm2d):
                conv = nn.utils.fusion.fuse_conv_bn_eval(layers[i], layers[i + 1])
                i+=1
            else:
                conv=layer
            convw=conv.weight.detach().numpy().flatten()
            weight_address_offect=address_header
            address_header+=convw.shape[0]
            weight_size=convw.shape[0]
            weight_bias_f_space+=convw.shape[0]
            scale_weights,convw_short=float2short(convw)

            parameters_float=np.concatenate((parameters_float,convw),axis=0)
            parameters_short=np.concatenate((parameters_short,convw_short),axis=0)
            if conv.bias is not None:
                convb=conv.bias.detach().numpy().flatten()
                bias_addres_offect=address_header
                address_header+=convb.shape[0]
                bias_size=convb.shape[0]
                weight_bias_f_space+=convb.shape[0]
                scale_bias,convb_short=float2short(convb)
                parameters_float=np.concatenate((parameters_float,convb),axis=0)
                parameters_short=np.concatenate((parameters_short,convb_short),axis=0)
                bias_param_offect="{}+{}".format(param_name,bias_addres_offect)
            else:
                bias_addres_offect=0
                bias_size=0
                bias_param_offect="NULL"
            kargs={"conv_weight":"{}+{}".format(param_name,weight_address_offect),"conv_bias":"{}".format(bias_param_offect),
                    "conv_weight_data_size":weight_size,"conv_bias_data_size":bias_size,"conv_weight_short_scale":scale_weights,
                    "conv_bias_short_scale":scale_bias,"conv_group":layer.groups
            }
            Layers_layer=Layers(forward,**kargs)
            net_dict.update({name_layer:Layers_layer})
        
        elif isinstance(layers[i],nn.ConvTranspose2d):

            forward="deconv{}x{}s{}_forward".format(layer.kernel_size[0],layer.kernel_size[1],layer.stride[0])
            if i+1<len(layers) and isinstance(layers[i+1],nn.BatchNorm2d):
                fused_deconv = copy.deepcopy(layers[i])
                fused_deconv.weight = torch.nn.Parameter(torch.transpose(layers[i].weight, 0, 1))
                conv = nn.utils.fusion.fuse_conv_bn_eval(fused_deconv, layers[i + 1])
                i+=1
            else:
                conv=layer
            
            convw=conv.weight.detach().numpy().flatten()
            weight_address_offect=address_header
            address_header+=convw.shape[0]
            weight_size=convw.shape[0]
            weight_bias_f_space+=convw.shape[0]
            scale_weights,convw_short=float2short(convw)
           
            parameters_float=np.concatenate((parameters_float,convw),axis=0)
            parameters_short=np.concatenate((parameters_short,convw_short),axis=0)
            if conv.bias is not None:
                convb=conv.bias.detach().numpy().flatten()
                bias_addres_offect=address_header
                address_header+=convb.shape[0]
                bias_size=convb.shape[0]
                weight_bias_f_space+=convb.shape[0]
                scale_bias,convb_short=float2short(convb)
                parameters_float=np.concatenate((parameters_float,convb),axis=0)
                parameters_short=np.concatenate((parameters_short,convb_short),axis=0)
                bias_param_offect="{}+{}".format(param_name,bias_addres_offect)
            else:
                bias_addres_offect=0
                bias_size=0
                bias_param_offect="NULL"
            kargs={"conv_weight":"{}+{}".format(param_name,weight_address_offect),"conv_bias":"{}".format(bias_param_offect),
                    "conv_weight_data_size":weight_size,"conv_bias_data_size":bias_size,"conv_weight_short_scale":scale_weights,
                    "conv_bias_short_scale":scale_bias,"conv_group":layer.groups
            }
            Layers_layer=Layers(forward,**kargs)
            net_dict.update({name_layer:Layers_layer})

            forward="deconvcrop_forward"
            Layers_layer=Layers(forward)#TODO deconvcrop参数
            net_dict.update({"{}/Deconvcrop".format(name_layer):Layers_layer})
       
        elif isinstance(layer,nn.Sigmoid) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.LeakyReLU) or isinstance(layer,nn.Hardswish) or isinstance(layer,nn.Hardsigmoid)  or isinstance(layer,nn.Tanh):
            
            
            forward_inplace="{}_forward_inplace".format(str(type(layer)).lower().replace(">","").replace("<","").replace("'","").split('.')[-1])
                           
            if isinstance(layer,nn.LeakyReLU):
                kargs={"leakyrelu_slope":"{}".format(layer.negative_slope)}
            else:
               kargs={}
            Layers_layer=Layers(forward_inplace=forward_inplace,support_inplace=True,**kargs)
            net_dict.update({name_layer: Layers_layer})

        elif isinstance(layer,nn.AdaptiveMaxPool2d) or isinstance(layer,nn.AdaptiveAvgPool2d) or isinstance(layer,nn.MaxPool2d) or isinstance(layer,nn.AvgPool2d):
            if isinstance(layer,nn.AdaptiveMaxPool2d):
                forward="pooling_global_forward"
                kargs={"pooling_type":0}
            elif isinstance(layer,nn.AdaptiveAvgPool2d):
                forward="pooling_global_forward"
                kargs={"pooling_type":1}
            elif isinstance(layer,nn.MaxPool2d):
                forward="pooling2x2s2_forward"
                kargs={"pooling_type":0}
            else:
                forward="pooling2x2s2_forward"
                kargs={"pooling_type":1}
            Layers_layer=Layers(forward,**kargs)
            net_dict.update({name_layer:Layers_layer})
        elif isinstance(layer,nn.Upsample):
            forward="upsample_forward"
            if layer.scale_factor is not None:
                if isinstance(layer.scale_factor,tuple):
                    scale_factor_h=layer.scale_factor(0)
                    scale_factor_w=layer.scale_factor(1)
                else:
                    scale_factor_h=layer.scale_factor
                    scale_factor_w=layer.scale_factor
            else:
                scale_factor_h=0
                scale_factor_w=0
            #print(layer)
            if layer.size is not None:
                size_h=layer.size[0]
                size_w=layer.size[1]
            else:
                size_h=0
                size_w=0
            kargs={"upsample_scale_factor_h":scale_factor_h,"upsample_scale_factor_w":scale_factor_w,"upsample_size_h":size_h,
                    "upsample_size_w":size_w,"upsample_mode":upsample_dict[layer.mode],"upsample_align_corners":layer.align_corners}
            Layers_layer=Layers(forward,**kargs)
            net_dict.update({name_layer:Layers_layer})
        Max_param=max(Max_param,weight_bias_f_space)
    return net_dict,parameters_float,parameters_short,Max_param

def float2short(param):
    scale=int(32760/np.maximum(np.max(param),-np.min(param))-10)
    param_out=np.int32(scale*param)
    return scale,param_out

def param_save(file,param,to_short,num_row):
    """参数保存的中间排版函数"""
    open(file,"w").close()
    size_param=param.shape[0]
    k0=size_param % num_row
    k1=int(size_param / num_row)
    param0=param[0:k1*num_row]
    param1=param[k1*num_row:size_param]
    with open(file,"ab") as f:
        if to_short:
            np.savetxt(f, param0.reshape(-1, 8), fmt='%d', delimiter=', ', newline=',\n')
            np.savetxt(f, param1.reshape(-1, k0), fmt='%d', delimiter=', ', newline=',\n')
        else:
            np.savetxt(f, param0.reshape(-1, 8), fmt='%1.10f', delimiter=', ', newline=',\n')
            np.savetxt(f, param1.reshape(-1, k0), fmt='%1.10f', delimiter=', ', newline=',\n')
        f.close()
    return file


class Blob_address(object):
    def __init__(self,address_start=0,address_end=0):
            self.start=address_start
            self.end=address_end
def multiple_input_sorted(input,input_index):
    input=input[0]
    if len(list(input))==0:
        return []
    else:
        input_index_sortindex=input_index.argsort()
        #print(input_index_sortindex,input)
        input=input[input_index_sortindex]
        # print(input)
        # exit()
        input=list(input[0])

        return input
def blob_space_allocation(list_of_layer,list_of_blob,dict_of_node,dict_of_layer,dict_of_blob,net_dict,blob_matrix_layer,layer_matrix_blob):
    blob_matrix_blob=[]
    blob_refcount=[]
    for name in list_of_blob:
        index_blob=list_of_blob.index(name)
        blob_bottoms=np.where(layer_matrix_blob[:,index_blob]!=0)
        blbo_bottoms_order=layer_matrix_blob[np.where(layer_matrix_blob[:,index_blob]!=0),index_blob]
        blob_bottoms=multiple_input_sorted(blob_bottoms,blbo_bottoms_order)
        if name in dict_of_node.keys():
            node=dict_of_node[name]
        if len(blob_bottoms)==0 and "IO Node"==node.op:
            if "input" in name:
                blob_bottoms=[0]
            else:
                print(name,"blob not have input error")
        for i in blob_bottoms:
            blob_to_blob=np.where(blob_matrix_layer[:,i]!=0)
            blob_to_blob_order=blob_matrix_layer[np.where(blob_matrix_layer[:,i]!=0),i]
            blob_to_blob=multiple_input_sorted(blob_to_blob,blob_to_blob_order)
            # print(blob_to_blob)
            # exit()
            blob_refcount.extend(blob_to_blob)
    node_layer_max=[]
    list_inplace=[]
    list_inplace_replace=[]
    for name in list_of_blob:
        index_blob=list_of_blob.index(name)
        blob_bottoms=np.where(layer_matrix_blob[:,index_blob]!=0)
        blbo_bottoms_order=layer_matrix_blob[np.where(layer_matrix_blob[:,index_blob]!=0),index_blob]
        blob_bottoms=multiple_input_sorted(blob_bottoms,blbo_bottoms_order)
        if name in dict_of_node.keys():
            node=dict_of_node[name]
        if len(blob_bottoms)==0 and "IO Node"==node.op:
            if "input" in name:
                blob_bottoms=[0]
            else:
                print(name,"blob not have input error")

        for i in blob_bottoms:
            blob_to_blob=np.where(blob_matrix_layer[:,i]!=0)
            blob_to_blob_order=blob_matrix_layer[np.where(blob_matrix_layer[:,i]!=0),i]
            blob_to_blob=multiple_input_sorted(blob_to_blob,blob_to_blob_order)
           
        if not "output/output" in name:
            blob_param=dict_of_blob[name]
            blob_c=int(blob_param[1])
            blob_h=int(blob_param[2])
            blob_w=int(blob_param[3])
            
            # print(blob_refcount)
            # exit()
            pppp=sorted(blob_refcount)
            pppp=list(set(pppp))
            tttttt=[x for x in pppp if x<list_of_blob.index(name)+1]
            for i in range(len( blob_to_blob)):
                blob_refcount.pop(0)

            if  blob_to_blob[0] not in blob_refcount and blob_param[4]:
                list_inplace.append(index_blob)
                list_inplace_replace.append(blob_to_blob[0])
                if index_blob in tttttt:
                    tttttt.remove(index_blob)
            node_layer_max.append(tttttt)
    # print(node_layer_max)
    # exit()
    for i in range(len(list_inplace_replace)):
        while(list_inplace_replace[i] in list_inplace):
            list_inplace_replace[i]=list_inplace_replace[list_inplace.index(list_inplace_replace[i])]

   
    for i in node_layer_max:
        for ii in i:
            if ii in list_inplace:
                i[i.index(ii)]=list_inplace_replace[list_inplace.index(ii)]
  
    for i in range(len(node_layer_max)-1,1,-1):
        if sorted(node_layer_max[i])==sorted(node_layer_max[i-1]):
            node_layer_max.pop(i)
    space_all_time=np.zeros(len(node_layer_max))
    ccccou=0
    for i in node_layer_max:
        for ii in i:
            blob_param=dict_of_blob[list_of_blob[ii]]
            blob_c=int(blob_param[1])
            blob_h=int(blob_param[2])
            blob_w=int(blob_param[3])
            space_all_time[ccccou]+=blob_c*alignptr(int(blob_w*blob_h),16)
        ccccou+=1
    anchor_point=np.argmax(space_all_time)
    space_max=np.max(space_all_time)
    flag_space=1
    while(flag_space):
       
        address_dict={}
        ptemp=0
        for i in node_layer_max[-1]:
            blob_param=dict_of_blob[list_of_blob[i]]
            blob_c=int(blob_param[1])
            blob_h=int(blob_param[2])
            blob_w=int(blob_param[3])
            temp=blob_c*alignptr(int(blob_w*blob_h),16)
            blob_address=Blob_address(ptemp,ptemp+temp)
            address_dict.update({i:blob_address})
            ptemp+=temp
        
        for i in range(len(node_layer_max)-2,0,-1):
            flag_need=0
            space_temp=np.zeros(int(space_max),dtype=np.uint8)
            blob_now=node_layer_max[i]
            blob_pre=node_layer_max[i-1]
            blob_after=node_layer_max[i+1]
            blob_intersection=list(set(blob_now).intersection(set(blob_pre)).intersection(set(blob_after)))

            for p in blob_intersection:
                if p in address_dict.keys():
                    blob_address=address_dict[p]
                    space_temp[blob_address.start:blob_address.end]=2
            for p in blob_now:
                if p not in blob_intersection and p in address_dict.keys():
                    blob_address=address_dict[p]
                    space_temp[blob_address.start:blob_address.end]=1
            for p in blob_now: 
                blob_param=dict_of_blob[list_of_blob[p]]
                blob_c=int(blob_param[1])
                blob_h=int(blob_param[2])
                blob_w=int(blob_param[3])
                temp=blob_c*alignptr(int(blob_w*blob_h),16)
                space_now=space_temp[space_temp!=2]
                if p not in address_dict.keys():
                    space_empty0=np.sum(space_temp[int(space_max)-np.argmin(space_temp[::-1])-temp:int(space_max)-np.argmin(space_temp[::-1])])
                    space_empty1=np.sum(space_temp[np.argmin(space_temp):np.argmin(space_temp)+temp])
                    if space_now[-1]==0 and space_empty0==0:
                        blob_address=Blob_address(int(space_max)-np.argmin(space_temp[::-1])-temp,int(space_max)-np.argmin(space_temp[::-1]))
                        address_dict.update({p:blob_address})
                        space_temp[blob_address.start:blob_address.end]=1
                    elif space_empty1==0:
                        blob_address=Blob_address(np.argmin(space_temp),np.argmin(space_temp)+temp)
                        address_dict.update({p:blob_address})
                        space_temp[blob_address.start:blob_address.end]=1
                    elif space_empty0==0:
                        blob_address=Blob_address(int(space_max)-np.argmin(space_temp[::-1])-temp,int(space_max)-np.argmin(space_temp[::-1]))
                        address_dict.update({p:blob_address})
                        space_temp[blob_address.start:blob_address.end]=1
                    else:
                        space_alreadyneed=np.min((space_empty0,space_empty1),0)
                        flag_need=1
                        break
            if flag_need:
                space_max+=space_alreadyneed
                break
            else:
                flag_space=0
    return address_dict,space_max


def toC(list_of_layer,list_of_blob,dict_of_node,dict_of_layer,dict_of_blob,net_dict,blob_matrix_layer,layer_matrix_blob,address_dict):
    """将字典中保存数据读出转换成对应C代码格式"""
    net_ex_init=""
    for name in list_of_layer:
       
        index_layer=list_of_layer.index(name)
        layer_bottoms=np.where(blob_matrix_layer[:,index_layer]!=0)
        layers_bottoms_order=blob_matrix_layer[np.where(blob_matrix_layer[:,index_layer]!=0),index_layer]
        layer_bottoms=multiple_input_sorted(layer_bottoms,layers_bottoms_order)

        layer_tops=np.where(layer_matrix_blob[index_layer,:]!=0)
        layer_tops_order=layer_matrix_blob[index_layer,np.where(layer_matrix_blob[index_layer,:]!=0)]
        layer_tops=multiple_input_sorted(layer_tops,layer_tops_order)
        if name in dict_of_node.keys():
            node=dict_of_node[name]
           
            if "aten::mul" == node.op or "aten::add" == node.op or "aten::sub" == node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],bool,NULL,{}_forward_inplace,0,1,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )  

            elif node.op=="aten::_convolution":
                name_split=name.rsplit("/",1)[0]
                dict_i=net_dict[name_split]
                net_ex_init+="  DEFINE_LAYER(layers[{}],Conv,{},NULL,1,0,{},list{},{},list{},{},list({},{},{},{},{},{},NULL,NULL,{}));\n".format(
                    index_layer,dict_i.forward,index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),
                    len(layer_tops),dict_i.Conv.weight,dict_i.Conv.bias,dict_i.Conv.weight_data_size,dict_i.Conv.bias_data_size,dict_i.Conv.weight_short_scale,
                    dict_i.Conv.bias_short_scale,dict_i.Conv.group
                    )
            elif "aten::cat"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Concat,{}_forward,NULL,0,0,{},list{},{},list{},{},list({}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0])
                    )
            elif "aten::upsample_bilinear2d"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],bool,{}_forward,NULL,1,0,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::leaky_relu"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],LeakyRelu,NULL,{}_forward_inplace,1,1,{},list{},{},list{},{},list({}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),dict_of_layer[name][0]
                    )
            elif "aten::leaky_relu_"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],LeakyRelu,NULL,{}forward_inplace,1,1,{},list{},{},list{},{},list({}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),dict_of_layer[name][0]
                    )
            elif "aten::relu" == node.op or "aten::tanh" == node.op or "aten::hardswish" == node.op or "aten::hardsigmoid" == node.op or"aten::sigmoid" == node.op:
                 net_ex_init+="  DEFINE_LAYER(layers[{}],bool,NULL,{}_forward_inplace,1,1,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::relu_" == node.op or "aten::tanh_" == node.op or "aten::hardswish_" == node.op or "aten::hardsigmoid_" == node.op or"aten::sigmoid_" == node.op:
                 net_ex_init+="  DEFINE_LAYER(layers[{}],bool,NULL,{}forward_inplace,1,1,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::adaptive_avg_pool2d"==node.op or "aten::adaptive_max_pool2d"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],bool,{}_forward,NULL,1,0,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::max_pool2d"==node.op or "aten::avg_pool2d"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Pool,{}_forward,NULL,1,0,{},list{},{},list{},{},list({},{},{},{}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),
                        int(dict_of_layer[name][0]),int(dict_of_layer[name][1]),int(dict_of_layer[name][2]),int(dict_of_layer[name][3]),
                    )
            elif "aten::replication_pad2d"==node.op:
                
                net_ex_init+="  DEFINE_LAYER(layers[{}],Padding,padding_forward,NULL,1,0,{},list{},{},list{},{},list({},{},1,0));\n".format(
                        index_layer,index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0]),int(dict_of_layer[name][2]),
                    )
            elif "aten::reflection_pad2d"==node.op:
              
                net_ex_init+="  DEFINE_LAYER(layers[{}],Padding,padding_forward,NULL,1,0,{},list{},{},list{},{},list({},{},2,0));\n".format(
                        index_layer,index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0]),int(dict_of_layer[name][2]),
                    )
            elif "aten::split"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Split,{}_forward,NULL,0,0,{},list{},{},list{},{},list({},{}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0]),int(dict_of_layer[name][1])
                    )
            elif "aten::softmax"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],bool,NULL,{}_forward_inplace,1,1,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::view"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],bool,NULL,{}_forward,1,0,{},list{},{},list{},{},list(0));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops)
                    )
            elif "aten::permute"==node.op:
                dim=np.zeros(3)
                dim[0]=0
                dim[1]=1
                dim[2]=2
                for i in range(1,len(dict_of_layer[name])):
                    dim[i-1]=dict_of_layer[name][i]
                net_ex_init+="  DEFINE_LAYER(layers[{}],Permute,NULL,{}_forward,1,0,{},list{},{},list{},{},list({},{},{}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),
                        int(dim[0]),int(dim[1]),int(dim[2])
                    )
            elif "aten::transpose"==node.op:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Transpose,NULL,{}_forward,1,0,{},list{},{},list{},{},list({},{}));\n".format(
                        index_layer,node.op.replace("aten::",""),index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),
                        int(dict_of_layer[name][0]), int(dict_of_layer[name][1])
                    )
        else:
            if "Padding" in name:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Padding,padding_forward,NULL,1,0,{},list{},{},list{},{},list({},{},0,0));\n".format(
                        index_layer,index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0]),int(dict_of_layer[name][1]),
                    )
            elif "Deconvcrop" in name:
                net_ex_init+="  DEFINE_LAYER(layers[{}],Deconvcrop,deconvcrop_forward,NULL,1,0,{},list{},{},list{},{},list({},{},{},{}));\n".format(
                        index_layer,index_layer,tuple(layer_bottoms),len(layer_bottoms),tuple(layer_tops),len(layer_tops),int(dict_of_layer[name][0]),
                        int(dict_of_layer[name][1]), int(dict_of_layer[name][2]), int(dict_of_layer[name][3])
                    )

    blob_input=[]
    for name in list_of_blob:
       
        blob_matrix_blob=[]
        index_blob=list_of_blob.index(name)
        blob_bottoms=np.where(layer_matrix_blob[:,index_blob]!=0)
        blbo_bottoms_order=layer_matrix_blob[np.where(layer_matrix_blob[:,index_blob]!=0),index_blob]
        blob_bottoms=multiple_input_sorted(blob_bottoms,blbo_bottoms_order)
        blob_tops=np.where(blob_matrix_layer[index_blob,:]!=0)
        blob_tops_order=blob_matrix_layer[index_blob,np.where(blob_matrix_layer[index_blob,:]!=0)]
        blob_tops=multiple_input_sorted(blob_tops,blob_tops_order)
        if name in dict_of_node.keys():
            node=dict_of_node[name]
        if len(blob_bottoms)==0 and "IO Node"==node.op:
            if "input" in name:
                blob_bottoms=[-1]
                blob_input.append(index_blob)
            else:
                print(name,"blob not have input error")
       
        for i in blob_bottoms:
            blob_to_blob=np.where(blob_matrix_layer[:,i]!=0)
            blob_to_blob_order=blob_matrix_layer[np.where(blob_matrix_layer[:,i]!=0),i]
            blob_to_blob=multiple_input_sorted(blob_to_blob,blob_to_blob_order)
            blob_matrix_blob.extend(blob_to_blob)
        if not "output/output" in name:
            blob_param=dict_of_blob[name]
            blob_c=int(blob_param[1])
            blob_h=int(blob_param[2])
            blob_w=int(blob_param[3])
            if len(blob_tops)==0:
                blob_tops=[-1] 
            if index_blob in  address_dict.keys():
               
                net_ex_init+="  DEFINE_BLOB(&blobs[{}],{},list{},{},list(NULL,{},{},{},{},0,4,0,0));\n".format(index_blob,blob_bottoms[0],tuple(blob_tops),
                                    address_dict[index_blob].start,blob_c,blob_h,blob_w,alignptr(int(blob_w*blob_h),16),
                                    )
            else:
             
                net_ex_init+="  DEFINE_BLOB(&blobs[{}],{},list{},-1,list(NULL,{},{},{},{},0,4,0,0));\n".format(index_blob,blob_bottoms[0],tuple(blob_tops),
                                    blob_c,blob_h,blob_w,alignptr(int(blob_w*blob_h),16),
                                    )
        elif "output/output" in name:
           
            for i in blob_bottoms:
                blob_to_blob=np.where(blob_matrix_layer[:,i]!=0)
                blob_to_blob_order=blob_matrix_layer[np.where(blob_matrix_layer[:,i]!=0),i]
                blob_to_blob=multiple_input_sorted(blob_to_blob,blob_to_blob_order)
    
            net_ex_init+="  DEFINE_BLOB(&blobs[{}],{},list{},-1,list(NULL,{},{},{},{},0,4,0,0));\n".format(index_blob,blob_bottoms[0],tuple(blob_to_blob),
                                    0,0,0,0,
                                    )
    return  net_ex_init,blob_input   

def alignptr(ptr,n):
    return ((ptr+n-1)&-n)


def network_visualization(list_of_layer,list_of_blob,blob_matrix_layer,layer_matrix_blob):
    G=nx.DiGraph()
    G.add_nodes_from(list_of_layer)
    colors=[]
    list_blob=[]
    for i in range(len(list_of_layer)):
        colors.append('red')
    for i in range(len(list_of_blob)):
        list_blob.append(i)
        colors.append('green')
    G.add_nodes_from(list_blob)
    x,y=np.where(blob_matrix_layer!=0)
    value=blob_matrix_layer[x,y]
    for i in range(len(x)):
        G.add_edge(x[i],list_of_layer[y[i]],weight=value[i])
    pos=nx.spring_layout(G)
    #nx.draw(G,with_labels=True,node_color=colors,node_size=60,font_size=8)
    #nx.draw_networkx()
    plt.figure(1,figsize=(200,80))
    nx.draw(G,with_labels=True,node_color=colors,node_size=30,font_size=4)
    nx.draw_networkx_edge_labels(G,pos)
    plt.show()
    print('end')
    exit()




def Cfile(net_init,Param_float,Param_short,framework,toshort,list_of_layer,list_of_blob,space_max,blob_input,Max_param):
  
    net_head_file="#include\"net.h\"\n"
    net_head_file+="#include\"net_api.h\"\n".format((framework))
    net_head_file+="#include\"layer.h\"\n"
    if toshort:
        net_head_file+="#include\"parameters_short_{}.h\"\n".format(framework)
    else:
        net_head_file+="#include\"parameters_float_{}.h\"\n".format(framework)
    net_ex_init= net_head_file
    net_ex_init+="Extractor* net_ex_init(){\n"
    net_ex_init+="  int layer_count = {};\n".format(len(list_of_layer))
    net_ex_init+="  Layer** layers = (Layer**)malloc(layer_count * sizeof(Layer*));\n"
    net_ex_init+="  for (int i = 0;i < layer_count; i++)\n"
    net_ex_init+="      layers[i] = (Layer*)malloc(sizeof(Layer));\n"
    net_ex_init+="  int blob_count = {};\n".format(len(list_of_blob))
    net_ex_init+="  Blob* blobs = (Blob*)malloc(blob_count * sizeof(Blob));\n"
    net_ex_init+=net_init
 
    net_ex_init+="  Option opt = { 1,0,0,0,NULL };\n"
    
    net_ex_init+="  Net _net = { forward_layer, opt, layers, blobs, layer_count , blob_count};\n"
    net_ex_init+="  Net* net = (Net*)malloc(sizeof(Net));\n"
    net_ex_init+="  *net = _net;\n"
    net_ex_init+="  Extractor* ex = (Extractor*)malloc(sizeof(Extractor));\n"
    net_ex_init+="  extractor(ex, net, net->blob_count);\n"
    net_ex_init+="  return ex;\n"
    net_ex_init+="}\n"
    net_ex_init+="int detect_{}(const Mat **bgr, Mat **out,float* data,Extractor* ex)\n".format(framework)
    net_ex_init+="{\n"
    net_ex_init+="#if NCNN_VULKAN\n"
    net_ex_init+="  mobilenet.opt.use_vulkan_compute = true;\n"
    net_ex_init+="#endif // NCNN_VULKAN\n"
    #net_ex_init+="  Extractor* ex = net_ex_init();\n"
    net_ex_init+="  int blob_number={};\n".format(len(list_of_blob)-1)
    net_ex_init+="""for (int i = 0; i < blob_number; i++)\n
    {\n
        if (ex->net->blobs[i].mat_addressoffect != -1)\n
            ex->blob_mats[i].data = data + ex->net->blobs[i].mat_addressoffect;\n
        else
            ex->blob_mats[i].data = -1;\n
    }\n"""

    net_ex_init+="  int input_no[{}]={{{}}};\n".format(len(blob_input),str(blob_input).replace('[','').replace(']',''))
    net_ex_init+="  int input_num=sizeof(input_no)/sizeof(int);\n"
    net_ex_init+="  for(int i=0;i<input_num;i++)\n"
    net_ex_init+="      ex->input(ex, i, *bgr[i]);\n"
    net_ex_init+="  for (int i = 1; i <blob_number; i++)\n{\n"
    net_ex_init+="    if (ex->net->blobs[i].producer==-1)\n"
    net_ex_init+="        continue;\n"
    net_ex_init+="    ex->extract(ex, i);\n}\n"
    net_ex_init+="  for (int i = 0; i < ex->net->blobs[blob_number].consumer_num; i++)\n{\n"
    net_ex_init+="  memcpy((*out[i]).data, (float*)(ex->blob_mats[ex->net->blobs[blob_number].consumers[i]].data), total(*out[i])*(*out[i]).elemsize);\n}\n"
    net_ex_init+="  return 0;\n"
    net_ex_init+="}\n"
    net_ex_init+="void main()//名称及参数需要改成与原先接口一致\n{\n"
    net_ex_init+="     float* data = (float*)malloc({} * sizeof(float));\n".format(int(space_max))
    net_ex_init+="Extractor* ex=net_ex_init();\n"
    net_ex_init+="Mat **input= (Mat**)malloc(() * sizeof(Mat*));//TODO  需要填写输入数目\n"
    net_ex_init+="Mat **output= (Mat**)malloc(() * sizeof(Mat*));/TODO  需要填写输出数目\n"

    if toshort:
        net_ex_init+="""     float* mem_param=(float*)malloc({} * sizeof(float));
     ex->opt.mem=mem_param;
     ex->opt.use_short_comession=True;
     ex->opt.short_2_float_once={};\n""".format(Max_param,toshort==2)
    net_ex_init+="//TODO  从图像到MAT类型的预处理部分\n"

    net_ex_init+="ret=detect_{}(input, output,data,ex);\n".format(framework)
    net_ex_init+="//TODO 从MAT转到输出的后面处理部分 \n"
    if toshort:
        net_ex_init+="     free(mem_param);\n"
    net_ex_init+="     free(data);\n"
    net_ex_init+="}\n"
    param_float="#if 1\n"
    param_float+="static float param_{}[{}]=".format(framework,Param_float.shape[0])
    param_float+="{\n"  
    file=param_save("param_float.txt",Param_float,0,8)
    with open(file, 'r') as f:
        param_float += f.read()
    param_float+= '};\n\n'
    param_float += '#endif\n'

    param_short="#if 1\n"
    param_short+="static short param_{}[{}]=".format(framework,Param_short.shape[0])
    param_short+="{\n"  
    file=param_save("param_short.txt",Param_short,1,8)
    with open(file, 'r') as f:
        param_short += f.read()
    param_short+= '};\n\n'
    param_short += '#endif\n'

    net_h="#include\"net.h\"\n"
    net_h+="int detect_{}(const Mat bgr, Mat *out);\n".format(framework)

    return net_ex_init,param_float,param_short,net_h
def get_main(net,input,to_short,save_path,framework):
    net_dict=dict()
    Param_name="param_{}".format(framework)
    name_node=net.__class__.__name__
    layers,name_layers=get_parameters_layer(net,name_node)
    list_of_layer, dict_of_layer,list_of_blob,dict_of_blob,blob_matrix_layer,layer_matrix_blob,dict_of_node=get_trace_graph(net,input)
    #network_visualization(list_of_layer,list_of_blob,blob_matrix_layer,layer_matrix_blob)

    net_dict,parameters_float,parameters_short,Max_param=get_layer_param(layers,name_layers,Param_name)
    if to_short==2:
        Max_param=parameters_short.shape[0]
    address_dict,space_max=blob_space_allocation(list_of_layer,list_of_blob,dict_of_node,dict_of_layer,dict_of_blob,net_dict,blob_matrix_layer,layer_matrix_blob)
    net_init,blob_input=toC(list_of_layer,list_of_blob,dict_of_node,dict_of_layer,dict_of_blob,net_dict,blob_matrix_layer,layer_matrix_blob,address_dict)
    net_ex_init,param_float,param_short,net_h=Cfile(net_init,parameters_float,parameters_short,framework,to_short,list_of_layer,list_of_blob,space_max,blob_input,Max_param)
    if to_short:
        txt_p = open(os.path.join(save_path,"parameters_short_{}.h".format(framework)), "w")
        txt_p.write(param_short)
        txt_p.close()
    else:
        txt_p = open(os.path.join(save_path,"parameters_float_{}.h".format(framework)), "w")
        txt_p.write(param_float)
        txt_p.close()

    # txt_p = open(os.path.join(save_path,"net_{}.h".format(framework)), "w")
    # txt_p.write(net_h)
    # txt_p.close()

    txt_p = open(os.path.join(save_path,"net_{}.c".format(framework)), "w")
    txt_p.write(net_ex_init)
    txt_p.close()
    print("end")







