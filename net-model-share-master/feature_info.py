# -*- coding: utf-8 -*-
import torch
import numpy as np 
from name_539 import feature_info_key
from similarity_6193.MobileNet import MNV3_bufen_new5 as simi_net

class feature_info(object):
    def __init__(self):
        self.feature = np.zeros(540)
        checkpoint_simi = torch.load('./similarity_6193/14_270_net_G.pth', 'cpu')
        model = simi_net(2, 1, 4, n_blocks=1)
        model.load_state_dict(checkpoint_simi, strict=True)  #
        self.similarity_model = model.cuda().eval()

    def set_feature_info(self, name:str, value):
        index = feature_info_key[name]
        self.feature[index] = value

    def set_feature_info_array(self, index:np.array, value):
        self.feature[index] = value
    
    def set_feature_info_dic(self, feat_dic:dict):
        for key in feat_dic.keys():
            index = feature_info_key[str(key)]
            self.feature[index] = feat_dic[key]
    

if __name__ == '__main__':
    fi = feature_info()
    fi.set_feature_info('nInDenO', 5)
    index = np.arange(5)
    value = np.ones(5)
    fi.set_feature_info_array(index, value)
    print(fi.feature)