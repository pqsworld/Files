import torch

from models.PTQ import bit_type

from .base import BaseObserver
import math
import numpy as np
import copy
from scipy import stats
import matplotlib.pyplot as plt
import os

class KLDivergenceObserver(BaseObserver):
    def __init__(self, module_type, bit_type, calibration_mode, bins=4095, kldiv_sigma=0.01):
        super(KLDivergenceObserver, self).__init__(
            module_type, bit_type, calibration_mode)
        self.symmetric = self.bit_type.signed
        self.bins = bins
        self.kldiv_sigma = kldiv_sigma

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        if self.calibration_mode == "layer_wise":
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()    
    
    def update_histc(self, v):
        assert self.symmetric and self.calibration_mode == 'layer_wise'        
        v = self.reshape_tensor(v)
        # v = torch.abs(v)
        absmax_val = torch.max(-self.min_val, self.max_val)
        if self.histc is None:
            self.histc = torch.histc(v, bins=self.bins, min=-absmax_val, max=absmax_val)
            # self.histc = torch.histc(v, bins=self.bins, min=0, max=absmax_val)
        else:
            self.histc += torch.histc(v, bins=self.bins, min=-absmax_val, max=absmax_val)
            # self.histc += torch.histc(v, bins=self.bins, min=0, max=absmax_val)
    
    # Origin Distribution
    def get_quantization_params(self, *args, **kwargs):
        def own_kl(p, q):
            pk = 1.0 * p / np.sum(p)
            qk = 1.0 * q / np.sum(q)
            t = 0
            for i in range(pk.shape[0]):
                t += pk[i] * np.log(pk[i]) - pk[i] * np.log(qk[i])
            
            return t

        def threshold_distribution(distribution, target_bin=255):
            """
            Return the best threshold value. 
            Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
            Args:
                distribution: list, activations has been processed by histogram and normalize,size is 2048
                target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
            Returns:
                target_threshold: int, num of bin with the minimum KL 
            """   
            # normalize
            distribution = distribution / sum(distribution)
            # distribution = distribution[1:]
            length = distribution.size
            threshold_sum1 = sum(distribution[:(length - target_bin) // 2])
            threshold_sum2 = sum(distribution[(length + target_bin) // 2:])
            kl_divergence = np.zeros((length - target_bin) // 2)
            # print(kl_divergence.shape, length)
            for threshold in range(target_bin, length, 2):
                sliced_nd_hist = copy.deepcopy(distribution[(length - threshold) // 2:(length + threshold) // 2])           

                # generate reference distribution p
                p = sliced_nd_hist.copy()
                if p[0] == 0 and p[threshold - 1] == 0:
                    kl_divergence[(threshold - target_bin) // 2] = 1000
                    threshold_sum1 = threshold_sum1 - distribution[(length - threshold) // 2 - 1]
                    threshold_sum2 = threshold_sum2 - distribution[(length + threshold) // 2]
                    continue
                p[0] += threshold_sum1
                p[threshold-1] += threshold_sum2
                threshold_sum1 = threshold_sum1 - distribution[(length - threshold) // 2 - 1]
                threshold_sum2 = threshold_sum2 - distribution[(length + threshold) // 2]
                # is_nonzeros[k] indicates whether hist[k] is nonzero
                is_nonzeros = (p != 0).astype(np.int64)
                # 
                quantized_bins = np.zeros(target_bin, dtype=np.float64)
                # calculate how many bins should be merged to generate quantized distribution q
                num_merged_bins = sliced_nd_hist.size // target_bin
                
                # merge hist into num_quantized_bins bins
                for j in range(target_bin):
                    start = j * num_merged_bins
                    stop = start + num_merged_bins
                    quantized_bins[j] = sliced_nd_hist[start:stop].sum()
                quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
                
                # expand quantized_bins into p.size bins
                q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
                for j in range(target_bin):
                    start = j * num_merged_bins
                    if j == target_bin - 1:
                        stop = -1
                    else:
                        stop = start + num_merged_bins
                    norm = is_nonzeros[start:stop].sum()
                    if norm != 0:
                        q[start:stop] = float(quantized_bins[j]) / float(norm)
                q[p == 0] = 0
                # p = _smooth_distribution(p) # with some bugs, need to fix
                # q = _smooth_distribution(q)
                # p[p == 0] = 0.0001
                # q[q == 0] = 0.0001
                p[p == 0] = 1e-12
                q[q == 0] = 1e-12
                
                # calculate kl_divergence between q and p
                t = stats.entropy(p, q)
                # print((threshold - target_bin) // 2)
                kl_divergence[(threshold - target_bin) // 2]  = t
                # ot = own_kl(p, q)

            min_kl_divergence = np.argmin(kl_divergence)
            # print(np.min(kl_divergence))
            threshold_value = min_kl_divergence + target_bin // 2 + 1

            return threshold_value 
            
        # op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** (self.bit_type.bits - 1))
        # op_thr = (op_bin + 0.5) * torch.max(-self.min_val, self.max_val) / self.bins
        op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** self.bit_type.bits - 1)
        per_num_bin = 2 * torch.max(-self.min_val, self.max_val) / self.bins
        op_thr = (op_bin + 0.5) * per_num_bin
        
        # Plot Distribution & Quantizer Range
        normalize_histc = self.histc.cpu().numpy() / sum(self.histc.cpu().numpy())
        plt.plot((np.array(range(-self.bins // 2, self.bins // 2)) + 0.5) * per_num_bin.item(), normalize_histc, 'b')
        plt.plot(np.array([op_thr.item(), op_thr.item()]), np.array([0, np.max(normalize_histc)]), 'r')
        plt.plot(np.array([-op_thr.item(), -op_thr.item()]), np.array([0, np.max(normalize_histc)]), 'r')
        #plt.plot(-np.ones(np.max(self.histc.cpu().numpy() / 1000).astype('int64'))*op_thr.item(), range(np.max(self.histc.cpu().numpy()).astype('int64')) / sum(self.histc.cpu().numpy()), 'r')
        i = 0
        while i < 200:
            filename = '/hdd/file-input/linwc/match/datasets/supermatch/output/distribution/{0}.png'.format(i)
            if os.path.exists(filename):
                i += 1
            else:
                plt.savefig(filename)
                print(filename)
                plt.close()
                break
        
        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound
        scale = torch.ones_like(self.max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(self.max_val, dtype=torch.int64)
        # max_val = torch.max(-min_val, max_val)
        scale = op_thr / (float(qmax - qmin) / 2)
        scale.clamp_(self.eps)
        zero_point = torch.zeros_like(
            self.max_val, dtype=torch.int64)
        return scale, zero_point

    # # Absolute Distribution
    # def get_quantization_params(self, *args, **kwargs):
    #     def own_kl(p, q):
    #         pk = 1.0 * p / np.sum(p)
    #         qk = 1.0 * q / np.sum(q)
    #         t = 0
    #         for i in range(pk.shape[0]):
    #             t += pk[i] * np.log(pk[i]) - pk[i] * np.log(qk[i])
            
    #         return t

    #     def smooth_distribution(distribution, win_size = 3):
    #         radius = win_size // 2
    #         length = distribution.size
    #         out = np.zeros(length, dtype=np.float64)
    #         for j in range(radius):
    #             out[j], out[-j-1] = distribution[j], distribution[-j-1]
    #         for i in range(radius, length - radius):
    #             for r in range(-radius, radius + 1):
    #                 out[i] += distribution[i + r]
    #         out = out / (2 * radius + 1) 
    #         out = out / sum(out)
    #         return out
            
    #     def threshold_distribution(distribution, target_bin=128):
    #         """
    #         Return the best threshold value. 
    #         Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    #         Args:
    #             distribution: list, activations has been processed by histogram and normalize,size is 2048
    #             target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    #         Returns:
    #             target_threshold: int, num of bin with the minimum KL 
    #         """   
    #         # normalize
    #         distribution = distribution
    #         # distribution = distribution[1:]
    #         length = distribution.size
    #         threshold_sum = sum(distribution[target_bin:])
    #         kl_divergence = np.zeros(length - target_bin)
    #         # print(kl_divergence.shape, length)
    #         for threshold in range(target_bin, length):
    #             sliced_nd_hist = copy.deepcopy(distribution[:threshold])           
    #             # sliced_nd_hist = smooth_distribution(sliced_nd_hist)
    #             # generate reference distribution p
    #             p = sliced_nd_hist.copy()
    #             p[threshold-1] += threshold_sum
    #             threshold_sum = threshold_sum - distribution[threshold]
    #             # is_nonzeros[k] indicates whether hist[k] is nonzero
    #             is_nonzeros = (p != 0).astype(np.int64)
    #             quantized_bins = np.zeros(target_bin, dtype=np.int64)
    #             # calculate how many bins should be merged to generate quantized distribution q
    #             num_merged_bins = sliced_nd_hist.size // target_bin
                
    #             # merge hist into num_quantized_bins bins
    #             for j in range(target_bin):
    #                 start = j * num_merged_bins
    #                 stop = start + num_merged_bins
    #                 quantized_bins[j] = sliced_nd_hist[start:stop].sum()
    #             quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
                
    #             # expand quantized_bins into p.size bins
    #             q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
    #             for j in range(target_bin):
    #                 start = j * num_merged_bins
    #                 if j == target_bin - 1:
    #                     stop = -1
    #                 else:
    #                     stop = start + num_merged_bins
    #                 norm = is_nonzeros[start:stop].sum()
    #                 if norm != 0:
    #                     q[start:stop] = float(quantized_bins[j]) / float(norm)
    #             q[p == 0] = 0
    #             # p = _smooth_distribution(p) # with some bugs, need to fix
    #             # q = _smooth_distribution(q)
    #             # p[p == 0] = 0.0001
    #             # q[q == 0] = 0.0001
    #             p[p == 0] = 1e-12
    #             q[q == 0] = 1e-12
                
    #             # calculate kl_divergence between q and p
    #             t = stats.entropy(p, q)
    #             # print((threshold - target_bin) // 2)
    #             kl_divergence[threshold - target_bin] = t
    #             # ot = own_kl(p, q)

    #         min_kl_divergence = np.argmin(kl_divergence)
    #         threshold_value = min_kl_divergence + target_bin + 1

    #         return threshold_value 
    #     # op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** (self.bit_type.bits - 1))
    #     # op_thr = (op_bin + 0.5) * torch.max(-self.min_val, self.max_val) / self.bins
    #     op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** (self.bit_type.bits - 1))
    #     per_num_bin = torch.max(-self.min_val, self.max_val) / self.bins
    #     op_thr = op_bin * per_num_bin
    #     plt.plot((np.array(range(self.bins)) + 1) * per_num_bin.item(), self.histc.cpu().numpy() / sum(self.histc.cpu().numpy()), 'b')
    #     plt.plot(np.ones(np.max(self.histc.cpu().numpy()).astype('int32'))*op_thr.item(), range(np.max(self.histc.cpu().numpy()).astype('int32')) / sum(self.histc.cpu().numpy()), 'r')
    #     # plt.plot(-np.ones(np.max(self.histc.cpu().numpy()).astype('int32'))*op_thr.item(), range(np.max(self.histc.cpu().numpy()).astype('int32')) / sum(self.histc.cpu().numpy()), 'r')
    #     i = 0
    #     while i < 100:
    #         filename = '/hdd/file-input/linwc/match/datasets/supermatch/output/distribution/{0}.png'.format(i)
    #         if os.path.exists(filename):
    #             i += 1
    #         else:
    #             plt.savefig(filename)
    #             print(filename)
    #             plt.close()
    #             break
    #     qmax = self.bit_type.upper_bound
    #     qmin = self.bit_type.lower_bound
    #     scale = torch.ones_like(self.max_val, dtype=torch.float32)
    #     zero_point = torch.zeros_like(self.max_val, dtype=torch.int64)
    #     # max_val = torch.max(-min_val, max_val)
    #     scale = op_thr / (float(qmax - qmin) / 2)
    #     scale.clamp_(self.eps)
    #     zero_point = torch.zeros_like(
    #         self.max_val, dtype=torch.int64)
    #     return scale, zero_point

    # # Dynamic Update    
    # def update(self, v):
    #     # channel-wise needs too much time.
    #     assert self.symmetric and self.calibration_mode == 'layer_wise'
    #     v = self.reshape_tensor(v)
    #     def threshold_distribution(distribution, target_bin=255):
    #         """
    #         Return the best threshold value. 
    #         Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    #         Args:
    #             distribution: list, activations has been processed by histogram and normalize,size is 2048
    #             target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    #         Returns:
    #             target_threshold: int, num of bin with the minimum KL 
    #         """   
    #         # normalize
    #         distribution = distribution / sum(distribution)
    #         # distribution = distribution[1:]
    #         length = distribution.size
    #         threshold_sum1 = sum(distribution[:(length - target_bin) // 2])
    #         threshold_sum2 = sum(distribution[(length + target_bin) // 2:])
    #         kl_divergence = np.zeros((length - target_bin) // 2)
    #         # print(kl_divergence.shape, length)
    #         for threshold in range(target_bin, length, 2):
    #             sliced_nd_hist = copy.deepcopy(distribution[(length - threshold) // 2:(length + threshold) // 2])           

    #             # generate reference distribution p
    #             p = sliced_nd_hist.copy()
    #             if p[0] == 0 and p[threshold - 1] == 0:
    #                 kl_divergence[(threshold - target_bin) // 2] = 1000
    #                 threshold_sum1 = threshold_sum1 - distribution[(length - threshold) // 2 - 1]
    #                 threshold_sum2 = threshold_sum2 - distribution[(length + threshold) // 2]
    #                 continue
    #             p[0] += threshold_sum1
    #             p[threshold-1] += threshold_sum2
    #             threshold_sum1 = threshold_sum1 - distribution[(length - threshold) // 2 - 1]
    #             threshold_sum2 = threshold_sum2 - distribution[(length + threshold) // 2]
    #             # is_nonzeros[k] indicates whether hist[k] is nonzero
    #             is_nonzeros = (p != 0).astype(np.int64)
    #             # 
    #             quantized_bins = np.zeros(target_bin, dtype=np.float64)
    #             # calculate how many bins should be merged to generate quantized distribution q
    #             num_merged_bins = sliced_nd_hist.size // target_bin
                
    #             # merge hist into num_quantized_bins bins
    #             for j in range(target_bin):
    #                 start = j * num_merged_bins
    #                 stop = start + num_merged_bins
    #                 quantized_bins[j] = sliced_nd_hist[start:stop].sum()
    #             quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
                
    #             # expand quantized_bins into p.size bins
    #             q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
    #             for j in range(target_bin):
    #                 start = j * num_merged_bins
    #                 if j == target_bin - 1:
    #                     stop = -1
    #                 else:
    #                     stop = start + num_merged_bins
    #                 norm = is_nonzeros[start:stop].sum()
    #                 if norm != 0:
    #                     q[start:stop] = float(quantized_bins[j]) / float(norm)
    #             q[p == 0] = 0
    #             # p = _smooth_distribution(p) # with some bugs, need to fix
    #             # q = _smooth_distribution(q)
    #             # p[p == 0] = 0.0001
    #             # q[q == 0] = 0.0001
    #             p[p == 0] = 1e-12
    #             q[q == 0] = 1e-12
                
    #             # calculate kl_divergence between q and p
    #             t = stats.entropy(p, q)
    #             # print((threshold - target_bin) // 2)
    #             kl_divergence[(threshold - target_bin) // 2]  = t
    #             # ot = own_kl(p, q)

    #         min_kl_divergence = np.argmin(kl_divergence)
    #         # print(np.min(kl_divergence))
    #         threshold_value = min_kl_divergence + target_bin // 2 + 1

    #         return threshold_value 
    #     # op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** (self.bit_type.bits - 1))
    #     # op_thr = (op_bin + 0.5) * torch.max(-self.min_val, self.max_val) / self.bins
        
    #     cur_max = v.max(axis=1).values.max()
    #     cur_min = v.min(axis=1).values.min()
    #     absmax_val = torch.max(-cur_min, cur_max).item()     
    #     self.histc = torch.histc(v, bins=self.bins, min=-absmax_val, max=absmax_val)
    #     op_bin = threshold_distribution(self.histc.cpu().numpy(), 2 ** self.bit_type.bits - 1)
    #     per_num_bin = 2 * torch.max(-cur_min, cur_max) / self.bins
    #     cur_max = (op_bin + 0.5) * per_num_bin

    #     if self.max_val is None:
    #         self.max_val = cur_max
    #     else:
    #         self.max_val = self.max_val + \
    #             self.kldiv_sigma * (cur_max - self.max_val)

    #     if self.min_val is None:
    #         self.min_val = cur_min
    #     else:
    #         self.min_val = self.min_val + \
    #             self.kldiv_sigma * (cur_min - self.min_val)

    # def get_quantization_params(self, *args, **kwargs):
    #     max_val = self.max_val
    #     min_val = self.min_val

    #     qmax = self.bit_type.upper_bound
    #     qmin = self.bit_type.lower_bound

    #     scale = torch.ones_like(max_val, dtype=torch.float32)
    #     zero_point = torch.zeros_like(max_val, dtype=torch.int64)

    #     scale = max_val / (float(qmax - qmin) / 2)
    #     scale.clamp_(self.eps)
    #     zero_point = torch.zeros_like(
    #         max_val, dtype=torch.int64)
    #     return scale, zero_point