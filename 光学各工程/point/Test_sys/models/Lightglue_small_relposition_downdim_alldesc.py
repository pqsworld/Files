# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import models.performer_pytorch as PF
import numpy as np

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, H, W):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = H, W
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

class KeypointEncoderMore(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores, dbvalue):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1), dbvalue.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))        

class KeypointEncoderOnly(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, scores, dbvalue):
        inputs = [scores.unsqueeze(1), dbvalue.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))   

class KeypointEncoderOne(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([1] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, dbvalue):
        # inputs = [scores.unsqueeze(1), dbvalue.unsqueeze(1)]
        inputs = dbvalue.unsqueeze(1)
        return self.encoder(inputs)   

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        M = 2   # 2: [x, y] 3: [x, y, ori] 4: [x, y, ori, dbvalue]
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encode position vector """
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = F.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def attention_autoregressive(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    autoregressive_mask = torch.tril(torch.ones_like(scores)).to(scores.device)
    scores = scores * autoregressive_mask
    prob = F.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def linear_attention(query, key, value):
    dim = query.shape[1]
    soft_query = F.softmax(query, dim=1)
    soft_key = F.softmax(key, dim=-1)
    scores = torch.einsum('bdhn,bdhm->bhnm', soft_query, soft_key)
    # scores = torch.einsum('bdhn,bdhm->bhnm', soft_query, soft_key) / dim**.5
    # prob = F.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', scores, value), scores

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_cached_rotary_emb(
        freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.nb_features = self.dim // 2
        # self.merge = nn.Sequential(nn.Conv1d(d_model, d_model // 8, kernel_size=1),
        #              nn.Conv1d(d_model // 8, d_model, kernel_size=1)
        # )
        self.merge = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=1))

        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.fast_attention = PF.FastAttention(self.dim, self.nb_features)

    def forward(self, query, key, value, encoding):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        if encoding is not None:
            query = apply_cached_rotary_emb(encoding, query.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            key = apply_cached_rotary_emb(encoding, key.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #     # autoregressive
        #     x, _ = attention_autoregressive(query, key, value)
        # else:
        #     # Normal Attention
        #     x, _ = attention(query, key, value)

        # Normal Attention
        x, _ = attention(query, key, value)
        
        # # Dual-softmax Attention
        # x, _ = linear_attention(query, key, value)

        # # Performer
        # query = query.permute(0 ,2, 3, 1) # bdhn->bhdn
        # key = key.permute(0, 2, 3, 1)       
        # value = value.permute(0 ,2, 3, 1)
        # x = self.fast_attention(query, key, value).permute(0, 3, 1, 2)    # bhnd -> bdhn

        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2 // 8, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, encode):
        message = self.attn(x, source, source, encode)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, head_num: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, head_num)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, encoding0, encoding1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
                encode0, encode1 = None, None
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                encode0, encode1 = encoding0, encoding1
            delta0, delta1 = layer(desc0, src0, encode0), layer(desc1, src1, encode1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def sigmoid_log_double_softmax(
        sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(
        sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0 + scores1 + certainties)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True) # nn.Conv1d(dim, 1, kernel_size=1) # nn.Linear(dim, 1, bias=True)  
        self.final_proj = nn.Linear(dim, dim, bias=True) # nn.Conv1d(dim, dim, kernel_size=1) # nn.Linear(dim, dim, bias=True) 

        nn_weight = torch.nn.Parameter(torch.tensor(dim ** 0.5))    # 8
        self.register_parameter('nn_weight', nn_weight)

        self.nn_bias = nn.Parameter(torch.tensor(0.))
        self.nn_negative_slope = 0.1

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def scores(self, desc0: torch.Tensor, desc1: torch.Tensor):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1
    
    def forward_all(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25
        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        # print(scores.shape)
        m0 = torch.sigmoid(z0).squeeze(-1)
        m1 = torch.sigmoid(z1).squeeze(-1)
        # print(m0.shape)
        return scores, sim, m0, m1
    
    def forward_all_nn(self, desc0: torch.Tensor, desc1: torch.Tensor, nn_sim: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25

        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)
        sim = sim + nn_sim * self.nn_weight

        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        # print(scores.shape)
        m0 = torch.sigmoid(z0).squeeze(-1)
        m1 = torch.sigmoid(z1).squeeze(-1)
        # print(m0.shape)
        return scores, sim, m0, m1

    def forward_all_nn_affine(self, desc0: torch.Tensor, desc1: torch.Tensor, nn_sim: torch.Tensor):
        """ build assignment matrix from descriptors """
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**.25, mdesc1 / d**.25

        sim = torch.einsum('bmd,bnd->bmn', mdesc0, mdesc1)

        # sim = sim + nn_sim * self.nn_weight
        sim = sim + F.leaky_relu(self.nn_weight * nn_sim + self.nn_bias, negative_slope=self.nn_negative_slope)

        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        # print(scores.shape)
        m0 = torch.sigmoid(z0).squeeze(-1)
        m1 = torch.sigmoid(z1).squeeze(-1)
        # print(m0.shape)
        return scores, sim, m0, m1

class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            # nn.Conv1d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """ get confidence tokens """
        return (
            self.token(desc0.detach().float()).squeeze(-1),
            self.token(desc1.detach().float()).squeeze(-1))


class LightAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, head_num: int):
        super().__init__()        
        self.names = layer_names
        self.n_layers = len(layer_names)

        self.depth_confidence = -1 # 0.95    # early stopping, disable with -1 
        self.width_confidence = [0.9725, 0.960, 0.963]    # point pruning, disable with -1
        self.prune_layer = [3, 5, 7]
        self.last_pruning_thr = 0     # 0.5

        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, head_num)
            for _ in range(self.n_layers)])

        self.log_assignment = nn.ModuleList(
            [MatchAssignment(feature_dim) for _ in range(self.n_layers)])
        
        self.token_confidence = nn.ModuleList([
            TokenConfidence(feature_dim) for _ in range(self.n_layers-1)])

    def confidence_threshold(self, layer_index: int) -> float:
        """ scaled confidence threshold """
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return torch.clamp(torch.tensor(threshold), min=0, max=1)

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor,
                         layer_index: int, match_thr: float) -> torch.Tensor:
        """ mask points which should be removed """
        threshold = self.confidence_threshold(layer_index)
        if confidences is not None:
            scores = torch.where(
                confidences > threshold, scores, scores.new_tensor(1.0))
        return scores > match_thr # (1 - self.width_confidence)

    def check_if_stop(self,
                      confidences0: torch.Tensor,
                      confidences1: torch.Tensor,
                      layer_index: int, num_points: int) -> torch.Tensor:
        """ evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_threshold(layer_index)
        pos = 1.0 - (confidences < threshold).float().sum() / num_points
        return pos > self.depth_confidence

    def forward(self, desc0, desc1, encoding0, encoding1, nn_sim):
        b, fea_dim, m = desc0.shape
        n = desc1.shape[-1]
        do_early_stop = self.depth_confidence > 0
        cal_point_score = len(self.width_confidence) > 0
        # 是否剪枝
        do_point_pruning = False # self.width_confidence > 0
        scores_all, scores0_all, scores1_all = [], [], []
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=desc0.device)[None].repeat(b, 1)
            ind1 = torch.arange(0, n, device=desc1.device)[None].repeat(b, 1)
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            if name == 'cross':
                src0, src1 = desc1, desc0
                encode0, encode1 = None, None
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                encode0, encode1 = encoding0, encoding1
            delta0, delta1 = layer(desc0, src0, encode0), layer(desc1, src1, encode1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

            # 最后一层和
            if name == 'self' or i == self.n_layers - 1:        
                continue

            # # 测试第4层 - 3
            # if i == self.n_layers - 9:
            #     break
            
            desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)
            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0, token1, (i - 1) / 2, m+n):
                    break

            if cal_point_score:
                # scores, _ = self.log_assignment[i](desc0, desc1)
                scores, _, scores0, scores1 = self.log_assignment[i].forward_all_nn_affine(desc0, desc1, nn_sim)
                scores_all.append(scores)
                scores0_all.append(scores0)
                scores1_all.append(scores1)
                if do_point_pruning and i in self.prune_layer:
                    mask0 = self.get_pruning_mask(token0, scores0, (i - 1) / 2, 1 - self.width_confidence[int((i - self.prune_layer[0]) / 2)])  #.unsqueeze(-1).repeat(1, 1, fea_dim)
                    mask1 = self.get_pruning_mask(token1, scores1, (i - 1) / 2, 1 - self.width_confidence[int((i - self.prune_layer[0]) / 2)])  #.unsqueeze(-1).repeat(1, 1, fea_dim)
                    ind0, ind1 = ind0[mask0][None], ind1[mask1][None]

                    desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
                    if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
                        break
                    # print(encoding0.shape)
                    encoding0 = encoding0[:, :, mask0][:, None]
                    encoding1 = encoding1[:, :, mask1][:, None]
                    prune0[:, ind0] += 1
                    prune1[:, ind1] += 1
            
            desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)  

        desc0, desc1 = desc0.transpose(1, 2), desc1.transpose(1, 2)      
        scores, _, scores0, scores1 = self.log_assignment[i].forward_all_nn_affine(desc0, desc1, nn_sim)

        scores_final = deepcopy(scores) 
        # scores_final = torch.zeros((1, m + 1, n + 1)).to(scores.device) 
        # scores_final_prune = torch.zeros((1, m + 2, n + 2)).to(scores.device)
        
        if do_point_pruning:            
            mask0 = self.get_pruning_mask(token0, scores0, (i - 1) / 2, self.last_pruning_thr).squeeze()  #.unsqueeze(-1).repeat(1, 1, fea_dim)
            mask1 = self.get_pruning_mask(token1, scores1, (i - 1) / 2, self.last_pruning_thr).squeeze()  #.unsqueeze(-1).repeat(1, 1, fea_dim)
            last_prune_mask0 = torch.cat((mask0, torch.ones(1).to(mask0.device).bool()), dim=-1)
            last_prune_mask1 = torch.cat((mask1, torch.ones(1).to(mask1.device).bool()), dim=-1)
            # ind0, ind1 = ind0[~mask0][None], ind1[~mask1][None]

            scores[:, ~(last_prune_mask0), :] = scores[:, :-1, :-1].min() - 1
            scores[:, :, ~(last_prune_mask1)] = scores[:, :-1, :-1].min() - 1
            # desc0, desc1 = desc0[mask0][None], desc1[mask1][None]
            # if desc0.shape[-2] == 0 or desc1.shape[-2] == 0:
            #     break
            # prune0[:, ind0] += 1
            # prune1[:, ind1] += 1 

            pre_prune_mask0, pre_prune_mask1 = (prune0 < 4).squeeze(), (prune1 < 4).squeeze() # (prune0 < (i + 1) / 2).squeeze(), (prune1 < (i + 1) / 2).squeeze()

            pre_prune_mask0 = torch.cat((pre_prune_mask0, torch.zeros(1).to(pre_prune_mask0.device).bool()), dim=-1)
            pre_prune_mask1 = torch.cat((pre_prune_mask1, torch.zeros(1).to(pre_prune_mask1.device).bool()), dim=-1)
            
            scores_final[:, pre_prune_mask0, :] = scores[:, :-1, :-1].min() - 1
            scores_final[:, :, pre_prune_mask1] = scores[:, :-1, :-1].min() - 1
            scores_final[(scores_final != (scores[:, :-1, :-1].min() - 1))] = scores.view(-1)

            # scores_final_prune[:-1, :-1] = scores_final
            # scores_final_prune[:-2, -1] = prune0
            # scores_final_prune[-1, :-2] = prune1
            
        return scores_final # scores_all[-1]   # , scores0_all, scores1_all

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def dual_softmax(scores, alpha):
    
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)
    
    couplings_max_col = torch.max(couplings, dim=1).values.unsqueeze(1).repeat(1, m+1, 1)
    couplings_max_row = torch.max(couplings, dim=2).values.unsqueeze(2).repeat(1, 1, n+1)
    # print(couplings_max_col.shape, couplings_max_row.shape)
    out_scores = (F.softmax(couplings - couplings_max_col, 1) * F.softmax(couplings - couplings_max_row, 2)).log()
    return out_scores


def get_hammingsim(desc0, desc1):
    half_dim = desc0.shape[1] // 2
    sim_same = (torch.einsum('bdn,bdm->bnm', desc0[:, :half_dim, :], desc1[:, :half_dim, :]) + half_dim) / 2
    sim_neg = (torch.einsum('bdn,bdm->bnm', desc0[:, half_dim:, :], desc1[:, half_dim:, :]) + half_dim) / 2
    sim_neg  = torch.max(sim_neg, half_dim - sim_neg)
    sim = (sim_same + sim_neg) / desc0.shape[1]
    # sim = F.softmax(sim, dim=-1)
    return sim


class Lightglue_small_relposition_downdim_alldesc(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'input_dim': 256,
        'descriptor_dim': 64,
        'weights': 'indoor',
        'keypoint_encoder': [16, 32],    # [16, 32, 64],
        'GNN_layers': ['self', 'cross'] * 5,
        'sinkhorn_iterations': 10,
        'match_threshold': 0.2,
        'head_num': 2,
    }

    def __init__(self):
        super().__init__()
        self.config = {**self.default_config}

        self.kenc = KeypointEncoderOnly( # KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.relkenc = LearnableFourierPositionalEncoding( # KeypointEncoder(
            self.config['descriptor_dim'] // self.config['head_num'], self.config['descriptor_dim'] // self.config['head_num'])

        self.gnn = LightAttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'], self.config['head_num'])

        self.input_proj = nn.Conv1d(
            self.config['input_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        # self.final_proj = nn.Conv1d(
        #     self.config['descriptor_dim'], self.config['descriptor_dim'],
        #     kernel_size=1, bias=True)

        # bin_score = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(str(path)))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # desc0, desc1 = data['descriptors0_same'], data['descriptors1_same']
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        # desc0_neg, desc1_neg = data['descriptors0_neg'], data['descriptors1_neg']   # B x 128 x NA
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        # [0, 1]相似度
        nn_sim = get_hammingsim(desc0, desc1)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['H'], data['W'])
        kpts1 = normalize_keypoints(kpts1, data['H'], data['W'])

        # input_proj downsample
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        # # 0/45 desc share parameters
        # quart_input_dim = self.config['input_dim'] // 4
        # desc0_0 = self.input_proj(torch.cat((desc0[:, :quart_input_dim, :], desc0[:, 2*quart_input_dim:3*quart_input_dim, :]), dim=1))
        # desc0_45 = self.input_proj(torch.cat((desc0[:, quart_input_dim:2*quart_input_dim, :], desc0[:, 3*quart_input_dim:, :]), dim=1))
        # desc1_0 = self.input_proj(torch.cat((desc1[:, :quart_input_dim, :], desc1[:, 2*quart_input_dim:3*quart_input_dim, :]), dim=1))
        # desc1_45 = self.input_proj(torch.cat((desc1[:, quart_input_dim:2*quart_input_dim, :], desc1[:, 3*quart_input_dim:, :]), dim=1))
        # desc0 = torch.cat((desc0_0, desc0_45), dim=1)
        # desc1 = torch.cat((desc1_0, desc1_45), dim=1)

        # Keypoint absolute MLP encoder.
        desc0 = desc0 + self.kenc(data['angles0'], data['dbvalue0'])
        desc1 = desc1 + self.kenc(data['angles1'], data['dbvalue1'])

        # kpts0 = torch.cat((kpts0, data['angles0'].unsqueeze(-1), data['dbvalue0'].unsqueeze(-1)), dim=-1)
        # kpts1 = torch.cat((kpts1, data['angles1'].unsqueeze(-1), data['dbvalue1'].unsqueeze(-1)), dim=-1)

        # desc0 = desc0 + self.kenc(data['dbvalue0'])
        # desc1 = desc1 + self.kenc(data['dbvalue1'])

        # kpts0 = torch.cat((kpts0, data['angles0'].unsqueeze(-1)), dim=-1)
        # kpts1 = torch.cat((kpts1, data['angles1'].unsqueeze(-1)), dim=-1)

        # Keypoint relative MLP encoder.
        encoding0 = self.relkenc(kpts0)
        encoding1 = self.relkenc(kpts1)

        # Multi-layer Transformer network.
        scores = self.gnn(desc0, desc1, encoding0, encoding1, nn_sim)

        # # Final MLP projection.
        # mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # # Compute matching descriptor distance.
        # scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # scores = scores / self.config['descriptor_dim']**.5

        # # Dual-Softmax
        # scores = dual_softmax(scores, self.bin_score)
        # scores = F.softmax(scores, 1) * F.softmax(scores, 2)

        # # Run the optimal transport.
        # scores = log_optimal_transport(
        #     scores, self.bin_score,
        #     iters=self.config['sinkhorn_iterations'])

        # # Get the matches with score above "match_threshold".
        # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        # indices0, indices1 = max0.indices, max1.indices
        # mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)  #当二者互为匹配点时才认为是匹配点
        # mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        # zero = scores.new_tensor(0)
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        # valid1 = mutual1 & valid0.gather(1, indices1)
        # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return scores

if __name__ == "__main__":
    config = {
        'descriptor_dim': 128,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 5,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    model = Superglue_small()
    total = 0
    for name, parameters in model.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.5fM" % (total / 1e6))

    data = {}
    data.update({'descriptors0':torch.rand(2,128,347)})
    data.update({'descriptors1':torch.rand(2,128,361)})
    data.update({'keypoints0':torch.rand(2,347,2)})
    data.update({'keypoints1':torch.rand(2,361,2)})
    data.update({'scores0':torch.rand(2,347)})
    data.update({'scores1':torch.rand(2,361)})
    data.update({'H':128})
    data.update({'W':128})

    result = model(data)



