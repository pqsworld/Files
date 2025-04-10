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


def attention(query, key, value, hanmingsim):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = F.softmax(scores, dim=-1)
    # 增加汉明距作为引导
    if hanmingsim is not None:
        prob = (prob + hanmingsim[:, None, :, :]) / 2
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def attention_extern(query, key, value, hanmingsim):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = F.softmax(scores, dim=-1)
    # 增加汉明距作为引导
    if hanmingsim is not None:
        prob = hanmingsim.unsqueeze(1).repeat(1, prob.shape[1], 1, 1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def linear_attention(query, key, value):
    dim = query.shape[1]
    soft_query = F.softmax(query, dim=1)
    soft_key = F.softmax(key, dim=-1)
    scores = torch.einsum('bdhn,bdhm->bhnm', soft_query, soft_key)
    # scores = torch.einsum('bdhn,bdhm->bhnm', soft_query, soft_key) / dim**.5
    # prob = F.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', scores, value), scores


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.nb_features = self.dim // 2
        self.merge = nn.Sequential(nn.Conv1d(d_model, d_model // 8, kernel_size=1),
                     nn.Conv1d(d_model // 8, d_model, kernel_size=1)
        )
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.fast_attention = PF.FastAttention(self.dim, self.nb_features)

    def forward(self, query, key, value, hanmingsim):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # Normal Attention
        x, _ = attention_extern(query, key, value, hanmingsim)
        
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

    def forward(self, x, source, hanmingsim):
        message = self.attn(x, source, source, hanmingsim)
        return self.mlp(torch.cat([x, message], dim=1))

def get_hammingsim(desc0, desc1):
    half_dim = desc0.shape[1] // 2
    sim_same = (torch.einsum('bdn,bdm->bnm', desc0[:, :half_dim, :], desc1[:, :half_dim, :]) + half_dim) / 2
    sim_neg = (torch.einsum('bdn,bdm->bnm', desc0[:, half_dim:, :], desc1[:, half_dim:, :]) + half_dim) / 2
    sim_neg  = torch.max(sim_neg, half_dim - sim_neg)
    sim = (sim_same + sim_neg) / desc0.shape[1]
    soft_sim = F.softmax(sim, dim=-1)
    return soft_sim

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, desc0_all, desc1_all):
        used_name = []
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
                src0_all, src1_all = desc1_all, desc0_all
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                src0_all, src1_all = desc0_all, desc1_all

            if 'self' in used_name and 'cross' in used_name:
                delta0, delta1 = layer(desc0, src0, None), layer(desc1, src1, None)
            else:
                hammingsim0 = get_hammingsim(desc0_all, src0_all)
                hammingsim1 = get_hammingsim(desc1_all, src1_all)
                delta0, delta1 = layer(desc0, src0, hammingsim0), layer(desc1, src1, hammingsim1)

            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            used_name.append(name)
        return desc0, desc1


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


class Superglue_small_alldesc(nn.Module):
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
        'descriptor_dim': 128,
        'weights': 'indoor',
        'keypoint_encoder': [16, 32, 64],
        'GNN_layers': ['self', 'cross'] * 6, # 5,
        'sinkhorn_iterations': 10,
        'match_threshold': 0.2,
    }

    def __init__(self):
        super().__init__()
        self.config = {**self.default_config}

        self.kenc = KeypointEncoderMore( # KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(str(path)))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0_same'], data['descriptors1_same']
        desc0_all, desc1_all = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

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

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['angles0'], data['dbvalue0'])
        desc1 = desc1 + self.kenc(kpts1, data['angles1'], data['dbvalue1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1, desc0_all, desc1_all)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # # Dual-Softmax
        # scores = dual_softmax(scores, self.bin_score)
        # scores = F.softmax(scores, 1) * F.softmax(scores, 2)

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

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



