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
from timeit import default_timer as timer

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        out = (out - 0.5)*6
        return out

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
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        return self.encoder(inputs)

class KeypointDecoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.decoder = MLP([feature_dim] + layers + [2])
        nn.init.constant_(self.decoder[-1].bias, 0.0)

    def forward(self, inputs):
        return self.decoder(inputs)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

    # query = torch.nn.functional.softmax(query, dim=1)
    # key = torch.nn.functional.softmax(key, dim=-1)
    # scores = torch.einsum('bdhn,bdhm->bhnm', query, key)
    # return torch.einsum('bhnm,bdhm->bdhn', scores, value), scores

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Sequential(nn.Conv1d(d_model, 16, kernel_size=1),
        nn.Conv1d(16, d_model, kernel_size=1)
        )
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        # x, _ = attention(query, key, value)
        # return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

        query, key, value = [l(x).view(batch_dim, self.num_heads,self.dim, -1).permute(0,2,1,3)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.permute(0,2,1,3).contiguous().view(batch_dim, self.dim*self.num_heads, -1))



class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, 32, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
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
    return couplings

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def log_dual_softmax(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    Z = (F.softmax(couplings,dim=0)*F.softmax(couplings,dim=1)).log()
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        print(x.size())
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class Superglue_noscore_small_H(nn.Module):
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
        'GNN_layers': ['self'] * 5 +['cross'] * 5,
        # 'GNN_layers': ['self', 'cross'] * 9,
        # 'GNN_layers': ['self']*10 + ['cross'] * 2 + ['self'] * 2 + ['cross'] * 5,
        'keypoint_decoder': [64, 32, 16],
        'sinkhorn_iterations': 10,
        'match_threshold': 0.2,
    }

    def __init__(self):
        super().__init__()
        self.config = {**self.default_config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        self.kend = KeypointDecoder(
            self.config['descriptor_dim'], self.config['keypoint_decoder'])
        self.nonlinear = hsigmoid()

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
        desc0, desc1 = data['descriptors0'], data['descriptors1']
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
        # start = timer()
        desc0_encoder = desc0 + self.kenc(kpts0)
        desc1_encoder = desc1 + self.kenc(kpts1)
        # end = timer()
        # print("encoder: ",end -start)
        # start = timer()
        # Multi-layer Transformer network.
        desc0_encoder, desc1_encoder = self.gnn(desc0_encoder, desc1_encoder)
        # end = timer()
        # print("gnn: ",end -start)
        # start = timer()
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0_encoder), self.final_proj(desc1_encoder)
        
        kpt0 = self.kend(mdesc0 - desc0)
        kpt1 = self.kend(mdesc1 - desc1)
        
        kpt0 = self.nonlinear(kpt0)
        kpt1 = self.nonlinear(kpt1)

        # end = timer()
        # print("sinkhorn: ",end -start)
        # exit()
        return kpt0, kpt1

if __name__ == "__main__":
    config = {
        'descriptor_dim': 128,
        'weights': 'indoor',
        'keypoint_encoder': [16, 32, 64],
        'GNN_layers': ['self'] * 5 +['cross'] * 5,
        # 'GNN_layers': ['self', 'cross'] * 9,
        # 'GNN_layers': ['self']*10 + ['cross'] * 2 + ['self'] * 2 + ['cross'] * 5,
        'keypoint_decoder': [64, 32, 16],
        'sinkhorn_iterations': 10,
        'match_threshold': 0.2,
    }
    model = Superglue_noscore_small_H()
    total = 0
    for name, parameters in model.named_parameters():
        total += parameters.nelement()
        print(name, ":", parameters.size())
    print("Number of parameter: %.5fM" % (total / 1e6))

    data = {}
    data.update({'descriptors0':torch.rand(2,128,200)})
    data.update({'descriptors1':torch.rand(2,128,210)})
    data.update({'keypoints0':torch.rand(2,200,2)})
    data.update({'keypoints1':torch.rand(2,210,2)})
    data.update({'scores0':torch.rand(2,200)})
    data.update({'scores1':torch.rand(2,210)})
    data.update({'H':128})
    data.update({'W':128})

    result = model(data)



