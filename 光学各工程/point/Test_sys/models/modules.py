import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block()

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def forward(self, x):

        sobel = kornia.spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = kornia.spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = kornia.spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = torch.cat([dx, dy, dx**2., dy**2., dx*dy, dxy, dxy**2., dxx, dyy, dxx*dyy], dim=1)

        return hc_feats


class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()

    def forward(self, x):
        x = self.conv2(self.conv1(self.conv0(x)))
        return x


def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class NonMaxSuppression(torch.nn.Module):
    '''
        NonMaxSuppression class
    '''
    def __init__(self, thr=0.0, nms_size=5):
        nn.Module.__init__(self)
        padding = nms_size // 2
        self.max_filter = torch.nn.MaxPool2d(kernel_size=nms_size, stride=1, padding=padding)
        self.thr = thr

    def forward(self, scores):

        # local maxima
        maxima = (scores == self.max_filter(scores))

        # remove low peaks
        maxima *= (scores > self.thr)

        return maxima.nonzero().t()[2:4]


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return F.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)           # max scores of kernel windows mask

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def simple_nms_modified(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        x_mpool, x_indexes =  F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius, return_indices=True)
        x_indexes_rank = torch.tensor(list(range(x_indexes.shape[-1]*x_indexes.shape[-2]))).view(x_indexes.shape[-2], x_indexes.shape[-1]).repeat(x_indexes.shape[0], x_indexes.shape[1], 1, 1).to(x_indexes.device)
        mask = x_indexes == x_indexes_rank
        return x_mpool, mask

    zeros = torch.zeros_like(scores)
    # max_mask = scores == max_pool(scores)           # max scores of kernel windows mask
    _, max_mask = max_pool(scores) 

    for _ in range(2):
        supp_mask = max_pool(max_mask.float())[0] > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        # new_max_mask = supp_scores == max_pool(supp_scores)
        new_max_mask = max_pool(supp_scores)[1]
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = F.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index, :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        descriptors_ = F.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors


class DKD(nn.Module):
    def __init__(self, radius=2, top_k=0, scores_th=0.2, n_limit=20000):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius            # nms radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)

        # local xy grid
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)     # ex: [-2, -1, 0, 1, 2]
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]

    def detect_keypoints(self, scores_map, sub_pixel=True):
        b, c, h, w = scores_map.shape       # [B, 1, H, W]
        scores_nograd = scores_map.detach()
        # nms_scores = simple_nms(scores_nograd, self.radius)
        nms_scores = simple_nms_modified(scores_nograd, self.radius)

        # remove border
        nms_scores[:, :, :self.radius + 1, :] = 0
        nms_scores[:, :, :, :self.radius + 1] = 0
        nms_scores[:, :, h - self.radius:, :] = 0
        nms_scores[:, :, :, w - self.radius:] = 0

        # detect keypoints without grad
        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = topk.indices  # B x top_k
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if masks.sum() == 0:                # 如果阈值过大，取全局均值为阈值
                    th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []  # list, B x (any size)
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero(as_tuple=False)[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[:self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        eps = 1e-12
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(patches)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                patch_scores = patch[indices_kpt]  # M x (kernel**2)

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()  # M * (kernel**2), in [0, 1]

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = x_exp @ self.hw_grid / (x_exp.sum(dim=1)[:, None] + eps)  # Soft-argmax, Mx2

                hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                           dim=-1) ** 2
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / (x_exp.sum(dim=1) + eps)

                # compute result keypoints
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / keypoints_xy.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                kptscore = F.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms / keypoints_xy_nms.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                kptscore = F.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(None)
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores

    def forward(self, scores_map, descriptor_map, sub_pixel=False):
        """
        :param scores_map:  Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        # 点（w,h） range:(-1~1, -1~1)，第二项用于计算dispersity peak loss， 点对应分数
        keypoints, scoredispersitys, kptscores = self.detect_keypoints(scores_map,
                                                                       sub_pixel)

        descriptors = sample_descriptor(descriptor_map, keypoints, sub_pixel)

        # keypoints: B M 2
        # descriptors: B M D
        # scoredispersitys:
        return keypoints, descriptors, kptscores, scoredispersitys
