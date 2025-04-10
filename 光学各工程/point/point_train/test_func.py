import torch
import torch.nn.functional as F

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        x_mpool, x_indexes =  F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius, return_indices=True)
        x_indexes_rank = torch.tensor(list(range(x_indexes.shape[-1]*x_indexes.shape[-2]))).view(x_indexes.shape[-1], x_indexes.shape[-2]).repeat(x_indexes.shape[0], x_indexes.shape[1], 1, 1)
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

if __name__ == '__main__':
    a = torch.tensor([[[[1, 3, 4, 4], [0 , -1, 1, 4], [0, 1, 2, 4], [0, 0, 0, 0]]]], dtype=torch.float32)
    print(a.shape)
    a_s = simple_nms(a, 2)
    print(a_s)