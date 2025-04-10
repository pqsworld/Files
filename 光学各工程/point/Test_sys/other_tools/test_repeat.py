import os
import cv2
import torch
import numpy as np
import pandas as pd

def draw_keypoints_pair(input_img, input_pts, label_pts, color=(0, 255, 0), radius=3, s=3):
    # anchor = int(input_img['img_1'].shape[1])
    # img = np.hstack((input_img['img_1'], input_img['img_2'])) * 255
    img = np.repeat(cv2.resize(input_img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)

    for c in np.stack(input_pts):
        # c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (255, 0, 0), thickness=-1)
        # cv2.putText(img, str(round(c[2],5)), tuple((s * c[:2]).astype(int)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.25, (0, 0, 255), 1)
    
    # flag = (label_pts[:, 1]<anchor) & (label_pts[:, 0]<anchor) & (label_pts[:, 0]>0) & (label_pts[:, 1]>0)
    # label_pts = label_pts[flag, :]
    if label_pts.size == 0:
        return img

    for c in np.stack(label_pts):
        if c.size == 1:
            break
        # c[0] += anchor
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, (0, 255, 0), thickness=1)
    
    return img

def get_dis(p_a, p_b):
        if p_a.shape == torch.Size([]) or p_b.shape[0] == torch.Size([]):
            return torch.tensor([])
        if p_a.shape == torch.Size([2]):
            p_a = p_a.unsqueeze(0)
        if p_b.shape == torch.Size([2]):
            p_b = p_b.unsqueeze(0)
        eps = 1e-12
        x = torch.unsqueeze(p_a[:, 0], 1) - torch.unsqueeze(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
        y = torch.unsqueeze(p_a[:, 1], 1) - torch.unsqueeze(p_b[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + eps)
        return dis

def get_point_pair_repeat(kptA_tensor, kptB_tensor, correspond=2):  # 获得匹配点
        dis = get_dis(kptA_tensor, kptB_tensor)
        if dis.shape[0] == 0 or dis.shape[1] == 0 or dis.shape == torch.Size([]):
            return torch.tensor([]), torch.tensor([])
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < correspond
        a2b_min_id = a2b_min_id[ch]
    
        return a2b_min_id, ch


if __name__ == '__main__':
    eps = 1e-6
    path_net = '/hdd/file-input/qint/6159_parallel/Test_sys/repeat_data/net/net_100/'
    path_sift = '/hdd/file-input/qint/6159_parallel/Test_sys/repeat_data/sift_filter/'

    save_dir = '/hdd/file-input/qint/6159_parallel/Test_sys/result/'
    save_dir = os.path.join(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    '''net/sift文件地址列表'''
    net_list = []
    sift_list = []
    for root, folder, file in os.walk(path_net):
        for f in file:
            net_list.append(path_net + f)
            sift_list.append(path_sift + f)
        pass
    
    content = []
    count = len(net_list)
    i = 0
    for p_net, p_sift in zip(net_list, sift_list):
        i += 1
        print("progress:{0}%".format(round((i + 1) * 100 / count)), end="\r")

        name = p_net.split('/')[-1][:-4]
        pts_net = np.load(p_net)    #(y, x)
        pts_sift = np.load(p_sift)

        pts_net = torch.tensor(pts_net[:, [1, 0]], dtype=torch.float32)     #(x, y)
        pts_sift = torch.tensor(pts_sift[:, [1, 0]], dtype=torch.float32)

        '''计算重复率: 距离<2pixel'''
        match_idx, _ = get_point_pair_repeat(pts_sift, pts_net, correspond=2)
        net_repeat = len(set(match_idx.numpy().tolist())) / (len(pts_sift.squeeze()) + eps)
        # print("匹配点数: {},网络点数: {},重复率(dis<2): {}".format(len(match_idx), len(pts_net), net_repeat))

        '''计算重复率: 完全重合'''
        match_idx_2, _ = get_point_pair_repeat(pts_sift, pts_net, correspond=1)
        sift_repeat = len(set(match_idx_2.numpy().tolist())) / (len(pts_sift.squeeze()) + eps)
        # print("重合点数: {},传统点数: {},重复率(dis=0): {}".format(len(match_idx_2), len(pts_sift), sift_repeat))

        '''输出示意图:绿：sift，蓝：net'''
        img = np.zeros((136, 36))
        out_img = draw_keypoints_pair(img, pts_net, pts_sift)
        f = save_dir + name + ".bmp"
        cv2.imwrite(str(f), out_img)

        '''输出csv统计表'''
        content.append([name, len(set(match_idx.numpy().tolist())), len(set(match_idx_2.numpy().tolist())), len(pts_net), net_repeat, len(pts_sift), sift_repeat])

    df = pd.DataFrame(
        content,
        columns=[
            'name',
            'match_num',
            'coincident_num',
            'net_pointsnum',
            'net_repeat',
            'sift_pointsnum',
            'sift_repeat'
        ])
    df.to_csv(os.path.join(save_dir, 'result.csv'))
    pass