import os
import cv2
import torch
import copy
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image, ImageDraw
from skimage.measure import ransac
from skimage.transform import AffineTransform

def stitch(img_a, img_p, model, save_path, image_name):

    save_img_a = Image.new("RGB", (img_a.size[0], img_a.size[1]))
    save_img_a.paste(img_a, (0, 0))
    a = np.array(save_img_a)
    img_a_a = cv2.warpAffine(a, model.params[:2], (32, 160), flags=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img_a_a,cv2.COLOR_RGB2GRAY)
    save_img_p = Image.new("RGB", (img_a.size[0], img_a.size[1]))
    save_img_p.paste(img_p, (0, 0))
    p = np.array(save_img_p)
    r = cv2.cvtColor(p,cv2.COLOR_RGB2GRAY)

    save_mg = Image.new("L", (img_a.size[0], img_a.size[1]))
    b = np.array(save_mg)
    img_merge = cv2.merge([b, g, r])
    cv2.imwrite(os.path.join(save_path, image_name), img_merge)
    return img_merge, img_a_a, img_p

def ransac_twice(image_path, img_a, img_p, save_img, index_, matched_left, matched_right, dist_list_return):
    # ransac
    contents = []
    save_img_ = copy.deepcopy(save_img)
    # save_img_a = copy.deepcopy(img_a.convert('RGB'))
    # save_img_p = copy.deepcopy(img_p.convert('RGB'))
    draw = ImageDraw.Draw(save_img_) 
    # draw_a = ImageDraw.Draw(save_img_a)
    # draw_p = ImageDraw.Draw(save_img_p) 
    
    nameAB = str(image_path[0]) + '_' + str(image_path[1])
    save_t_path = Path(image_path[2])
    os.makedirs(save_t_path, exist_ok=True)
    
    matched_left = np.array(matched_left)
    matched_right = np.array(matched_right)
    dist_list_return = np.array(dist_list_return)
    # 一些参数的初始化
    simility = 0
    same_area = 0
    valid_area = 0
    matched_num = 0
    match_after_ransac_num = 0
    model_00 = 0
    model_01 = 0
    model_02 = 0
    model_10 = 0
    model_11 = 0
    model_12 = 0

    if len(matched_right) > 3:
        model, inliers = ransac((matched_left[:, :2],matched_right[:, :2]),
                            AffineTransform, min_samples=8,
                            residual_threshold=1, max_trials=1000, random_state=2000)
        
        # model, inliers = cv2.findHomography(matched_left,
        #                         matched_right,
        #                         cv2.RANSAC)
        # inliers = inliers.flatten().astype(bool)
        
        # inliers = None
        if inliers is None:
            for idx, left_point in enumerate(matched_left):
                draw.rectangle((left_point[0] - 1, left_point[1] + 1, left_point[0] + 1, left_point[1] - 1), 'red', 'red')
                right_x = matched_right[idx][0] + 32
                right_y = matched_right[idx][1]
                draw.rectangle((right_x - 1, right_y + 1, right_x + 1, right_y - 1), 'red', 'red')
                newnew__weizhi = [tuple(left_point), (matched_right[idx][0] + 32, matched_right[idx][1])]
                draw.line(newnew__weizhi, fill='blue', width=1)

            # # 将特征点画在原图上
            # for left_point in matched_left:
            #     # draw_a.point((left_point[0], left_point[1]), 'red')
            #     draw_a.rectangle((left_point[0] - 1, left_point[1] + 1, left_point[0] + 1, left_point[1] - 1), 'red', 'red')
            # for right_point in matched_right:
            #     # draw_p.point((right_point[0], right_point[1]), 'red')
            #     draw_p.rectangle((right_point[0] - 1, right_point[1] + 1, right_point[0] + 1, right_point[1] - 1), 'red', 'red')

            new_name = nameAB + '.bmp'
            save_img_.save(os.path.join(save_t_path, new_name))
            # save_img_a.save(os.path.join(save_t_path, str('%.5d' % index_) + '_a.bmp'))
            # save_img_p.save(os.path.join(save_t_path, str('%.5d' % index_) + '_p.bmp'))
            
        else: 
            
            # model_00 = model[0,0]
            # model_01 = model[0,1]
            # model_02 = model[0,2]
            # model_10 = model[1,0]
            # model_11 = model[1,1]
            # model_12 = model[1,2]

            model_00 = model.params[0, 0]
            model_01 = model.params[0, 1]
            model_02 = model.params[0, 2]
            model_10 = model.params[1, 0]
            model_11 = model.params[1, 1]
            model_12 = model.params[1, 2]

            # 拼接获得融合图像
            merge_name = nameAB + '_merge_net.bmp'
            stitch(img_a, img_p, model, save_t_path, merge_name)
            # 计算相似度
            # enhance_img_a_name = image_path[0].replace('AB', '_B_binary')
            # enhance_img_p_name = image_path[1].replace('AB', '_B_binary')
            # simility, same_area, valid_area = compute_simility(enhance_img_a_name, enhance_img_p_name, model)
            # stitch_pin(img_a, img_p, model, save_t_path, merge_name)
            inlier_keypoints_left = matched_left[inliers][:, :2]
            inlier_keypoints_right = matched_right[inliers][:, :2]
            inlier_dist_list = dist_list_return[inliers]

            # # 将特征点画在原图上
            # for left_point in inlier_keypoints_left:
            #     draw_a.rectangle((left_point[0] - 1, left_point[1] + 1, left_point[0] + 1, left_point[1] - 1), 'red', 'red')
            # for right_point in inlier_keypoints_right:
            #     draw_p.rectangle((right_point[0] - 1, right_point[1] + 1, right_point[0] + 1, right_point[1] - 1), 'red', 'red')

            if len(inlier_keypoints_left) > 0:
                print("Number of matches:", matched_left.shape[0])
                matched_num = matched_left.shape[0]
                print("Number of inliers:", inliers.sum())
                match_after_ransac_num = inliers.sum()
            
            for idx, left_point in enumerate(inlier_keypoints_left):
                newnew__weizhi = [tuple(left_point), (inlier_keypoints_right[idx][0] + 32, inlier_keypoints_right[idx][1])]
                draw.rectangle((left_point[0] - 1, left_point[1] + 1, left_point[0] + 1, left_point[1] - 1), 'red', 'red')
                right_x = inlier_keypoints_right[idx][0] + 32
                right_y = inlier_keypoints_right[idx][1]
                draw.rectangle((right_x - 1, right_y + 1, right_x + 1, right_y - 1), 'red', 'red')
                draw.line(newnew__weizhi, fill='blue', width=1)
            
            new_name = nameAB + '_inlier.bmp'
            save_img_.save(os.path.join(save_t_path, new_name))
            # save_img_a.save(os.path.join(save_t_path, str('%.5d' % index_) + '_a.bmp'))
            # save_img_p.save(os.path.join(save_t_path, str('%.5d' % index_) + '_p.bmp'))
            # 保存匹配点坐标和距离
            save_csv_path = os.path.join(save_t_path, 'csv')
            if not os.path.exists(save_csv_path):
                os.mkdir(save_csv_path)
            left_x = inlier_keypoints_left[:, 0]
            left_y = inlier_keypoints_left[:, 1]
            right_x = inlier_keypoints_right[:, 0]
            right_y = inlier_keypoints_right[:, 1]
            print('simility: ', simility)
            df = pd.DataFrame({'path1': image_path[0], 'path2': image_path[1], 'left_x': left_x, 'left_y': left_y, 'right_x': right_x, 'right_y': right_y,
            'inlier_dist_list': inlier_dist_list})
            df.to_csv(os.path.join(save_csv_path, nameAB + '.csv'), index=False)
            
    else:
        for idx, left_point in enumerate(matched_left):
            draw.rectangle((left_point[0] - 1, left_point[1] + 1, left_point[0] + 1, left_point[1] - 1), 'red', 'red')
            right_x = matched_right[idx][0] + 32
            right_y = matched_right[idx][1]
            draw.rectangle((right_x - 1, right_y + 1, right_x + 1, right_y - 1), 'red', 'red')

            newnew__weizhi = [tuple(left_point), (matched_right[idx][0] + 32, matched_right[idx][1])]
            draw.line(newnew__weizhi, fill='blue', width=1)
    
        new_name = nameAB + '_inlier.bmp'
        save_img_.save(os.path.join(save_t_path, new_name))
        # save_img_a.save(os.path.join(save_t_path, str('%.5d' % index_) + '_a.bmp'))
        # save_img_p.save(os.path.join(save_t_path, str('%.5d' % index_) + '_p.bmp'))
        
    contents.append([image_path[0], image_path[1], os.path.join(save_t_path, new_name), matched_num, match_after_ransac_num, model_00, model_01, model_02, model_10, model_11, model_12])
    return contents

def sample_desc_from_points(coarse_desc, pts):
    # --- Process descriptor.
    cell = 8
    H, W = coarse_desc.shape[2] * cell, coarse_desc.shape[3] * cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.to("cpu")
        desc = torch.nn.functional.grid_sample(coarse_desc.to("cpu"), samp_pts, align_corners=True)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc

def compute_dists_fast(d_a, d_b, p_a, p_b, thre, dist_mod='cos'):
    # 计算距离，如果特征被选择，最后面的标记变为对应的坐标元组
    m = d_a.size(0)
    n = d_b.size(0)
    eps = 1e-12
    if dist_mod == 'cos':
        dn_a = torch.norm(d_a, p=2, dim=1)
        dn_b = torch.norm(d_b, p=2, dim=1)

        d_a = d_a.div(torch.unsqueeze(dn_a, 1))
        d_b = d_b.div(torch.unsqueeze(dn_b, 1))

        cosine_measure = torch.mm(d_a, d_b.permute(1, 0))
        dist = torch.clamp(1 - cosine_measure, min=0)
        
    else:
        d_a_2 = torch.pow(d_a, 2).sum(1).unsqueeze(1)
        d_b_2 = torch.pow(d_b, 2).sum(1).unsqueeze(0)
        dist = d_a_2 + d_b_2
        dist = dist - 2*torch.mm(d_a,d_b.permute(1, 0)) 
    
    a2b_min_id = torch.argmin(dist, dim=1)
    len_p = len(a2b_min_id)
    ch = dist[list(range(len_p)), a2b_min_id] < thre #保留小于阈值的点

    dist_list_return = dist[list(range(len_p)), a2b_min_id].cpu().numpy()
    a2b_min_id = a2b_min_id.cpu().numpy()
    ch = ch.cpu().numpy()
    
    # selected_right是每个left对应一个的，所以当左边多个对一个的时候，会有几个selected_right元素是相同的
    # 右边匹配点重复
    have_del = []
    for idx, item in enumerate(a2b_min_id):
        if item in have_del:
            idx_before = have_del.index(item)
            if dist_list_return[idx_before] > dist_list_return[idx]:
                ch[idx_before] = False
                have_del[idx_before] = -1 #用-1占位
                have_del.append(item)
            else:
                ch[idx] = False
                have_del.append(-1)
        else:
            have_del.append(item)

    reshape_as = np.array(p_a.transpose(1, 0))
    reshape_bs = np.array(p_b.transpose(1, 0))

    final_left = reshape_as[ch]
    final_right = reshape_bs[a2b_min_id[ch]]
    dist_list_return = dist_list_return[ch]

    # # 将图a的点的最小距离,及对应点的坐标存为字典
    # key_index_dict= {}
    # for idx_a, point_a in enumerate(p_a):
    #     if a2b_min_id[idx_a] not in key_index_dict.keys():
    #         key_index_dict[a2b_min_id[idx_a]]=[]
    #     key_index_dict[a2b_min_id[idx_a]].append([point_a, dist_list_return[idx_a]])

    # final_left = []
    # final_right = []
    # dist_list_return = []
    # for select_r in list(key_index_dict.keys()):
    #     select_l_d_list = np.array(key_index_dict[select_r], dtype=object)[:, 1]
    #     min_dist_idx =np.argmin(select_l_d_list)
    #     final_left.append(tuple(np.array(key_index_dict[select_r], dtype=object)[:, 0][min_dist_idx]))
    #     final_right.append(tuple(p_b[select_r]))
    #     dist_list_return.append(np.array(key_index_dict[select_r], dtype=object)[:, 1][min_dist_idx])
    
    return final_left, final_right, dist_list_return