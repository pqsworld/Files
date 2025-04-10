import cv2
import numpy as np
from skimage.measure import ransac

def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False, shape=(128,128)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    # shape = data['prob'].shape
    print("shape: ", shape)
    real_H = data['homography']

    # Keeps only the points shared between the two views
    # keypoints = keep_shared_points(data['prob'],
    #                                real_H, keep_k_points)
    # warped_keypoints = keep_shared_points(data['warped_prob'],
    #                                       np.linalg.inv(real_H), keep_k_points)
    # keypoints = data['prob'][:,:2]
    keypoints = data['pntsA'][:,[1, 0]]
    # warped_keypoints = data['warped_prob'][:,:2]
    warped_keypoints = data['pntsB'][:,[1, 0]]
    # desc = data['desc'][keypoints[:, 0], keypoints[:, 1]]
    # warped_desc = data['warped_desc'][warped_keypoints[:, 0],
    #                                   warped_keypoints[:, 1]]
    if data['descA'] is not None:
        desc = data['descA']
        warped_desc = data['descB']

        # Match the keypoints with the warped_keypoints with nearest neighbor search
        # def get_matches():
        if orb:
            desc = desc.astype(np.uint8)
            warped_desc = warped_desc.astype(np.uint8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        print("desc: ", desc.shape)
        print("w desc: ", warped_desc.shape)
        cv2_matches = bf.match(desc, warped_desc)
        matches_idx = np.array([m.queryIdx for m in cv2_matches])
        m_keypoints = keypoints[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in cv2_matches])
        m_dist = np.array([m.distance for m in cv2_matches])
        m_warped_keypoints = warped_keypoints[matches_idx, :]
        matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
        print(f"matches: {matches.shape}")
        # get_matches()
        # from export_classical import get_sift_match
        # data = get_sift_match(sift_kps_ii=keypoints, sift_des_ii=desc, 
                # sift_kps_jj=warped_keypoints, sift_des_jj=warped_desc, if_BF_matcher=True) 
        # matches_pts = data['match_quality_good']
        # cv_matches = data['cv_matches']
        # print(f"matches: {matches_pts.shape}")
        

        # Estimate the homography between the matches using RANSAC
        H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                        m_warped_keypoints[:, [1, 0]],
                                        cv2.RANSAC)

        # H, inliers = cv2.findHomography(matches_pts[:, [1, 0]],
        #                                 matches_pts[:, [3, 2]],
        #                                 cv2.RANSAC)
                                        
        inliers = inliers.flatten()
        # print(f"cv_matches: {np.array(cv_matches).shape}, inliers: {inliers.shape}")

        print(real_H)
        print(H)
        # Compute correctness
        if H is None:
            correctness = 0
            H = np.identity(3)
            print("no valid estimation")
        else:
            corners = np.array([[0, 0, 1],
                                [0, shape[0] - 1, 1],
                                [shape[1] - 1, 0, 1],
                                [shape[1] - 1, shape[0] - 1, 1]])
            print("corner: ", corners)
            # corners = np.array([[0, 0, 1],
            #             [0, shape[1] - 1, 1],
            #             [shape[0] - 1, 0, 1],
            #             [shape[0] - 1, shape[1] - 1, 1]])
            real_warped_corners = np.dot(corners, np.transpose(real_H))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            print("real_warped_corners: ", real_warped_corners)
            
            warped_corners = np.dot(corners, np.transpose(H))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            print("warped_corners: ", warped_corners)
            
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            # correctness = float(mean_dist <= correctness_thresh)
            correctness = mean_dist <= correctness_thresh

        return {'correctness': correctness,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': matches,  # cv2.match
                'cv2_matches': cv2_matches,
                'mscores': m_dist/(m_dist.max()), # normalized distance
                'inliers': inliers,
                'homography': H,
                'mean_dist': mean_dist
                }
    else:
        return {
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': None
                }

def get_dis(p_a, p_b):
    c = 2
    eps = 1e-12
    x = np.expand_dims(p_a[:, 0], 1) - np.expand_dims(p_b[:, 0], 0)  # N 2 -> NA 1 - 1 NB -> NA NB
    y = np.expand_dims(p_a[:, 1], 1) - np.expand_dims(p_b[:, 1], 0)
    dis = np.sqrt(np.square(x) + np.square(y) + eps)
    return dis

def compute_match(data, keep_k_points=1000, correctness_thresh=3, orb=False, shape=(128,128)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    # shape = data['prob'].shape
    print("shape: ", shape)
    real_H = data['homography']

    keypoints = data['pntsA'][:,[1, 0]]  # N (y,x,s) -> N (x,y)
    warped_keypoints = data['pntsB'][:,[1, 0]]
   
    desc = data['descA']  # N C
    warped_desc = data['descB']

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    AB_dis = get_dis(keypoints,warped_keypoints)
    AB = np.matmul(desc,warped_desc.transpose())

    A_index, B_index = np.where(AB > 0.65)

    m_keypoints = keypoints[A_index, :]
    m_warped_keypoints = warped_keypoints[B_index, :]

    matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
    matches = np.concatenate([matches,np.expand_dims(AB[A_index,B_index],1)],axis=1)

    print(f"matches: {matches.shape}")

    H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv2.RANSAC)

    # H, inliers = cv2.findHomography(matches_pts[:, [1, 0]],
    #                                 matches_pts[:, [3, 2]],
    #                                 cv2.RANSAC)
                                    
    inliers = inliers.flatten()
    # print(f"cv_matches: {np.array(cv_matches).shape}, inliers: {inliers.shape}")

    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        print("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        # print("corner: ", corners)
        # corners = np.array([[0, 0, 1],
        #             [0, shape[1] - 1, 1],
        #             [shape[0] - 1, 0, 1],
        #             [shape[0] - 1, shape[1] - 1, 1]])
        real_warped_corners = np.dot(corners, np.transpose(real_H))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        # print("real_warped_corners: ", real_warped_corners)
        
        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        # print("warped_corners: ", warped_corners)
        
        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        # correctness = float(mean_dist <= correctness_thresh)
        correctness = mean_dist <= correctness_thresh
        print(real_H)
        print(H)
       
    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,  # cv2.match
            'inliers': inliers,
            'homography': H,
            'mean_dist': mean_dist
            }

