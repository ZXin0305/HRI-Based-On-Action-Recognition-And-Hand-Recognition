import email
import cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from IPython import embed
from lib.utils.tools import show_map
from time import time

hand_skel_edge = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    [0, 17], [17, 18], [18, 19], [19, 20]]

def draw_bd_handpose_c(img_,hand_,x,y,thick=3):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)


def change_hand_skel_order(hand_):
    hand_pose_list = []
    for key in hand_.keys():
        hand_pose_list.append([hand_[key]['x'], hand_[key]['y']])
    return hand_pose_list

def handpose_track_keypoints21_pipeline(img, hands_dict, handpose_model = None, vis = False, skel_vis = False):
    """
    """
    hands_list = []

    if img is not None:
        algo_img = img.copy()
        x_min = hands_dict[0]
        y_min = hands_dict[1]
        x_max = hands_dict[2]
        y_max = hands_dict[3]
        
        w_ = max(abs(x_max-x_min),abs(y_max-y_min))
        w_ = w_*1.26

        x_mid = (x_max+x_min)/2
        y_mid = (y_max+y_min)/2

        x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)

        x1 = np.clip(x1,0,img.shape[1]-1)
        x2 = np.clip(x2,0,img.shape[1]-1)

        y1 = np.clip(y1,0,img.shape[0]-1)
        y2 = np.clip(y2,0,img.shape[0]-1)

        # 手关节点预测
        pts_ = handpose_model.predict(algo_img[y1:y2,x1:x2,:])

        plam_list = []
        pts_hand = {}
        for ptk in range(int(pts_.shape[0]/2)):
            xh = (pts_[ptk*2+0]*float(x2-x1))
            yh = (pts_[ptk*2+1]*float(y2-y1))
            
            pts_hand[str(ptk)] = {
                "x":xh,
                "y":yh,
                }
            if ptk in [0,1,5,9,13,17]:
                plam_list.append((xh+x1,yh+y1))
            if ptk == 0: #手掌根部
                hand_root_ = int(xh+x1),int(yh+y1)
            if ptk == 4: # 大拇指
                thumb_ = int(xh+x1),int(yh+y1)
            if ptk == 8: # 食指
                index_ = int(xh+x1),int(yh+y1)
            if vis:
                if ptk == 0:# 绘制腕关节点
                    cv2.circle(img, (int(xh+x1),int(yh+y1)), 9, (250,60,255),-1)
                    cv2.circle(img, (int(xh+x1),int(yh+y1)), 5, (20,180,255),-1)
                cv2.circle(img, (int(xh+x1),int(yh+y1)), 4, (255,50,60),-1)
                cv2.circle(img, (int(xh+x1),int(yh+y1)), 3, (25,160,255),-1)
        if skel_vis:
            draw_bd_handpose_c(img,pts_hand,x1,y1,2)
        '''
        shape_ = []
        shape_.append(plam_center)
        for i in range(18):
            if i in [0,5,9,13,17]:
                shape_.append((pts_hand[str(i)]["x"]+x1,pts_hand[str(i)]["y"]+y1))
        reprojectdst, euler_angle,translation_vec = get_hand_pose(np.array(shape_).reshape((len(shape_),2)),img,vis = False)
        x_,y_,z_ = translation_vec[0][0],translation_vec[1][0],translation_vec[2][0]
        cv2.putText(img, 'x,y,z:({:.1f},{:.1f},{:.1f})'.format(x_,y_,z_), (int(x_min+2),y2+19),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255,10,10),5)
        cv2.putText(img, 'x,y,z:({:.1f},{:.1f},{:.1f})'.format(x_,y_,z_), (int(x_min+2),y2+19),cv2.FONT_HERSHEY_COMPLEX, 0.45, (185, 255, 55))
        '''
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        hands_list = change_hand_skel_order(pts_hand)

        return hands_list, img

def handSkelVis(centerA, centerB, accumulate_vec_map, thre, hand_img_size):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    stride = 1
    crop_size_y = hand_img_size
    crop_size_x = hand_img_size
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA  # x,y
    limb_z = 1.0
    norm = np.linalg.norm(limb_vec)
    if norm == 0.0:
        norm = 1.0
    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)   #round:对数字进行舍入计算
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1)) 
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)   # to be a grid
    xx = xx.astype(int)
    yy = yy.astype(int)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D
    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[0, yy, xx] = np.repeat(mask[np.newaxis, :, :], 1, axis=0)
    vec_map[0, yy, xx] *= limb_z
    mask = np.logical_or.reduce((np.abs(vec_map[0, :, :]) != 0))
    
    accumulate_vec_map += vec_map
    
    return accumulate_vec_map

def generateHandFeature(hand_numpy, hand_img_size, kernel = [3, 3]):
    hand_feature = np.zeros((1, hand_img_size, hand_img_size))
    for i in range(hand_numpy.shape[0] - 1):
        centerA = np.array(hand_numpy[hand_skel_edge[i][0]], dtype=int)
        centerB = np.array(hand_numpy[hand_skel_edge[i][1]], dtype=int)
        hand_feature += handSkelVis(centerA, centerB, hand_feature, 1, hand_img_size)
    
    hand_feature[hand_feature > 1] = 1
    hand_feature[0] = cv2.GaussianBlur(hand_feature[0], kernel, 0)
    # show_map(hand_feature[0])
    # embed()
    hand_feature *= 255

    return hand_feature[0]