import cv2
import numpy as np
from IPython import embed
import pandas as pd
import os
from matplotlib import pyplot as plt
import time
import copy
def read_csv(path):
    try:
        data = pd.read_csv(path, header=0)
    except:
        print('dataset not exist')
        return 
    return data

hand_skel_edge = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    [0, 17], [17, 18], [18, 19], [19, 20]]

hand_type = 'left'
gesture_num = '05'
num = '217'
depth_file = os.path.join("/media/xuchengjun/disk1/zx", hand_type, "HAND", gesture_num, "00/depth/", gesture_num + '_00_d_' + num + '.png')
skel_file = os.path.join("/media/xuchengjun/disk1/zx", hand_type, "HAND", gesture_num, "00/skel/", gesture_num + '_00_r_' + num + '.csv')
depth = cv2.imread(depth_file)
data = read_csv(skel_file)
# depth = depth / 255.0 * 4096.0 / 10   # 从mm 变成 cm
h, w, c = depth.shape
true_hand_center = np.array([0,0], dtype=np.float64)
max_hand_y = 0
min_hand_y = 0
max_hand_x = 0
min_hand_x = 0
colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(15,15,255), ]
# embed()
# ret, th2 = cv2.threshold(depth[:,:,0], 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
# print(ret)
# res = cv2.Canny(depth[:,:,0], 32,128)
# cv2.imshow("new", res)
# cv2.waitKey(0)

def show_map(map_, id=1):
    # map_ = map_.detach().cpu()
    map_ = np.array(map_)

    # map_ *= 255

    # show one img once a time 
    plt.subplot(111)
    plt.imshow(map_)
    plt.axis('off')
    # plt.savefig(f'/home/xuchengjun/ZXin/smap/results/hand_{count}.jpg')
    plt.show()


def depth_mapping(depth, ratio_ = 0.30, size_ = 5, H_ = 12, W_ = 5):
    """
    depth: 深度图
    ratio_: 阈值范围
    size_: 中心范围尺寸
    """
    shape = depth.shape
    max_val = 1 + ratio_
    min_val = 1 - ratio_

    img_center = [int(shape[0] / 2 + 0.5 + H_), int(shape[0] / 2 + 0.5 + W_)]
    depth = depth / 255.0 * 4096.0 / 10   # 从mm 变成 cm

    total_depth = np.sum(depth[img_center[0] - size_ : img_center[0] + size_,
                               img_center[1] - size_ : img_center[1] + size_], dtype=np.float)
    # print(f'total_depth: {total_depth:0.3f}')

    mean_depth_val = total_depth / (pow(size_ * 2, 2.0))

    depth[depth > mean_depth_val * max_val] = 0
    depth[depth < mean_depth_val * min_val] = 0
    depth[depth != 0] = 1
    cv2.imshow('depth', depth * 255)
    cv2.waitKey(0)
    return np.array(depth * 255, dtype=np.uint8)

# ---------------------------------
# res = np.zeros(shape=(h, w))
# res[1:, :] += np.abs(depth[1:, :, 0] - depth[0:h-1, :, 0])
# res[:, 1:] += np.abs(depth[:, 1:, 0] - depth[:, 0:w-1, 0])
# res[0:h-1, :] += np.abs(depth[0:h-1, :, 0] - depth[1:, :, 0])
# res[:, 0:w-1] += np.abs(depth[:, 0:w-1, 0] - depth[:, 1:, 0])
# res /= 4
# embed()
# res[res >= 0.25] = 0
# res[res <= 0.01] = 1
# res = np.array(res, dtype=np.uint8)
# depth[:,:,0] *= res
# # res *= 255.0
# depth[:,:,0] = depth[:,:,0] * 10 / 4096 * 255.0
# cv2.imshow("new", depth[:,:,0])
# cv2.waitKey(0)


# depth_mapping(depth[:,:,0])

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

def generateHandFeature(hand_numpy, hand_img_size=160, kernel = [3, 3]):
    hand_feature = np.zeros((1, hand_img_size, hand_img_size))
    for i in range(hand_numpy.shape[0] - 1):
        centerA = np.array(hand_numpy[hand_skel_edge[i][0]], dtype=int)
        centerB = np.array(hand_numpy[hand_skel_edge[i][1]], dtype=int)
        hand_feature += handSkelVis(centerA, centerB, hand_feature, 10, hand_img_size)
    
    hand_feature[hand_feature > 1] = 1
    hand_feature[0] = cv2.GaussianBlur(hand_feature[0], kernel, 0)
    show_map(hand_feature[0])
    # embed()
    # hand_feature *= 255

    return hand_feature[0]

def cal_sum_depth(hand_skel, depth):
    global true_hand_center
    global max_hand_y 
    global min_hand_y
    global max_hand_x
    global min_hand_x
    total_depth = 0
    size_ = 2
    size_1 = 2
    ratio_ = 0.1
    max_ = 1 + ratio_
    min_ = 1 - ratio_
    # expend_border = 20
    expend_borders=[8,8,8,8]  # 10, 10, 20, 20
    max_hand_y = min(160, int(np.max(hand_skel[:,1] + expend_borders[0] + 0.5)))
    min_hand_y = max(0, int(np.min(hand_skel[:,1] - expend_borders[1] + 0.5)))
    max_hand_x = min(160, int(np.max(hand_skel[:,0] + expend_borders[2] + 0.5)))
    min_hand_x = max(0, int(np.min(hand_skel[:,0] - expend_borders[3] + 0.5)))

    for i in range(21):
        hand_joint = hand_skel[i]
        x = int(hand_joint[0])
        y = int(hand_joint[1])
        total_depth += np.sum(depth[y - size_ : y + size_,
                                    x - size_ : x + size_], dtype=np.float)
    
    mean_depth_val1 = total_depth / ((pow(size_ * 2, 2.0)) * 21)
    
    # true_hand_center = (hand_skel[0] + hand_skel[2] + \
    #                    hand_skel[5] + hand_skel[9] + \
    #                    hand_skel[13] + hand_skel[17]) / 6

    list_ = [0, 2, 5, 9, 13, 17]
    for i in list_:
        true_hand_center += hand_skel[i]
    true_hand_center /= len(list_)
    print(true_hand_center)
    total_depth2 = np.sum(depth[int(true_hand_center[1]) - size_1 : int(true_hand_center[1]) + size_1,
                            int(true_hand_center[0]) - size_1 : int(true_hand_center[0]) + size_1], dtype=np.float)
    # embed()
    
    print(total_depth)
    mean_depth_val2 = total_depth2 / ((pow(size_1 * 2, 2.0)))
    mean_depth_val = 0.1 * mean_depth_val1 + 0.9 * mean_depth_val2
    print(mean_depth_val)
    # embed()
    depth[depth > max_ * mean_depth_val] = 0
    depth[depth < min_ * mean_depth_val] = 0
    # embed()
    depth[0:min_hand_y, :] = 0
    depth[:, 0:min_hand_x] = 0
    # depth[max_hand_y:, :] = 0
    depth[:, max_hand_x:] = 0
    return depth

# /media/xuchengjun/disk1/zx/left/HAND/04/00/skel/04_00_r_156.csv
# /media/xuchengjun/disk1/zx/left/HAND/00/00/skel/00_00_r_156.csv
# data = read_csv('/media/xuchengjun/disk1/zx/left/HAND/06/00/skel/06_00_r_230.csv')
# embed()
skel_numpy = np.array(data, dtype=np.float)[0:21, 1:]

st = time.time()
zero_num = len(np.where(depth[:,:,0] == 0)[0])
ratio = zero_num / (h * w)
print(f'non_zero: {zero_num}, ratio: {ratio}')
depth_new = cal_sum_depth(skel_numpy, copy.deepcopy(depth[:,:,0]))
et = time.time()
print(et - st)

# skel_img = generateHandFeature(skel_numpy).astype('uint8')

# depth[:,:,0] *= skel_img

four_coords = [(min_hand_x, min_hand_y), (min_hand_x, max_hand_y), (max_hand_x, max_hand_y), (max_hand_x, min_hand_y)]
depth_box = depth.copy()
cv2.line(depth_box, (int(four_coords[0][0]), int(four_coords[0][1])), (int(four_coords[1][0]), int(four_coords[1][1])), color=colors[4], thickness=1)   
cv2.line(depth_box, (int(four_coords[0][0]), int(four_coords[0][1])), (int(four_coords[3][0]), int(four_coords[3][1])), color=colors[4], thickness=1)   
cv2.line(depth_box, (int(four_coords[1][0]), int(four_coords[1][1])), (int(four_coords[2][0]), int(four_coords[2][1])), color=colors[4], thickness=1)   
cv2.line(depth_box, (int(four_coords[3][0]), int(four_coords[3][1])), (int(four_coords[2][0]), int(four_coords[2][1])), color=colors[4], thickness=1)  
cv2.circle(depth, center=(int(true_hand_center[0]), int(true_hand_center[1])), color=colors[2], radius=2, thickness=-1)
for i in range(21):
    cv2.circle(depth, center=(int(skel_numpy[i][0]), int(skel_numpy[i][1])), color=colors[4], radius=2, thickness=-1) 

cv2.imwrite('./circle_depth.jpg', depth)
cv2.imwrite('./circle_box.jpg', depth_box)
cv2.imwrite('./depth_new.jpg', depth_new)
cv2.imshow('depth_new', depth_new)
cv2.imshow('depth', depth)
cv2.imshow('depth_box', depth_box)
cv2.waitKey(0)

