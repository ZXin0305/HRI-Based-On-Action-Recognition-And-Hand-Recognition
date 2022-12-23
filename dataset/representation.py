import cv2
import numpy as np
import math
from IPython import embed

def generate_heatmap(bodys, output_shape, stride, keypoint_num, kernel=(7, 7)):
    
    heatmaps = np.zeros((keypoint_num, *output_shape), dtype='float32')
    for i in range(keypoint_num):
        for j in range(len(bodys)):
            if bodys[j][i][3] < 1:
                continue
            joint_x = bodys[j][i][0] / stride
            joint_y = bodys[j][i][1] / stride
            # if joint_x < 0 or joint_x >= output_shape[1] or \
            #     joint_y < 0 or joint_y >= output_shape[0]:
            #     continue
            heatmaps[i, int(joint_y), int(joint_x)] = 1
        heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)  #kernel一旦确定，这个kernel size中的概率值就固定了
       
        maxi = np.amax(heatmaps[i])

        if maxi <= 1e-8:
            continue
        heatmaps[i] /= maxi / 255    #除以maxi之后，高斯核中心的那个值为1，将其转换成255， 这个中心附近的值也是在[0, 255)之间了， 便于后面的卷积

    return heatmaps

# 这个只是生成的root joint的depth vector
# 这个用vector表示，是为了能够更精确的回归出troot joint 的depth
def generate_rdepth(meta, stride, root_idx, max_people, output_shape):
    bodys = meta['bodys']
    scale = meta['scale']
    rdepth = np.zeros((max_people, 3), dtype='float32')  # (20,3)
    for j in range(len(bodys)):

        if bodys[j][root_idx, 3] < 1 or j >= max_people:
            continue
        
        rdepth[j, 0] = bodys[j][root_idx, 1] / stride   # y / stride: is to scale the coor to output size
        rdepth[j, 1] = bodys[j][root_idx, 0] / stride   # x
        rdepth[j, 2] = bodys[j][root_idx, 2] / bodys[j][root_idx, 7] / scale  # normalize by f(设定成fx,CMU) and scale 
        # embed()
    rdepth = rdepth[np.argsort(-rdepth[:, 2])]  # a root depth list
    return rdepth

def generate_rdepth_map(labels_rdepth, output_shape, sigma=1, th=0.5):
    rdepth_map = np.zeros(shape=output_shape, dtype=np.float32)
    rdepth_mask = np.zeros(shape=output_shape, dtype=np.float32)
    count = np.zeros(shape=output_shape, dtype=np.float32)

    width = output_shape[1]
    height = output_shape[0]
    delta = math.sqrt(th * 2)

    for root in labels_rdepth:
        if root[2] == 0:
            continue
        root_x , root_y = int(root[1]) , int(root[0])  #(x,y)

        x0 = int(max(0,root_x - delta * sigma + 0.5))
        y0 = int(max(0,root_y - delta * sigma + 0.5))

        x1 = int(min(width, root_x + delta * sigma + 0.5))
        y1 = int(min(height, root_y + delta * sigma + 0.5))

        if x0 > width or x1 < 0 or x1 <= x0:
            continue
        if y0 > height or y1 < 0 or y1 <= y0:
            continue

        ## fast way
        arr_heat = rdepth_map[y0:y1, x0:x1]  #　一整张图  center_map 只有一个channnel

        exp_factorx = 1 / sigma / sigma # (1/2) * (1/sigma^2)
        exp_factory = 1 / sigma / sigma
        x_vec = (np.arange(x0, x1) - root_x) ** 2
        y_vec = (np.arange(y0, y1) - root_y) ** 2
        arr_sumx = exp_factorx * x_vec
        arr_sumy = exp_factory * y_vec
        xv, yv = np.meshgrid(arr_sumx, arr_sumy)   #这一步是进行网格化

        arr_sum = xv + yv
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0    # 在一定的范围内保存深度值
        arr_exp[arr_sum <= th] = root[2]
        mask_tmp = arr_exp.copy()
        mask_tmp[arr_sum <= th] = 1
        count_tmp = mask_tmp.copy()

        rdepth_map[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
        rdepth_mask[y0:y1, x0:x1] = np.maximum(mask_tmp,rdepth_mask[y0:y1, x0:x1])
        count[y0:y1, x0:x1] += np.maximum(count_tmp, count[y0:y1, x0:x1])
        
    count[count == 0] += 1
    rdepth_map = np.divide(rdepth_map, count)
    return rdepth_map, rdepth_mask, count

def generate_paf(bodys, output_shape, params_transform, paf_num, paf_vector, paf_thre, with_mds):
    pafs = np.zeros((paf_num * 3, *output_shape), dtype='float32')
    count = np.zeros((paf_num, *output_shape), dtype='float32')
    for i in range(paf_num):
        for j in range(len(bodys)):
            if paf_thre > 1 and with_mds:
                if bodys[j][paf_vector[i][0]][3] < 2 or bodys[j][paf_vector[i][1]][3] < 2:
                    continue
            elif bodys[j][paf_vector[i][0]][3] < 1 or bodys[j][paf_vector[i][1]][3] < 1:
                continue
            centerA = np.array(bodys[j][paf_vector[i][0]][:3], dtype=int)  #with depth
            centerB = np.array(bodys[j][paf_vector[i][1]][:3], dtype=int)
            pafs[i*3:i*3+3], count[i] = putVecMaps3D(centerA, centerB, pafs[i*3:i*3+3], count[i], \
                                                     params_transform, paf_thre)
    pafs[0::3] *= 127
    pafs[1::3] *= 127

    return pafs

#是在这里对part relative depth map和pafs一起编码的
def putVecMaps3D(centerA, centerB, accumulate_vec_map, count, params_transform, thre):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    z_A = centerA[2]  #joint-A 的depth
    z_B = centerB[2]  #joint-B 的depth
    centerA = centerA[:2]  #centerA --> (x,y)
    centerB = centerB[:2]  #centerB --> (x,y)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA  # x,y
    limb_z = z_B - z_A
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:  # limb is too short, ignore it
        return accumulate_vec_map, count

    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)   #round:对数字进行舍入计算
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1)) 
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)   # to be a grid
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    vec_map[:2, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]  # x , y
    vec_map[2, yy, xx] *= limb_z                                     # z
    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) != 0, np.abs(vec_map[1, :, :]) != 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map

    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])
    count[mask == True] = 0

    return accumulate_vec_map, count

def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform, thre):
    """Implement Part Affinity Fields
    :param centerA: int with shape (2,) or (3,), centerA will pointed by centerB.
    :param centerB: int with shape (2,) or (3,), centerB will point to centerA.
    :param accumulate_vec_map: one channel of paf.
    :param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
    """

    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
    vec_map[:, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, :]) > 0, np.abs(vec_map[1, :, :]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[np.newaxis, :, :])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[np.newaxis, :, :])  #在编码Paf的时候，如果在肢体范围中，有多个人重叠或者是肢体接触了，那么存放在这里的偏移值应该除以重叠的像素
    count[mask == True] = 0

    return accumulate_vec_map, count


