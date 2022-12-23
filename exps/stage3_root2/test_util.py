import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import numpy as np
import copy
import torch
from exps.stage3_root2.config import cfg
from lib.utils.post_3d import get_3d_points
from IPython import embed
# from lib.utils.tools import *
from lib.utils.tools import write_json

joint_to_limb_heatmap_relationship = cfg.DATASET.PAF.VECTOR
paf_z_coords_per_limb = list(range(cfg.DATASET.KEYPOINT.NUM)) # 0~14
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)  # 14


# --------------------------------------------------------------------------------------
# get crose pred_bodys
def register_pred(pred_bodys, gt_bodys, root_n=2):

    if len(pred_bodys) == 0:
        return np.asarray([])
    if gt_bodys is not None:
        """
        function:
        select the coresponding pose that has the closest distance with gt-pose;
        corres: store the index of pred-poses , len(corres) means the gt-poses order,
        so, if have 2-gt-poses and 3-pred-poses, the corres maybe [1,2], the closest pair is 0(gt)-1(pre), 1(gt)-2(pre)
        finally,generate new_pred_bodys to get refine-net train-data-set (remove the so far(wrong) predict poses)
        """
        root_gt = gt_bodys[:, root_n, :2]        # gt
        root_pd = pred_bodys[:, root_n, :2]      # pre (x,y)  the cooors of the root joint 
        distance_array = np.linalg.norm(root_gt[:, None, :] - root_pd[None, :, :], axis=2)  # --> (len(gt), len(pred))None is to insert a additional dim
        corres = np.ones(len(gt_bodys), np.int) * -1
        occupied = np.zeros(len(pred_bodys), np.int)

        while np.min(distance_array) < 30:
            min_idx = np.where(distance_array == np.min(distance_array))  # 2d: ([x1,x2,..], [y1,y2,..]); 3d has z-list
            for i in range(len(min_idx[0])):
                distance_array[min_idx[0][i]][min_idx[1][i]] = 50
                if corres[min_idx[0][i]] >= 0 or occupied[min_idx[1][i]]:
                    continue
                else:
                    corres[min_idx[0][i]] = min_idx[1][i]
                    occupied[min_idx[1][i]] = 1
        new_pred_bodys = np.zeros((len(gt_bodys), len(gt_bodys[0]), 4), np.float)
        for i in range(len(gt_bodys)):
            if corres[i] >= 0:
                new_pred_bodys[i] = pred_bodys[corres[i]]
    else:
        # run_inference
        new_pred_bodys = pred_bodys[pred_bodys[:, root_n, 3] != 0]
    return new_pred_bodys


def chain_bones(pred_bodys, depth_v, i, depth_0=0, root_n=2):
    """
    function:to get the each joint's total-relative-depth corespondence to root-joint by adding the rel-depth gradually
    here is not to add the root-depth to get each joint's actual depth, which is performed in post_3d.py get_3d_points
    """
    if root_n == 2:
        # middle of the hips is the root ..
        start_number = 2
        pred_bodys[i][2][2] = depth_0                    # root depth, set to 0
        pred_bodys[i][0][2] = pred_bodys[i][2][2] - depth_v[i][1]  # neck depth = 0 - (root - neck) 因为在编码relative depth的时候，neck和root之间是root减去neck,但是实际上应该是neck减去root,所以这里相当于取反了
    else:
        # neck is the root..
        start_number = 1
        pred_bodys[i][0][2] = depth_0
    pred_bodys[i][1][2] = pred_bodys[i][0][2] + depth_v[i][0]       # 这里就是正常的了，head本来就是相对于neck的深度，加上去就好了
    for k in range(start_number, NUM_LIMBS):
        src_k = joint_to_limb_heatmap_relationship[k][0]
        dst_k = joint_to_limb_heatmap_relationship[k][1]
        pred_bodys[i][dst_k][2] = pred_bodys[i][src_k][2] + depth_v[i][k]  # save the depth ..


def generate_relZ(pred_bodys, paf_3d_upsamp, root_d_upsamp, scale, num_intermed_pts=10, root_n=2):
    """
    pred_bodys: 2d results
    paf_3d_upsamp:relative depth maps
    root_d_upsamp:root depth maps
    scale: some information
    num_intermed_pts:sample steps to get accurate relative depth
    """    
    limb_intermed_coords = np.empty((2, num_intermed_pts), dtype=np.intp)   # relative deepth value --> (2, sample steps)
    depth_v = np.zeros((len(pred_bodys), NUM_LIMBS), dtype=np.float)        # (person num, 14)
    depth_roots_pred = np.zeros(len(pred_bodys), dtype=np.float)            # (person num)

    for i, pred_body in enumerate(pred_bodys): 
        if pred_body[root_n][3] > 0:        #score
            depth_roots_pred[i] = root_d_upsamp[int(pred_body[root_n][1]), int(pred_body[root_n][0])] * scale['scale'] * scale['f_x']
            
            # using sample steps to read-out the relativate deepth
            for k, bone in enumerate(joint_to_limb_heatmap_relationship):
                joint_src = pred_body[bone[0]]  #start joint
                joint_dst = pred_body[bone[1]]  #end joint
                if joint_dst[3] > 0 and joint_src[3] > 0: # score
                    depth_idx = paf_z_coords_per_limb[k]
                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    # sample steps are 10
                    # function: to get more accurate rel-depth value
                    limb_intermed_coords[1, :] = np.round(np.linspace(joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(joint_src[1], joint_dst[1], num=num_intermed_pts))  # Same for the y coordinate
                    intermed_paf = paf_3d_upsamp[limb_intermed_coords[0, :],
                                                 limb_intermed_coords[1, :], depth_idx]
                    min_val, max_val = np.percentile(intermed_paf, [10, 90])
                    # filter the max_val and the min_val
                    intermed_paf[intermed_paf < min_val] = min_val
                    intermed_paf[intermed_paf > max_val] = max_val
                    mean_val = np.mean(intermed_paf)
                    depth_v[i][k] = mean_val
            # embed()
            # chain_bones:是将depth按照骨架顺序加起来，得到的结果是相对于root深度值,即:root_relative_depth
            chain_bones(pred_bodys, depth_v, i, depth_0=0)

    return depth_roots_pred


def gen_3d_pose(pred_bodys, depth_necks, scale, pad_value):
    bodys = copy.deepcopy(pred_bodys)

    bodys[:, :, 0] = bodys[:, :, 0]/scale['scale'] - (scale['net_width']/scale['scale']-scale['img_width'])/2
    bodys[:, :, 1] = bodys[:, :, 1]/scale['scale'] - (scale['net_height']/scale['scale']-scale['img_height'])/2

    # bodys[:, :, 0] = (bodys[:, :, 0] - pad_value[0])/scale['scale']
    # bodys[:, :, 1] = (bodys[:, :, 1] - pad_value[1])/scale['scale'] 
    
    K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])

    bodys_3d = get_3d_points(bodys, depth_necks, K)

    # 这一步会将前面没有检测到的2D姿态但是由于计算
    # 过程而使得具有3D值的关节点数值 置零
    for i in range(bodys_3d.shape[0]):
        for j in range(bodys_3d.shape[1]):
            if bodys_3d[i, j, 3] == 0:
                bodys_3d[i, j] = 0
    return bodys_3d


# refine the coors ..
def lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, device, root_n=2):
    """
    2d pose + root-relative 3d pose --> refined root-relative 3d pose
    """
    # 
    # no_refine_list = pred_bodys_3d[:, :, 3] == 0
    # need_refine_list = []
    # for i in range(len(no_refine_list)):
    #     tmp = [j for j in range(len(no_refine_list[i])) if no_refine_list[i][j] == True]
    #     need_refine_list.append(tmp)

    root_3d_bodys = copy.deepcopy(pred_bodys_3d)     # before refine
    root_2d_bodys = copy.deepcopy(pred_bodys_2d)
    score_after_refine = np.ones([pred_bodys_3d.shape[0], pred_bodys_3d.shape[1], 1], dtype=np.float)

    input_point = np.zeros((pred_bodys_3d.shape[0], 15, 5), dtype=np.float) # each joint includes total 5 ele, x,y & X,Y,Z
    input_point[:, root_n, :2] = root_2d_bodys[:, root_n, :2]
    input_point[:, root_n, 2:] = root_3d_bodys[:, root_n, :3]

    for i in range(len(root_3d_bodys)):               # human nums
        if root_3d_bodys[i, root_n, 3] == 0:          # root score == 0
            score_after_refine[i] = 0
        for j in range(len(root_3d_bodys[0])):        # keypoint num
            if j != root_n and root_3d_bodys[i, j, 3] > 0:
                input_point[i, j, :2] = root_2d_bodys[i, j, :2] - root_2d_bodys[i, root_n, :2]
                input_point[i, j, 2:] = root_3d_bodys[i, j, :3] - root_3d_bodys[i, root_n, :3]
    
    input_point = np.resize(input_point, (input_point.shape[0], 75))   # human_nums, 15 x 5
    inp = torch.from_numpy(input_point).float().to(device)
    pred = refine_model(inp)
    if pred.device.type == 'cuda':
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()
    pred = np.resize(pred, (pred.shape[0], 15, 3))   #human_num, 15, 3 // final is only (X,Y,Z)  --> relative root : 改变一下策略，对深度值进行修正
    # pred_ori = pred
    # pred_bodys_3d[:, root_n, 2] += pred[:, root_n, 2]

    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if j != root_n: #and pred_bodys_3d[i, j, 3] == 0:
                pred[i, j] += pred_bodys_3d[i, root_n, :3]
            else:
                pred[i, j] = pred_bodys_3d[i, j, :3]
    pred = np.concatenate([pred, score_after_refine], axis=2)

    # for i in range(len(need_refine_list)):
    #     for j in range(len(need_refine_list[i])):
    #         jidx = need_refine_list[i][j]
    #         root_3d_bodys[i][jidx] = pred[i][jidx]

    return pred
    # return root_3d_bodys

# generate refine train data ..
def save_result_for_train_refine(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths,
                                 img_path_split, output_dir, root_n=2):
    
    # format:
    # result: {person_1:{'pred_3d', 'pred_2d', 'gt_3d', 'root_d'}, person_2:{...}, .....}
    for i, pred_body in enumerate(pred_bodys_3d):
        result = dict()
        img_json_name = img_path_split[0] + '--' + img_path_split[-1].split('.')[0] + f'_{i}.json'
        pair_file_name = output_dir / img_json_name
        if pred_body[root_n][3] != 0:
            result['pred_3d'] = pred_body.tolist()
            result['pred_2d'] = pred_bodys_2d[i].tolist()
            result['gt_3d'] = gt_bodys[i][:, 4:7].tolist()  # just X,Y,Z
            result['root_d'] = pred_rdepths[i]
            write_json(pair_file_name, result=result)


def save_result(pred_bodys_2d, pred_bodys_3d, gt_bodys, pred_rdepths, img_path, result):
    pair = dict()
    pair['pred_2d'] = pred_bodys_2d.tolist()
    pair['pred_3d'] = pred_bodys_3d.tolist()
    pair['root_d'] = pred_rdepths.tolist()
    pair['image_path'] = img_path
    if gt_bodys is not None:
        pair['gt_3d'] = gt_bodys[:, :, 4:].tolist()
        pair['gt_2d'] = gt_bodys[:, :, :4].tolist()
    else:
        pair['gt_3d'] = list()
        pair['gt_2d'] = list()
    result['3d_pairs'].append(pair)
