import numpy as np
from IPython import embed
import sys 
sys.path.append('/home/xuchengjun/ZXin/smap')
import math
import json
from lib.utils.tools import *
"""
F1_score using date format is person: [[X1,Y1,Z1],
                                       [X2,Y2,Z2],
                                       [X3,Y3,Z3] 
                                       ...
                                       ]
"""

# def change_pose(bodys, is_gt=False):
#     new_bodys = np.zeros((len(bodys), 15, 4))
#     include_together = []
#     if len(bodys) > 0:
#         for i in range(len(bodys)):
#             for j in range(15):
#                 new_bodys[i,j,0] =  j    # jtype
#                 if is_gt:
#                     new_bodys[i, j, 1:] = bodys[i, j, 4:7]
#                 else:
#                     #pre
#                     new_bodys[i, j, 1:] = bodys[i, j , :3]
#     return new_bodys

def change_pose_f1(bodys, is_gt=False):
    """
    format: ---> a total list
    [[jtype1, X1, Y1, Z1], [jtype2, X2, Y2, Z2], ...]
    """
    include_together = []
    if len(bodys) > 0:
        for body in bodys:
            for i in range(15):
                joint = []
                if not is_gt:
                    X = body[i][0]
                    Y = body[i][1]
                    Z = body[i][2]
                else:
                    X = body[i][4]
                    Y = body[i][5]
                    Z = body[i][6]                    
                joint.append(i)
                joint.append(X)
                joint.append(Y)
                joint.append(Z)
                include_together.append(joint)
    return include_together

def dist(p1, p2, th):
    """
    type: (Seq, Seq, float) -> float
    3D Point Distance
    p1:predict point
    p2:GT point
    th:the max acceptable distance
    return:euclidean distance between the positions of the two joints
    这个前面要加上关节点的类型
    """
    if p1[0] != p2[0]:
        return np.nan
    # if p1[2] == 0:
    #     return 1.0
    d = np.linalg.norm(np.array(p1[1:]) - np.array(p2[1:]))
    # print(d)
    return d if d <= th else np.nan

def non_minima_suppression(x):
    """
    return:non-minima suppressed version of the input array
    supressed values become np.nan
    """
    min = np.nanmin(x)
    x[x != min] = np.nan
    if len(x[x == min]) > 1:
        ok = True
        for i in range(len(x)):
            if x[i] == min and ok:
                ok = False
            else:
                x[i] = np.nan
    return x

def not_nan_count(x):
    """
    :return: number of not np.nan elements of the array
    返回的是一个数
    """
    return len(x[~np.isnan(x)])


def joint_det_metrics(points_pre, points_true, th=7.0):
    """
    points_pre : the predict poses in camera coordinate
    points_true: the gt-truth poses in camera coordinate
    th:distance threshold; all distances > th will be considered 'np.nan'.
    return :  a dictionary of metrics, 'met', related to joint detection;
              the the available metrics are:
              (1) met['tp'] = number of True Positives
              (2) met['fn'] = number of False Negatives
              (3) met['fp'] = number of False Positives
              (4) met['pr'] = PRecision
              (5) met['re'] = REcall
              (6) met['f1'] = F1-score
    """
    # predict, gt = change_pose(points_pre=points_pre, points_true=points_true)
    res_json_file = '/home/xuchengjun/ZXin/human_pose/results/f1_res.json'
    predict = change_pose_f1(points_pre, is_gt=False)
    gt = change_pose_f1(points_true, is_gt=True)

    if len(predict) > 0 and len(gt) > 0:
        mat = []
        for p_true in gt:
            row = np.array([dist(p_pred, p_true, th=th) for p_pred in predict]).tolist()
            mat.append(row)
        
        # write_json(path=res_json_file, result=mat)

        mat = np.array(mat)
        mat = np.apply_along_axis(non_minima_suppression, 1, mat)
        mat = np.apply_along_axis(non_minima_suppression, 0, mat)
        # calculate joint detection metrics
        nr = np.apply_along_axis(not_nan_count, 1, mat)
        tp = len(nr[nr != 0])   #number of true positives  / 预测出来并且存在于真值中
        fn = len(nr[nr == 0])   #number of false negatives / 没有预测出来
        fp = len(predict) - tp  #预测出来但是并不对
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        f1 = (2 * tp) / (2 * tp + fn + fp)
        # embed()

    elif len(predict) == 0 and len(gt) == 0:
        tp = 0    #number of true positives
        fn = 0    #number of false negatives
        fp = 0    #number of false positive
        pr = 1.0
        re = 1.0
        f1 = 1.0
    elif len(predict) == 0:
        tp = 0
        fn = len(gt)
        fp = 0
        pr = 0.0
        re = 0.0
        f1 = 0.0
    else:
        tp = 0
        fn = 0
        fp = len(predict)
        pr = 0.0
        re = 0.0
        f1 = 0.0

    metrics = {
        'tp':tp, 'fn':fn, 'fp':fp,
        'pr':pr, 're':re, 'f1':f1,
    }

    return metrics

def change_pose_mpjpe(bodys, is_gt=False):   # (person_num, 15, 4)
    """
    format: ---> a total list
    [person1:[[X1,X2, ..], [Y1, Y2, ...], [Z1, Z2, ...]], person2: ....]
    因为在前面已经把gt和pre对齐了，因此后面直接按照pre的顺序计算mpjpe
    """
    total = []

    for body in bodys:
        X = []
        Y = []
        Z = []
        for i in range(15):   # joint type num
            if not is_gt:
                X.append(body[i][0])
                Y.append(body[i][1])
                Z.append(body[i][2])
            else:
                X.append(body[i][4])
                Y.append(body[i][5])
                Z.append(body[i][6])

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        single_person = [X,Y,Z]
        total.append(single_person)
    return total

# MPJPE
def cal_mpjpe(points_pre, points_true, metrics_root):
    total_err = 0.
    no_cal_num = 0
    use_cal_list = []

    if len(points_true) == 0 or len(points_pre) == 0:
        total_err = 0.
    else:
        predict = change_pose_mpjpe(points_pre, is_gt=False)
        gt = change_pose_mpjpe(points_true, is_gt=True)

        predict = np.array(predict)
        gt = np.array(gt)
        for i in range(len(predict)):
            tmp_err = 0.

            # 未经过修正前指标计算
            joint_isvalid = predict[i][0] == 0 
            useful_joints_idx =  [i for i in range(len(joint_isvalid)) if joint_isvalid[i] == False]
            x_err = (predict[i][0][useful_joints_idx] - gt[i][0][useful_joints_idx]) ** 2
            y_err = (predict[i][1][useful_joints_idx] - gt[i][1][useful_joints_idx]) ** 2
            z_err = (predict[i][2][useful_joints_idx] - gt[i][2][useful_joints_idx]) ** 2
            for j in range(len(useful_joints_idx)):
                tmp_err += math.sqrt(x_err[j] + y_err[j] + z_err[j])
            # if tmp_err / len(useful_joints_idx) > 80:
            #     continue
            # print(tmp_err / len(useful_joints_idx))
            total_err += (tmp_err / len(useful_joints_idx)) 
        total_err /= len(predict)

        #     x_err = (predict[i][0] - gt[i][0]) ** 2
        #     y_err = (predict[i][1] - gt[i][1]) ** 2
        #     z_err = (predict[i][2] - gt[i][2]) ** 2
        #     # tmp_err = math.sqrt(np.sum(x_err) + np.sum(y_err)+ np.sum(z_err))
        #     if math.sqrt(z_err[2]) > 1000:
        #         no_cal_num += 1
        #         continue
        #     else:
        #         use_cal_list.append(i)
        #     for j in range(15):
        #         tmp_err += math.sqrt(x_err[j] + y_err[j] + z_err[j])
        #     # total_err += (tmp_err / 15)         # 除以总关节数
        #     total_err += tmp_err / 15
        # if len(predict) - no_cal_num == 0:
        #     return 0, use_cal_list
        # else:
        #     total_err /= len(predict) - no_cal_num   # 除以总人数
        #     if total_err > 1000:
        #         return 0, use_cal_list

    return total_err, use_cal_list

def cal_rootErr(pre_bodys, gt_bodys):
    err = []
    for i in range(len(pre_bodys)):
        root_pre = pre_bodys[i, 2, 2]
        root_gt = gt_bodys[i, 2, 2]
        root_err = np.abs(root_gt - root_pre)
        err.append(root_err)
    return err