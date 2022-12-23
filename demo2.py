import sys
from tkinter.messagebox import NO

from numpy import dtype
from sklearn.model_selection import cross_val_predict
sys.path.append('/home/xuchengjun/ZXin/smap')
import argparse
import os
import cv2
import numpy as np
import torch
import random
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from human_pose_msg.msg import HumanList, Human, PointCoors
from time import time
from exps.stage3_root2.test import generate_3d_point_pairs

# from model.main_model.smap import SMAP   
# from model.main_model.mode_1 import SMAP_  #with mask
from model.main_model.new_model import SMAP_new
# from model.main_model.model_tmp import SMAP_tmp as SMAP

from model.refine_model.refinenet import RefineNet
from model.action.EARN import EARN
# from model.action.EARN_v2 import EARN
# from model.action.EARN_v3 import EARN
# from model.action.vsgcnn import VSGCNN

# hand 
from model.hand.hand_skel import handpose_x_model
from lib.utils.hand_skel_function import *

import dapalib_light
import dapalib
from exps.stage3_root2.config import cfg
from path import Path
from IPython import embed
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from lib.utils.tools import *
from lib.utils.camera_wrapper import CustomDataset, ImageReader, MultiModalSubV2, VideoReader, VideoReaderWithDepth, CameraReader, CameraInfo
from lib.utils.track_pose import *
from torch.utils.data import DataLoader
import copy
from exps.stage3_root2.test_util import *
import csv 
import h5py
from lib.collect_action.collect import *
from tqdm import tqdm
import time
from torchsummary import summary
from impacket.structure import Structure

#  cpp lib
import ctypes
import math
from ctypes import *

class Result_process(ctypes.Structure):
    
    _fields_ = [
        ('cropped_rgb', ctypes.c_char_p),
        ('cropped_depth', ctypes.c_char_p)
    ]

cppLib = ctypes.cdll.LoadLibrary
lib = cppLib('./lib/CPPlibs/libcppLib.so')
# python并不会直接读取到.so的源文件，需要使用.argtypes告诉python在c函数中需要什么参数
# 这样，在后面使用c函数时pytho会自动处理你的参数，从而达到像调用python参数一样
# 下面有四个函数
lib.cdraw_rectangle.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.cdraw_rectangle_depth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.csave_rgb_depth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.ccrop_rgb_depth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]

# 同时，python也看不到函数返回什么，默认情况下,python认为函数返回了一个c中的int类型
# 如果函数返回别的类型，就需要用到retype命令
lib.cdraw_rectangle.restype = ctypes.c_void_p
# lib.ccrop_rgb_depth.restype = POINTER(Result_process)  # 读取不到缓冲区的数据
lib.cget_cropped_rgb.restype = ctypes.c_void_p
lib.cget_cropped_depth.restype = ctypes.c_void_p
lib.cdraw_rectangle_depth.restype = ctypes.c_void_p
lib.cdelete.restype = ctypes.c_void_p

def process_single_image(model, refine_model, cfg, device, img_path, angle, json_path=None):
    """
    Note: 
        the order of the pred_2d_coors is [x, y, Z, score] instead of [y, x, Z, score]
        so as the gt_data
    """
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    # data = read_json(json_path)
    # gt_bodys = np.array(data['bodys'])

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)

    img_scale, scales, pad_value = croppad_img(image, cfg)
    # scales['f_x'] = gt_bodys[0, 0, 7]
    # scales['f_y'] = gt_bodys[0, 0, 8]
    # scales['cx'] = gt_bodys[0, 0, 9]
    # scales['cy'] = gt_bodys[0, 0, 10]

    # myself
    scales['f_x'] = 1059.95
    scales['f_y'] = 1053.93
    scales['cx'] = 954.88
    scales['cy'] = 523.74
    stride = cfg.dataset.STRIDE

    # print("pad_value", pad_value)
    img_trans = transform(img_scale, cfg)
    img_trans = img_trans.unsqueeze(0).to(device)
    # time_ = 0
    with torch.no_grad():
        # st = time.time()
        outputs_2d, outputs_3d, outputs_rd = model(img_trans)
        # et = time.time()
        # time_ += et-st
        outputs_3d = outputs_3d.cpu()
        outputs_rd = outputs_rd.cpu()

        hmsIn = outputs_2d[0]

        # for i in range(hmsIn.shape[0]):
        #     print(i)
        #     show_map(copy.deepcopy(outputs_3d[0,i]), i)
        # show_map(copy.deepcopy(hmsIn[2]), 1)

        # map_ = hmsIn[2].detach().cpu()   #paf
        # map_ = outputs_3d[0,3].detach().cpu() #relative depth
        # new_root_d_upsamp = np.array(map_)
        # # new_root_d_upsamp = np.maximum(map_, 0)
        # new_root_d_upsamp /= np.max(new_root_d_upsamp)
        # new_root_d_upsamp = cv2.resize(new_root_d_upsamp, (0,0), fx=1/scales['scale'], fy=1/scales['scale'])
        # new_root_d_upsamp = cv2.resize(new_root_d_upsamp, (1920,1080), fx=1/scales['scale'], fy=1/scales['scale'])
        # # new_root_d_upsamp = cv2.resize(new_root_d_upsamp, (1920,1080))
        # # plt.matshow(new_root_d_upsamp)
        # # plt.show()
        # new_root_d_upsamp = np.uint8(100 * new_root_d_upsamp)
        # new_root_d_upsamp = cv2.applyColorMap(new_root_d_upsamp, cv2.COLORMAP_JET)
        # ori_imgs = image.astype(np.uint8)
        # add_img = cv2.addWeighted(ori_imgs, 1, new_root_d_upsamp, 0.6, 0)
        # cv2.imwrite('/home/xuchengjun/ZXin/smap/results/add_img.jpg',add_img)

        hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255   # keypoint  maps
        hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127   # paf maps
        rDepth = outputs_rd[0][0]
        
        # show_map(rDepth, 3)  #展示root的图
        # st = time.time()
        pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)  # --> tensor, shape:(person_num, 15, 4)
        # et = time.time()
        # time_ += et-st

        if pred_bodys_2d.shape[0] == 0:
            print('here is no people ..')
            # zero_pose = [0] * 45
            # return zero_pose
            return None
        if pred_bodys_2d.shape[0] >= 2:
            print('误检测 ..')
            return None
        else:
            pred_bodys_2d[:,:,:2] *= stride
            pred_bodys_2d = pred_bodys_2d.numpy()

            pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
            root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
            #　upsample the outputs' shape to obtain more accurate results
            #　--> (256, 456)
            paf_3d_upsamp = cv2.resize(
                pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
            root_d_upsamp = cv2.resize(
                root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)

            # for i in range(14):
            #     print(i)
            #     show_map(copy.deepcopy(paf_3d_upsamp[:,:,i]), i)

            pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales) #
            pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, pad_value)
        
            # new_pred_bodys_3d --> numpy()
            if refine_model is not None:
                # st = time.time()
                new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                            device=device, root_n=cfg.DATASET.ROOT_IDX)
                # et = time.time()
            else:
                new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4) 
            
            # time_ += et - st
            # print(time_)

            for i in range(new_pred_bodys_3d.shape[0]):
                new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2 

            # show_3d_results(new_pred_bodys_3d, cfg.SHOW.BODY_EADGES)
            aug_pose_3d = augment_pose(pred_3d_bodys=new_pred_bodys_3d, angles=angle) 
            norm_pose_3d = vector_pose_normalization(aug_pose_3d[0])[:,:3]  # 每一帧都要进行norm
            pose_3d = change_pose(norm_pose_3d)

            return pose_3d
            
def process_video(model, refine_model, action_model, frame_provider, cfg, device):
    delay = 1
    esc_code = 27
    p_code = 112
    mean_time = 0
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    kpt_num = cfg.DATASET.KEYPOINT.NUM

    # 姿态跟踪使用
    pose_tracker = TRACK()
    track_maxFrames = 100
    last_pose_list = [] # global
    frame = 0
    last_id = 0
    thres = 0.5
    max_human = 100
    consec_list = []

    # 动作识别使用
    non_pose_frame = 0
    pose_dict_for_action = {}
    time_step = 54  #54
    time_flag = 0
    drop_rate = 20
    time_step = 54
    # action_model = None

    time_list = []
    time_num = []
    for (img, img_trans, scales) in frame_provider:   # img have processed
        current_time = cv2.getTickCount()
        img_trans = img_trans.to(device)
        st_1 = st_2 = st_3 = 0
        et_1 = et_2 = et_3 = 0
        with torch.no_grad():
            # st_1 = time.time()
            outputs_2d, outputs_3d, outputs_rd = model(img_trans)
            # et_1 = time.time()
            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()

            hmsIn = outputs_2d[0]
            hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255 
            hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 
            rDepth = outputs_rd[0][0]
            # show_map(rDepth)
            
            # st_2 = time.time()
            pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
            # et_2 = time.time()

            if len(pred_bodys_2d) > 0:
                # print(pred_bodys_2d[:,2,:])
                pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape
                pred_bodys_2d = pred_bodys_2d.numpy()

                # ori_resoulution_bodys = recover_origin_resolution(pred_bodys_2d, scales['scale'])
                # draw_lines(img, pred_bodys_2d, cfg.SHOW.BODY_EADGES, (255,0,0))

            # ----------------------------------------------------------------------------------------
            # pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
            # root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
            #　upsample the outputs' shape to obtain more accurate results
            #　--> (256, 456)
            # paf_3d_upsamp = cv2.resize(
            #     pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
            # root_d_upsamp = cv2.resize(
            #     root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
            # new_root_d_upsamp = cv2.resize(root_d, (0,0), fx=1/scales['scale'], fy=1/scales['scale'], interpolation=cv2.INTER_NEAREST) 
            # new_root_d_upsamp = np.maximum(root_d, 0)
            # new_root_d_upsamp /= np.max(new_root_d_upsamp)
            # new_root_d_upsamp = np.uint8(255 * new_root_d_upsamp)
            # new_root_depth_map = cv2.applyColorMap(new_root_d_upsamp, cv2.COLORMAP_JET)
            # new_root_depth_map = cv2.resize(new_root_depth_map, (1920,1080), fx=1/scales['scale'], fy=1/scales['scale'], interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite('/home/xuchengjun/ZXin/smap/results/depth.jpg', new_root_depth_map) 

            # -------------------------------------------------------------------
            # 12.05
            if len(pred_bodys_2d) > 0:
                K = scales['K']
                pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
                root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
                #　upsample the outputs' shape to obtain more accurate results
                #　--> (256, 456)
                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales) #
                pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, scales['pad_value'])
                
                """
                refine
                """
                # new_pred_bodys_3d --> numpy()
                if refine_model is not None:
                    # st_3 = time.time()
                    new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                                device=device, root_n=cfg.DATASET.ROOT_IDX)
                    # et_3 = time.time()
                else:
                    new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4)
                
                # print(len(pred_bodys_2d))
                # if len(pred_bodys_2d) == 10:
                #     time_num.append((et_1-st_1)+(et_2-st_2)+(et_3-st_3))
                #     print(f"human num --> {len(pred_bodys_2d)}, time --> {(et_1-st_1)+(et_2-st_2)+(et_3-st_3)}, avg --> {np.mean(np.array(time_num))} length --> {len(time_num)}")

                # show_3d_results(new_pred_bodys_3d, cfg.SHOW.BODY_EADGES)
                # print(new_pred_bodys_3d[:,2,2])

                for i in range(new_pred_bodys_3d.shape[0]):
                    new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2
                current_frame_human = copy.deepcopy(new_pred_bodys_3d[:,:,:3])  #for image embedding

                # 姿态跟踪
                current_pose_list = []
                non_pose_frame = 0
                if frame == 0:
                    for i in range(len(current_frame_human)):
                        human = HumanPoseID(current_frame_human[i], i)  #根据顺序配置ID
                        current_pose_list.append(human)
                        last_id += 1   #第一帧的时候进行更新到最新的last_id
                    last_pose_list = current_pose_list
                    consec_list.append(current_pose_list)
                    frame += 1
                elif frame > 0:
                    for i in range(len(current_frame_human)):
                        human = HumanPoseID(current_frame_human[i], -1)
                        current_pose_list.append(human)
                    # st = time.time()
                    last_id = pose_tracker.track_pose2(consec_list, last_pose_list, current_pose_list, last_id, thres, max_human)  # 这里的last_id得传回来，python好像不能在另外的函数中改变一个值，除非这个值和函数是在一个文件中的
                    # et = time.time()
                    # last_id = track_pose(last_pose_list, current_pose_list, last_id, thres, max_human)
                    last_pose_list = current_pose_list    # update

                    
                    for i, current_pose in enumerate(current_frame_human):
                        for j in range(15):
                            current_pose[j,0] = current_pose_list[i].filter[j][0](current_pose[j,0])
                            current_pose[j,1] = current_pose_list[i].filter[j][1](current_pose[j,1])
                            current_pose[j,2] = current_pose_list[i].filter[j][2](current_pose[j,2])
                
                    #计算跟踪时间
                    # if len(pred_bodys_2d) == 8:
                    #     time_num.append(et-st)
                    #     print(f"human num --> {len(pred_bodys_2d)}, time --> {et-st}, avg --> {np.mean(np.array(time_num)):0.8f} length --> {len(time_num)}")

                    # print("length -->  ",  len(consec_list))
                    consec_list.append(current_pose_list)
                    if len(consec_list) == track_maxFrames:
                        del consec_list[0]

                    frame += 1 

                if refine_model is not None:
                    refine_pred_2d = project_to_pixel(current_frame_human, K)
                    draw_lines(img, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                    # draw_cicles(refine_pred_2d, img)
                    for i in range(len(new_pred_bodys_3d)):
                        cv2.putText(img, 'ID: {}'.format(current_pose_list[i].human_id),(int(refine_pred_2d[i][0,0]-50),int(refine_pred_2d[i][0,1]-200)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255))                        
                else:
                    refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                    draw_lines(img, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                    draw_cicles(refine_pred_2d, img) 
            else:
                # frame = 0
                # pose_dict_for_action = {}
                # consec_list = []

                non_pose_frame += 1
                if non_pose_frame > time_step / 2:  #为了防止在短时间内人突然跟踪失败，这个时候最好别立马就清空frame而重新排序
                    frame = 0
                    pose_dict_for_action = {}
                if non_pose_frame >= int(track_maxFrames/2):
                    consec_list = []

    
                # cv2.imwrite('./results/img.jpg', img)
            # else:
                # cv2.imwrite('./results/img.jpg', img)

            # time_list.append((et_1-st_1)+(et_2-st_2)+(et_3-st_3))
            # print(np.mean(np.array(time_list)))

            # -------------------------------------------------------
            
            # current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            # if mean_time == 0:
            #     mean_time = current_time
            # else:
            #     mean_time = mean_time * 0.95 + current_time * 0.05
            
            # # del hmsIn, rDepth, img_trans

            # cv2.putText(img, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
            #             (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.imshow('Human Pose Estimation', img)

            key = cv2.waitKey(delay)
            if key == esc_code:
                time_list = np.array(time_list)
                avg = np.mean(time_list)
                print(f"average time --> {avg}, num --> {np.mean(np.array(time_num))}")
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1

def run_demo(model, refine_model, action_model, frame_provider, cfg, device):
    delay = 1
    esc_code = 27
    p_code = 112
    mean_time = 0
 
    model.eval()
    if refine_model is not None:
        refine_model.eval()
    if action_model is not None:
        action_model.eval()

    kpt_num = cfg.DATASET.KEYPOINT.NUM
    HumanPub = rospy.Publisher('pub_human', HumanList, queue_size=5)
    rate = rospy.Rate(60)
    idd = 0

    # 除去不合理的人
    remove_thres = 0.5

    # 姿态跟踪使用
    pose_tracker = TRACK()
    track_maxFrames = 100
    last_pose_list = [] # global
    frame = 0
    last_id = 0
    thres = 0.5
    max_human = 10
    consec_list = []
    operator = [0]

    # 动作识别使用
    non_pose_frame = 0
    pose_dict_for_action = {}
    time_step = 54
    time_flag = 0
    drop_rate = 20
    vel_list = []  #这个用来判断动作是不是快速运动还是慢速运动,因为还有一些未定义的动作,在不同动作的过渡中
    #动作类别
    static_thres = 0.75
    static_action = [0]
    dynamic_action = [1,2,3,4]
    cal_frames = 24

    time_list = []
    action_time_list = []
    st = et = 0
    while not rospy.is_shutdown():
        for (img, img_trans, scales) in frame_provider:   # img have processed
            # current_time = cv2.getTickCount()
            # total = 0
            img_trans = img_trans.to(device)
            with torch.no_grad():
                
                # st = time.time()
                outputs_2d, outputs_3d, outputs_rd = model(img_trans)
                # et = time.time()
                # total += et -st

                # outputs_2d, outputs_3d, outputs_rd = model(img_trans)
                outputs_3d = outputs_3d.cpu()
                outputs_rd = outputs_rd.cpu()

                hmsIn = outputs_2d[0]
                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255 
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 
                rDepth = outputs_rd[0][0]

                # st1 = time.time()
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
                # et1 = time.time()
                # total += et1 -st1
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape  pred_bodys_2d --> (人数,关节点个数,(y,x,Z=0,score))
                    pred_bodys_2d = pred_bodys_2d.numpy()

                    need_remove_person_idx = []
                    #除去不合理的人
                    for i in range(pred_bodys_2d.shape[0]):
                        non_zero = pred_bodys_2d[i,:,3] != 0
                        useful_joints_idx =  [i for i in range(len(non_zero)) if non_zero[i] == True]
                        score = sum(pred_bodys_2d[i, :, 3][useful_joints_idx]) / len(useful_joints_idx)

                        if score < remove_thres:
                            need_remove_person_idx.append(i)
                    pred_bodys_2d = np.delete(pred_bodys_2d, need_remove_person_idx, axis=0)

                    ori_resoulution_bodys_2D = recover_origin_resolution(copy.deepcopy(pred_bodys_2d), scales['scale'])
                    # draw_lines(img, pred_bodys_2d, cfg.SHOW.BODY_EADGES, (255,0,0))

                # -------------------------------------------------------------------
                # 12.05

                if len(pred_bodys_2d) > 0:
                    # print('working ..')
                    K = scales['K']
                    pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  # part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
                    root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
                    #　upsample the outputs' shape to obtain more accurate results
                    #　--> (256, 456)
                    paf_3d_upsamp = cv2.resize(
                        pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
                    root_d_upsamp = cv2.resize(
                        root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
                    pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales) #
                    pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, scales['pad_value'])

                    """
                    refine
                    """
                    # new_pred_bodys_3d --> numpy()
                    if refine_model is not None:
                        # st2 = time.time()
                        new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                                    device=device, root_n=cfg.DATASET.ROOT_IDX)
                        # et2 = time.time()
                        # total += et2 -st2
                    else:
                        new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pcd num,15,4)
     
                    # 计算时间
                    # time_list.append(total)
                    # print(f"total -> {total}  avg  --> {np.mean(np.array(time_list))}")
                    # show_3d_results(new_pred_bodys_3d, cfg.SHOW.BODY_EADGES)
                    # rospy.loginfo(new_pred_bodys_3d[:,2,:])

                    for i in range(new_pred_bodys_3d.shape[0]):
                        ori_resoulution_bodys_2D[i, 0, :2] = (np.array(ori_resoulution_bodys_2D[i,3,:2]) + np.array(ori_resoulution_bodys_2D[i,9,:2])) / 2
                        new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2

                    current_frame_human = copy.deepcopy(new_pred_bodys_3d[:,:,:3])  #for image embedding
                    # world_pose = copy.deepcopy(new_pred_bodys_3d) # for world pose

                    # 姿态跟踪
                    current_pose_list = []
                    non_pose_frame = 0
                    if frame == 0:
                        #根据深度先进行简单的排序
                        root_depth_value = []
                        for i in range(len(current_frame_human)):
                            root_depth_value.append(current_frame_human[i,2,2])
                        root_depth_value = np.array(root_depth_value)
                        sort_idx = np.argsort(root_depth_value)
                        current_frame_human = current_frame_human[sort_idx]   #初始化帧时根据深度确定人员的ID

                        for i in range(len(current_frame_human)):
                            human = HumanPoseID(current_frame_human[i], i)  #根据顺序配置ID
                            current_pose_list.append(human)
                            last_id += 1   #第一帧的时候进行更新到最新的last_id
                        last_pose_list = current_pose_list
                        consec_list.append(current_pose_list)
                        frame += 1
                    elif frame > 0:
                        for i in range(len(current_frame_human)):
                            human = HumanPoseID(current_frame_human[i], -1)
                            current_pose_list.append(human)
                        # 跟踪
                        last_id = pose_tracker.track_pose2(consec_list, last_pose_list, current_pose_list, last_id, thres, max_human)
                        
                        #smooth the pose  用的是一欧元滤波
                        # for i, current_pose in enumerate(current_frame_human):
                        #     for j in range(15):
                        #         current_pose[j,0] = current_pose_list[i].filter[j][0](current_pose[j,0])
                        #         current_pose[j,1] = current_pose_list[i].filter[j][1](current_pose[j,1])
                        #         current_pose[j,2] = current_pose_list[i].filter[j][2](current_pose[j,2])
                        

                        last_pose_list = current_pose_list    # update
                        #跟踪匹配序列
                        # print(len(consec_list))
                        consec_list.append(current_pose_list)
                        if len(consec_list) >= track_maxFrames:
                            del consec_list[0]

                        frame += 1

                    # 姿态的归一化 动作识别 这里需要进行判断一下
                    # if frame == 0:
                    #     for 人数
                    #     Norm
                    #     change_pose
                    #     pose_dict_for_action[str(id)] = [总帧数， 缺的帧数， pose[]] ..
                    # else:
                    #     id_with_action = []  #顺序
                    #     for 人数：
                    #         if human id in dict_keys():
                    #             ...
                    #         else:
                    #             pose_dict_for_action[str(id)] = [总帧数， 缺的帧数， pose[]] ..

                    #动作识别
                    id_with_action = []
                    if time_flag == 0:
                        #第一帧
                        for i in range(len(current_pose_list)):
                            last_action = -1
                            human_id = current_pose_list[i].human_id
                            current_human_pose = copy.deepcopy(current_frame_human[i])  #为什么这里用的是current_frame_human,因为current_pose_list就是用current_frame_human的顺序添加的
                            
                            # norm_pose = vector_pose_normalization(copy.deepcopy(current_frame_human[i]))
                            norm_pose = pose_normalization(copy.deepcopy(current_frame_human[i]))  #(human_num[i], 15, 3) 
                            #  
                            change_pose_ = change_pose(norm_pose)
                            pose_dict_for_action[str(human_id)] = [1,[change_pose_], last_action, [current_human_pose]] # 总帧数， 规范化的数,上一帧的动作, 未规范化的数
                            id_with_action.append([human_id, -1])
                        time_flag = 1
                    elif time_flag == 1:
                        for i in range(len(current_pose_list)):  #便利所有的人
                            human_id = current_pose_list[i].human_id
                            if str(human_id) in pose_dict_for_action.keys():
                                
                                current_human_pose = copy.deepcopy(current_frame_human[i]) #这里的原因一样
                                
                                # norm_pose = vector_pose_normalization(copy.deepcopy(current_frame_human[i]))
                                norm_pose = pose_normalization(copy.deepcopy(current_frame_human[i]))

                                change_pose_ = change_pose(norm_pose)

                                pose_dict_for_action[str(human_id)][0] += 1  #存在，那么这个总帧数加1
                                pose_dict_for_action[str(human_id)][1].append(change_pose_)
                                pose_dict_for_action[str(human_id)][3].append(current_human_pose)  #记录时间段内的姿态数据
                                # if human_id == 0:
                                #     print(pose_dict_for_action[str(human_id)][0])

                                if pose_dict_for_action[str(human_id)][0] == time_step:  #等于现在的步长
                            
                                    if action_model is not None:
                                        # if human_id == 0:
                                        #     st = time.time()
                                        embed_image = get_embedding(pose_dict_for_action[str(human_id)][1])
                                        embed_image = embed_image.transpose((2, 0, 1)).astype(np.float32)
                                        embed_image = torch.from_numpy(embed_image).unsqueeze(0).to(device)  #1,3,54,15
                                        pre = action_model(embed_image)
                                        action = int(pre.argmax(1)[0])
                                        # if human_id == 0:
                                        #     et = time.time()
                                        #     action_time_list.append(et-st)
                                        #     avg = np.mean(np.array(action_time_list))
                                        #     print(f"耗时: --> {(et - st)}; 平均耗时: --> {avg}")
                                            

                                        # ------------------------------------------------------------------------------------
                                        # if action in static_action:   #先判断是不是静态的
                                        #     pass
                                        # elif action in dynamic_action: #如果是动态的
                                        #     #首先判定这个时期的动作的变化范围,因为动作可能会产生抖动...
                                        #     print(action)
                                        angle_list = cal_pose_changes(pose_dict_for_action[str(human_id)][3], cal_frames, 15)
                                        # if action == 2:
                                        #     print(pose_change_values)
                                        #     # pose_change_values = cal_pose_changes(pose_dict_for_action[str(human_id)][2], cal_frames, 15)
                                        #     print(pose_change_values)
                                        #     # numerical_index = (-1) * (pose_change_values)
                                        #     # static_score = math.exp(numerical_index)
                                        #     static_score = pose_change_values
                                        #     if static_score < 1:            #静态的动作
                                        #         action = 0
                                        #     else:
                                            # if pose_dict_for_action[str(human_id)][3] != 0:  #上一次的动作不是静态的
                                            #     action = 
                                        # ---------------------------------------------------------------------------------------
  
                                        # 简单的判断是否为inter-action   
                                        if action == 4:
                                            angle_list = np.abs(angle_list - 90)
                                            if (current_pose_list[i].human_pose[1][10,10] - current_pose_list[i].human_pose[1][11,11]) < 10 \
                                                or len(angle_list[angle_list < 30]) == 0:
                                                action = 0                                                                
                                        if action == 2:
                                            if len(angle_list[angle_list < 50]) == 0:
                                                action = 0

                                        origin_action = action

                                        # 动作触发的信号
                                        if action in static_action:
                                            pass
                                        elif action in dynamic_action:
                                            if pose_dict_for_action[str(human_id)][2] != 0:
                                                action = -1
                                        # ---------------------------------------------------------------------------------------
 
                                        if human_id in operator:
                                            # cv2.putText(img, "action:  {} ".format(" "),(100, 200), cv2.ACCESS_MASK, 4, (255, 255, 0), 4)
                                            # if pose_dict_for_action[str(human_id)][2] == 0:  #这个是动作显示的标准
                                            cv2.putText(img, "action:  {} ".format(origin_action),(100, 200), cv2.ACCESS_MASK, 4, (255, 255, 0), 4)

                                        pose_dict_for_action[str(human_id)][2] = action  #保存上一次的动作

                                        id_with_action.append([human_id, action]) 
                                    else:
                                        id_with_action.append([human_id, -1])
                                    del pose_dict_for_action[str(human_id)][1][0:drop_rate]
                                    del pose_dict_for_action[str(human_id)][3][0:drop_rate]  #姿态序列减少
                                    pose_dict_for_action[str(human_id)][0] -= drop_rate      #总帧数减少
                                else:
                                    id_with_action.append([human_id, -1])   #-1代表现在没有动作  处于中间过渡帧的时候,是没有进行动作预测的
                            else:
                                last_action = -1
                                current_human_pose = copy.deepcopy(current_frame_human[i])

                                norm_pose = vector_pose_normalization(current_frame_human[i])
                                # norm_pose = pose_normalization(current_frame_human[i])  #不存在，就重新添加


                                change_pose_ = change_pose(norm_pose)
                                pose_dict_for_action[str(human_id)] = [1,[change_pose_], last_action,[current_human_pose]]     
                                id_with_action.append([human_id, -1])           
                                                             

                    # 姿态的发布
                    human_list = HumanList()
                    # new_score = copy.deepcopy(new_pred_bodys_3d[:,:,3])[:,:,np.newaxis]
                    # current_frame_human = np.concatenate([current_frame_human,new_score], axis=2)
                    refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)  #current_frame_human :滤波后  new_pred_bodys_3d ：原始 在这里进行转化就可以了  
                    for i in range(len(current_pose_list)):   #当前的帧
                        human_id = id_with_action[i][0]
                        if human_id == -1:   #这一步是：如果人员分配的id已经超过了设置的最大人数，那么超出的人员的姿态将不会被发布出去
                            continue
                        # if human_id != 0:  #测试，是为了只看到0的姿态信息
                        #     continue
                        human = Human()
                        for j in range(15):
                            point = PointCoors()
                            point.x = current_frame_human[i][j][0]
                            point.y = current_frame_human[i][j][1]
                            point.z = current_frame_human[i][j][2]
                            human.body_points.append(point)
                        human.human_id = id_with_action[i][0]
                        human.action = id_with_action[i][1]

                        # human.human_id = 0
                        # human.action = 1
                        #转换到像素坐标进行可视化
                        # refine_pred_2d    ori_resoulution_bodys_2D
                        if frame <= 99999:
                            if human.human_id == 0:
                                # pass
                                draw_lines_once_only_one(img, refine_pred_2d[i], cfg.SHOW.BODY_EADGES, color=pose_color(human_id, max_human))
                                # draw_cicles_once_only_one(bodys=refine_pred_2d[i], image=img, color=(255,0,0))
                            elif human_id < max_human and human_id != -1: # pose_color(human.human_id, max_human)
                                # color = (25 * human_id, 25 * human_id, 25 * human_id)
                                color = pose_color(human.human_id, max_human)
                                draw_lines_once_only_one(img, refine_pred_2d[i], cfg.SHOW.BODY_EADGES, color=color)
                        
                        if not np.isnan(refine_pred_2d[i][1][1]):
                            cv2.putText(img,"id: {}".format(human.human_id),(int(refine_pred_2d[i][1][0]-50),
                                                        int(refine_pred_2d[i][1][1]-50)),
                                                        cv2.ACCESS_MASK,1,(0,0,255),2)

                        human_list.human_list.append(human)
                    HumanPub.publish(human_list)
                    # rospy.loginfo('sending pose ..')
                else:
                    # frame = 0    #判断动作的时候要加上的  
                    non_pose_frame += 1
                    # if len(consec_list) > track_maxFrames:     #没有人的时候，仍会保留前面帧的pose，万一人又回来了，就可以匹配的到
                    #     del consec_list[0]
                    if non_pose_frame > time_step / 2:  #为了防止在短时间内人突然跟踪失败，这个时候最好别立马就清空frame而重新排序
                        frame = 0
                        pose_dict_for_action = {}
                    if non_pose_frame >= int(track_maxFrames/2):
                        consec_list = []
                    # if non_pose_frame > time_step:    #当没有人的帧超过了阈值的话，那么动作列表清空，是为了防止在人不在视野中，但是后面来的人接上了
                    #     pose_dict_for_action = {}
                    # consec_list = []
                    

                
            # current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            # if mean_time == 0:
            #     mean_time = current_time
            # else:
            #     mean_time = mean_time * 0.95 + current_time * 0.05

            # del hmsIn, rDepth, img_trans
            # cv2.putText(img, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
            #             (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            # cv2.putText(img, "human_id:  {}".format(0),(100, 200), cv2.ACCESS_MASK, 4, (0,0,255),4)
            cv2.imshow('Human Pose Estimation', img)
            # cv2.imshow('human',img_copy)

            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1
            rate.sleep()

def run_demo_with_hand(model, refine_model, hand_skel_model, frame_provider, cfg, device):
    delay = 1
    esc_code = 27
    p_code = 112
    mean_time = 0
    
    model.eval()
    if refine_model is not None:
        refine_model.eval()
    
    if hand_skel_model is not None:
        hand_skel_model.model_handpose.eval()
    
    kpt_num = cfg.DATASET.KEYPOINT.NUM
    HumanPub = rospy.Publisher('pub_human', HumanList, queue_size=5)  # 姿态话题发布
    rate = rospy.Rate(100)
    idd = 0   
    
    # 除去不合理的人
    remove_thres = 0.5
    
    # 姿态跟踪使用
    pose_tracker = TRACK()
    track_maxFrames = 100
    last_pose_list = [] # global
    frame = 0
    last_id = 0
    thres = 0.5
    max_human = 10
    consec_list = []
    operator = [0, 1]
    time_step = 54
    
    # 手势识别
    non_pose_frame = 0
    time_flag = 0
    initial_rec_size = 80   # 像素值
    rec_size_range = 200
    hand_center_ratio = 10 / 4 # 边界框乘与两倍  10 / 4
    detect_thres = 1080 # 1080 * 3 / 4
    # 手势识别列表
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    hand_gesture_dict = yaml_parser('config', 'hand_gesture', cur_dir)
    
    # 手图
    resized_size = 160   # 这个记得一定要进行改动 ！！ 如果你返回的大小变化的话。。
    int_arr3 = ctypes.c_int * 3
    imgattr_para = int_arr3()
    imgattr_para[0] = resized_size   # width
    imgattr_para[1] = resized_size   # height
    imgattr_para[2] = 3              # channel
    lhand_img_name = 'left_hand_img'
    rhand_img_name = 'right_hand_img'
    lhand_depth_name = 'left_hand_depth'
    rhand_depth_name = 'right_hand_depth'

    non_useful = []
    non_csv = []

    # 提前将手部的RGB\depth还有肌电信号图放到GPU上
    # 不过也可以在做实验的时候，进行初始化
    
    # 手的检测框的点
    int_resPoints = ctypes.c_int * 8   # (x,y) * 4
    points_arr = int_resPoints()
    
    total_time = 0.0
    while not rospy.is_shutdown():
        for (img, img_trans, scales, depth, img_name, depth_name, root) in frame_provider:   # img, img_trans, scales, depth --> 单通道
            """
            self.color, net_input_image, scale, self.depth, img_name, depth_name
            img: 原图
            img_trans: 处理后送入网络中的图
            scales: 一些参数
            """

            # print(len(emg))
            img_trans = img_trans.to(device)
            with torch.no_grad():
                
                st = time.time()
                outputs_2d, outputs_3d, outputs_rd = model(img_trans)
                total_time += (time.time() - st)
                outputs_3d = outputs_3d.cpu()
                outputs_rd = outputs_rd.cpu()
                
                hmsIn = outputs_2d[0]  # 关节点热图 + 相对深度图
                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255 #处理
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 #处理
                rDepth = outputs_rd[0][0]               #根节点深度图
                
                st = time.time()
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
                total_time += (time.time() - st)
                
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # 将得到的2D坐标scale到输入的图像尺寸以读取对应的深度
                    pred_bodys_2d = pred_bodys_2d.numpy()
                    
                    # 出去不合理的人
                    need_remove_person_idx = []
                    for i in range(pred_bodys_2d.shape[0]):
                        non_zero = pred_bodys_2d[i,:,3] != 0
                        useful_joints_idx = [i for i in range(len(non_zero)) if non_zero[i] == True]  # 有的关节点被遮挡了，就不需要遮挡的置信度
                        score = sum(pred_bodys_2d[i, :, 3][useful_joints_idx]) / len(useful_joints_idx)
                        
                        if score < remove_thres:
                            need_remove_person_idx.append(i)
                    pred_bodys_2d = np.delete(pred_bodys_2d, need_remove_person_idx, axis=0)
                    
                    ori_resoulution_bodys_2D = recover_origin_resolution(copy.deepcopy(pred_bodys_2d), scales['scale'])
                
                
                if len(pred_bodys_2d) > 0:  # 除去不合理的人之后
                    K = scales['K']  # 相机的参数
                    pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  # 相对深度图
                    root_d = outputs_rd[0][0].numpy()                   # 根节点深度图
                    
                    paf_3d_upsamp = cv2.resize(
                        pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                    root_d_upsamp = cv2.resize(
                        root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                    pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales)
                    pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, scales['pad_value'])  # 3D姿态
                    
                    """
                    refine
                    """
                    if refine_model is not None:
                        st = time.time()
                        new_pred_bodys_3d = lift_and_refine_3d_pose(
                            pred_bodys_2d, pred_bodys_3d, refine_model, device=device, root_n=cfg.DATASET.ROOT_IDX)
                        total_time += (time.time() - st)
                    else:
                        new_pred_bodys_3d = pred_bodys_3d
                        
                    for i in range(new_pred_bodys_3d.shape[0]):
                        ori_resoulution_bodys_2D[i, 0, :2] = (np.array(ori_resoulution_bodys_2D[i,3,:2]) + np.array(ori_resoulution_bodys_2D[i,9,:2])) / 2
                        new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2                   
                    
                    
                    current_frame_human = copy.deepcopy(new_pred_bodys_3d[:,:,:3])  #for image embedding
                    
                    # 姿态跟踪
                    # ------------------------------------------------------------------------
                    current_pose_list = []
                    non_pose_frame = 0
                    if frame == 0:
                        # 根据深度先进行简单的排序
                        root_depth_value = []
                        for i in range(len(current_frame_human)):
                            root_depth_value.append(current_frame_human[i,2,2])  
                        root_depth_value = np.array(root_depth_value)
                        sort_idx = np.argsort(root_depth_value)
                        current_frame_human = current_frame_human[sort_idx]   #初始化帧时根据深度确定人员的ID
                        
                        for i in range(len(current_frame_human)):
                            human = HumanPoseID(current_frame_human[i], i)  #根据顺序配置ID
                            current_pose_list.append(human)
                            last_id += 1   #第一帧的时候进行更新到最新的last_id 
                            
                        last_pose_list = current_pose_list     
                        consec_list.append(current_pose_list)
                        frame += 1
                    elif frame > 0:
                        if frame >= 73:
                            embed()
                        for i in range(len(current_frame_human)):
                            human = HumanPoseID(current_frame_human[i], -1)
                            current_pose_list.append(human)   
                        # 跟踪
                        last_id = pose_tracker.track_pose2(consec_list, last_pose_list, current_pose_list, last_id, thres, max_human)
                        
                        #smooth the pose  用的是一欧元滤波
                        # for i, current_pose in enumerate(current_frame_human):
                        #     for j in range(15):
                        #         current_pose[j,0] = current_pose_list[i].filter[j][0](current_pose[j,0])
                        #         current_pose[j,1] = current_pose_list[i].filter[j][1](current_pose[j,1])
                        #         current_pose[j,2] = current_pose_list[i].filter[j][2](current_pose[j,2])

                        # for i, current_2d_pose in enumerate(ori_resoulution_bodys_2D):
                        #     for j in range(15):
                        #         current_2d_pose[j,0] = current_pose_list[i].filter_2d[j][0](current_2d_pose[j,0])
                        #         current_2d_pose[j,1] = current_pose_list[i].filter_2d[j][1](current_2d_pose[j,1])
                        
                        last_pose_list = current_pose_list    # update 
                        # 跟踪匹配序列
                        consec_list.append(current_pose_list)
                        if len(consec_list) >= track_maxFrames:
                            del consec_list[0]
                        
                        frame += 1
                    # ------------------------------------------------------------------------------
                        
                        """
                        手势识别
                        """
                        for i in range(len(current_pose_list)):

                            human_id = current_pose_list[i].human_id
                            current_human_pose = copy.deepcopy(current_frame_human[i])
                            
                            if human_id in operator:

                                # 初始化 
                                # -------------------------------------------------
                                lhand_is_detected = False
                                rhand_is_detected = False
                                cropped_lhand_img = None
                                cropped_rhand_img = None
                                cropped_lhand_depth = None
                                cropped_rhand_depth = None
                                cv2.namedWindow(lhand_img_name + str(human_id), cv2.WINDOW_NORMAL)
                                cv2.namedWindow(rhand_img_name + str(human_id), cv2.WINDOW_NORMAL)
                                cv2.namedWindow(lhand_depth_name + str(human_id), cv2.WINDOW_NORMAL)
                                cv2.namedWindow(rhand_depth_name + str(human_id), cv2.WINDOW_NORMAL)
                                # --------------------------------------------------
                                
                                
                                lelbow_y_3d = current_human_pose[4, 1]  # 左手肘的y值
                                lwrist_y_3d = current_human_pose[5, 1]  # 左手腕的y值
                                # lelbow_y_2d = ori_resoulution_bodys_2D[i, 4, 1] # 左手肘的y值(2d) 
                                
                                relbow_y_3d = current_human_pose[10, 1] # 右手肘的y值
                                rwrist_y_3d = current_human_pose[11, 1] # 右手腕的y值
                                # relbow_y_2d = ori_resoulution_bodys_2D[i, 10, 1] # 右手肘的y值(2d)
                                
                                # 判定手势框的检测
                                # 左手
                                frame_data = np.asarray(img, dtype = np.uint8)
                                frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
                                frame_data_depth = np.asarray(depth, dtype = np.uint8)
                                frame_data_depth = frame_data_depth.ctypes.data_as(ctypes.c_char_p)
                                img_shape = img.shape
                                depth_shape = depth.shape

                                """
                                left hand
                                """
                                # lelbow_y_3d > lwrist_y_3d：这个条件在制作数据集的时候不用添加
                                if ori_resoulution_bodys_2D[i, 4, 1] <= detect_thres:
                                    lwrist_z_3d = current_human_pose[5, 2]
                                    # rospy.loginfo('left wrist depth value: %d', lwrist_z_3d)
                                    lrec_size = initial_rec_size / lwrist_z_3d * rec_size_range  # 修正的框的大小
                                    
                                    # angle = (y2 - y1) / (x2 - x1)
                                    langle = math.atan2(ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1],
                                                    ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0]) * 180 / math.pi  # 转换成角度
                                    lforearm_length = math.sqrt(pow(ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1], 2.0)
                                                                +
                                                                pow(ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0], 2.0))
                                    lhand_center_x = ori_resoulution_bodys_2D[i, 5, 0] + (ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0]) / lforearm_length * (lforearm_length / hand_center_ratio)
                                    lhand_center_y = ori_resoulution_bodys_2D[i, 5, 1] + (ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1]) / lforearm_length * (lforearm_length / hand_center_ratio)
                                    
                                    # 1.分别对RGB 和 depth 进行裁剪 
                                    # ------------------------------------------------------------------------------------------------------------------------------------
                                    # # crop hand rgb
                                    # lencoded_cropped_hand_img = lib.cdraw_rectangle(img_shape[0], img_shape[1], frame_data, int(lhand_center_x), int(lhand_center_y), int(lrec_size), langle)  # RGB hand image
                                    # ltmp_rgb = ctypes.string_at(lencoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                    # lnparr = np.frombuffer(ltmp_rgb, np.uint8)
                                    # cropped_lhand_img = cv2.imdecode(lnparr, cv2.IMREAD_COLOR)

                                    # # crop hand depth
                                    # lencoded_cropped_depth_img = lib.cdraw_rectangle_depth(depth_shape[0], depth_shape[1], frame_data_depth, int(lhand_center_x), int(lhand_center_y), int(lrec_size), langle)
                                    # ltmp_depth = ctypes.string_at(lencoded_cropped_depth_img, imgattr_para[0] * imgattr_para[1])
                                    # lnparr_depth = np.frombuffer(ltmp_depth)
                                    # cropped_lhand_depth = cv2.imdecode(lnparr_depth, cv2.IMREAD_GRAYSCALE)

                                    # ------------------------------------------------------------------------------------------------------------------------------------

                                    # 2.同时对RGB 和 depth 进行裁剪
                                    # 但是会出现得到的几个参数会是NAN的情况，需要进行判断
                                    # ------------------------------------------------------------------------------------------------------------------------------------
                                    if np.isnan(lhand_center_x) or np.isnan(lhand_center_y) or np.isnan(lrec_size):
                                        print('NaN value have appear when processing left hand ...')
                                    else:
                                        lib.ccrop_rgb_depth(img_shape[0], img_shape[1], frame_data, frame_data_depth, int(lhand_center_x), int(lhand_center_y), int(lrec_size), langle)

                                        lencoded_cropped_hand_img = lib.cget_cropped_rgb()
                                        ltmp_rgb = ctypes.string_at(lencoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                        lnparr = np.frombuffer(ltmp_rgb, np.uint8)
                                        cropped_lhand_img = cv2.imdecode(lnparr, cv2.IMREAD_COLOR)

                                        lencoded_cropped_depth_img = lib.cget_cropped_depth()
                                        ltmp_depth = ctypes.string_at(lencoded_cropped_depth_img, imgattr_para[0] * imgattr_para[1])
                                        lnparr_depth = np.frombuffer(ltmp_depth)
                                        cropped_lhand_depth = cv2.imdecode(lnparr_depth, cv2.IMREAD_GRAYSCALE)
                                    #     lib.cdelete()

                                    #     # depth_sum = 0.0
                                    #     # for i in range(50, 150):
                                    #     #     for j in range(50, 150):
                                    #     #         depth_sum += cropped_lhand_depth[i,j] / 255.0 * 4096.0
                                    #     # print(f'hand depth: {depth_sum / 10000}')


                                    #     # 3.上面的是分开裁剪，试试能不能一起裁剪 -- > 返回结构体
                                    #     # image_struct = lib.ccrop_rgb_depth(img_shape[0], img_shape[1], frame_data, frame_data_depth, int(lhand_center_x), int(lhand_center_y), int(rec_size), angle)
                                    #     # tmp_rgb = ctypes.string_at(image_struct.contents.cropped_rgb, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                    #     # nparr_rgb = np.frombuffer(tmp_rgb, np.uint8)
                                    #     # cropped_lhand_img = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)  # 这里读不出来
                                    #     # lhand_img_tensor = torch.tensor(cropped_lhand_img).to('cuda:1')

                                    #     # draw hand bounding box
                                        lib.cget_rectangle_points(points_arr)
                                        # cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)
                                        lhand_is_detected = True

                                    #     if hand_skel_model is not None:
                                    #         lhand_dict = [0, 0, resized_size, resized_size]
                                    #         lhand_skel_list, cropped_lhand_img = handpose_track_keypoints21_pipeline(img=cropped_lhand_img, hands_dict=lhand_dict, \
                                    #                                                     handpose_model=hand_skel_model, 
                                    #                                                     vis=False, skel_vis=False)
                                    #         lhand_skel_list.append([lwrist_z_3d, lwrist_z_3d])

                                # relbow_y_3d > rwrist_y_3d: 这个条件在制作数据集的时候不用添加
                                if  ori_resoulution_bodys_2D[i, 11, 1] <= detect_thres:
                                    rwrist_z_3d = current_human_pose[11, 2]
                                    # rospy.loginfo('right wrist depth value: %d', rwrist_z_3d)
                                    rrec_size = initial_rec_size / rwrist_z_3d * rec_size_range  # 修正的框的大小
                                    
                                    rangle = math.atan2(ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1],
                                                    ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0]) * 180 / math.pi  # 转换成角度    
                                    
                                    rforearm_length = math.sqrt(pow(ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1], 2.0)
                                                                +
                                                                pow(ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0], 2.0))                                                            
                                    
                                    rhand_center_x = ori_resoulution_bodys_2D[i, 11, 0] + (ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0]) / rforearm_length * (rforearm_length / hand_center_ratio)
                                    rhand_center_y = ori_resoulution_bodys_2D[i, 11, 1] + (ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1]) / rforearm_length * (rforearm_length / hand_center_ratio)

                                    # 1.分别对RGB 和 depth 进行裁剪 
                                    # ------------------------------------------------------------------------------------------------------------------------------------
                                    # # crop hand rgb
                                    # rencoded_cropped_hand_img = lib.cdraw_rectangle(img_shape[0], img_shape[1], frame_data, int(rhand_center_x), int(rhand_center_y), int(rrec_size), rangle)
                                    # rtmp_rgb = ctypes.string_at(rencoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                    # rnparr = np.frombuffer(rtmp_rgb, np.uint8)
                                    # cropped_rhand_img = cv2.imdecode(rnparr, cv2.IMREAD_COLOR)
                                    # # rhand_img_tensor = torch.tensor(cropped_rhand_img).to('cuda:1')

                                    # # crop hand depth
                                    # rencoded_cropped_depth_img = lib.cdraw_rectangle_depth(depth_shape[0], depth_shape[1], frame_data_depth, int(rhand_center_x), int(rhand_center_y), int(rrec_size), rangle)
                                    # rtmp_depth = ctypes.string_at(rencoded_cropped_depth_img, imgattr_para[0] * imgattr_para[1])
                                    # rnparr_depth = np.frombuffer(rtmp_depth)
                                    # cropped_rhand_depth = cv2.imdecode(rnparr_depth, cv2.IMREAD_GRAYSCALE)
                                    # ------------------------------------------------------------------------------------------------------------------------------------

                                    # 2.同时对RGB 和 depth 进行裁剪
                                    # ------------------------------------------------------------------------------------------------------------------------------------
                                    if np.isnan(rhand_center_x) or np.isnan(rhand_center_y) or np.isnan(rrec_size):
                                        print('NaN value have appear when processing right hand ...')
                                    else:
                                        lib.ccrop_rgb_depth(img_shape[0], img_shape[1], frame_data, frame_data_depth, int(rhand_center_x), int(rhand_center_y), int(rrec_size), rangle)

                                        rencoded_cropped_hand_img = lib.cget_cropped_rgb()
                                        rtmp_rgb = ctypes.string_at(rencoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                        rnparr = np.frombuffer(rtmp_rgb, np.uint8)
                                        cropped_rhand_img = cv2.imdecode(rnparr, cv2.IMREAD_COLOR)

                                        rencoded_cropped_depth_img = lib.cget_cropped_depth()
                                        rtmp_depth = ctypes.string_at(rencoded_cropped_depth_img, imgattr_para[0] * imgattr_para[1])
                                        rnparr_depth = np.frombuffer(rtmp_depth)
                                        cropped_rhand_depth = cv2.imdecode(rnparr_depth, cv2.IMREAD_GRAYSCALE)
                                        lib.cdelete()  # 销毁堆中的内存                              

                                        # draw hand bounding box
                                        lib.cget_rectangle_points(points_arr)
                                        # cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
                                        # cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)                                
                                        rhand_is_detected = True

                                        # if hand_skel_model is not None:
                                        #     rhand_dict = [0, 0, resized_size, resized_size]
                                        #     rhand_skel_list, cropped_rhand_img = handpose_track_keypoints21_pipeline(img=cropped_rhand_img, hands_dict=rhand_dict, \
                                        #                                                 handpose_model=hand_skel_model, 
                                        #                                                 vis=False, skel_vis=False)
                                        #     rhand_skel_list.append([rwrist_z_3d, rwrist_z_3d])
                                        
                                """
                                下面的两段是用来保存裁剪后数据的
                                """
                                # 这个到后面需要改一下
                                # if lhand_is_detected and cropped_lhand_img is not None and cropped_lhand_depth is not None:
                                #     cv2.imshow(lhand_img_name + str(human_id), cropped_lhand_img)
                                #     cv2.imshow(lhand_depth_name + str(human_id), cropped_lhand_depth)
                                #     img_name_split = img_name.split('/')
                                #     depth_name_split = depth_name.split('/')
                                #     img_path = os.path.join(img_name_split[0], img_name_split[1], img_name_split[2], 
                                #                             'left', 'crop_rgb', img_name_split[-1])
                                #     depth_path = os.path.join(depth_name_split[0], depth_name_split[1], depth_name_split[2], 
                                #                             'left', 'crop_depth', depth_name_split[-1])  
                                #     skel_path = os.path.join(img_name_split[0], img_name_split[1], img_name_split[2], 
                                #                              'left', 'skel', img_name_split[-1].split('.')[0] + '.csv')                                           
                                    
                                #     if int(ori_resoulution_bodys_2D[i][5][0]) < 100 or int(ori_resoulution_bodys_2D[i][5][1]) < 100:
                                #         print('cannot draw the left line .. ')
                                #         non_useful.append(img_name_split[-1])
                                #     else:
                                #         cv2.imwrite(os.path.join(root, img_path), cropped_lhand_img)
                                #         cv2.imwrite(os.path.join(root, depth_path), cropped_lhand_depth)
                                #         lhand_skel_numpy = np.array(lhand_skel_list)
                                #         pd.DataFrame(lhand_skel_numpy).to_csv(os.path.join(root, skel_path))

                                                        
                                # if rhand_is_detected and cropped_rhand_img is not None and cropped_rhand_depth is not None:
                                #     cv2.imshow(rhand_img_name + str(human_id), cropped_rhand_img)
                                #     cv2.imshow(rhand_depth_name + str(human_id), cropped_rhand_depth)
                                #     img_name_split = img_name.split('/')
                                #     depth_name_split = depth_name.split('/')
                                #     img_path = os.path.join(img_name_split[0], img_name_split[1], img_name_split[2], 
                                #                             'right', 'crop_rgb', img_name_split[-1])
                                #     depth_path = os.path.join(depth_name_split[0], depth_name_split[1], depth_name_split[2], 
                                #                             'right', 'crop_depth', depth_name_split[-1])
                                #     skel_path = os.path.join(img_name_split[0], img_name_split[1], img_name_split[2], 
                                #                             'right', 'skel', img_name_split[-1].split('.')[0] + '.csv')                                            
                                    
                                #     if int(ori_resoulution_bodys_2D[i][11][0]) < 100 or int(ori_resoulution_bodys_2D[i][11][1]) < 100:
                                #         print('cannot draw the right line .. ')
                                #         non_useful.append(img_name_split[-1])
                                #     else:
                                #         cv2.imwrite(os.path.join(root, img_path), cropped_rhand_img)
                                #         cv2.imwrite(os.path.join(root, depth_path), cropped_rhand_depth)
                                #         rhand_skel_numpy = np.array(rhand_skel_list)
                                #         pd.DataFrame(rhand_skel_numpy).to_csv(os.path.join(root, skel_path))
                       
                        
                    id_with_action = [[0, 0], [1, 1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]    
                        
                    # 姿态的发布
                    human_list = HumanList()
                    # new_score = copy.deepcopy(new_pred_bodys_3d[:,:,3])[:,:,np.newaxis]
                    # current_frame_human = np.concatenate([current_frame_human,new_score], axis=2)
                    refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)  #current_frame_human :滤波后  new_pred_bodys_3d ：原始 在这里进行转化就可以了 
                    for i in range(len(current_pose_list)):   #当前的帧
                        # human_id = id_with_action[i][0]
                        human_id = current_pose_list[i].human_id
                        if human_id == -1:   #这一步是：如果人员分配的id已经超过了设置的最大人数，那么超出的人员的姿态将不会被发布出去
                            continue
                        # if human_id != 0:  #测试，是为了只看到0的姿态信息
                        #     continue
                        human = Human()
                        for j in range(15):
                            point = PointCoors()
                            point.x = current_frame_human[i][j][0]
                            point.y = current_frame_human[i][j][1]
                            point.z = current_frame_human[i][j][2]
                            human.body_points.append(point)
                            
                        human.human_id = human_id
                        human.action = 0
                        
                        # 转换到像素坐标进行可视化
                        # refine_pred_2d    ori_resoulution_bodys_2D
                        if frame <= 99999:
                            if human.human_id in operator:
                                color = pose_color(human_id, max_human)
                                # draw_lines_once_only_one(img, ori_resoulution_bodys_2D[i], cfg.SHOW.BODY_EADGES, color=pose_color(human_id, max_human))
                                draw_cicles_once_only_one(bodys=ori_resoulution_bodys_2D[i], image=img, color=color)
                                if not np.isnan(refine_pred_2d[i][1][1]):
                                    cv2.putText(img,"id: {}".format(human.human_id),(int(refine_pred_2d[i][1][0]-50),
                                                                int(refine_pred_2d[i][1][1]-50)),
                                                                cv2.ACCESS_MASK,1,color,2)
                                # cv2.line(img, (int(ori_resoulution_bodys_2D[i][4][0]), int(ori_resoulution_bodys_2D[i][4][1])), (int(ori_resoulution_bodys_2D[i][5][0]), int(ori_resoulution_bodys_2D[i][5][1])), color=(0,255,0), thickness=2)
                                # cv2.line(img, (int(ori_resoulution_bodys_2D[i][10][0]), int(ori_resoulution_bodys_2D[i][10][1])), (int(ori_resoulution_bodys_2D[i][11][0]), int(ori_resoulution_bodys_2D[i][11][1])), color=(0,255,0), thickness=2)
                            elif human_id < max_human and human_id != -1: # pose_color(human.human_id, max_human)
                                # color = (25 * human_id, 25 * human_id, 25 * human_id)
                                color = pose_color(human.human_id, max_human)
                                # draw_lines_once_only_one(img, ori_resoulution_bodys_2D[i], cfg.SHOW.BODY_EADGES, color=color)
                        
                                # if not np.isnan(refine_pred_2d[i][1][1]):
                                #     cv2.putText(img,"id: {}".format(human.human_id),(int(refine_pred_2d[i][1][0]-50),
                                #                                 int(refine_pred_2d[i][1][1]-50)),
                                #                                 cv2.ACCESS_MASK,color,(0,0,255),2)
                            
                    #     human_list.human_list.append(human)
                    # HumanPub.publish(human_list)
                    # rospy.loginfo('sending pose ..')
                else:                                                 
                    non_pose_frame += 1
                    if non_pose_frame > time_step / 2:  # 为了防止在短时间内人突然跟踪失败，这个时候最好别立马就清空frame而重新排序
                        frame = 0
                        pose_dict_for_action = {}  
                    if non_pose_frame >= int(track_maxFrames / 2):
                        consec_list = []
            # cv2.imwrite('/media/xuchengjun/disk1/zx/dataset/tmp/demo_for_paper/'+str(frame)+'.jpg', img)            
            # current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            current_time = total_time
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            # print(mean_time)
            total_time = 0.0

            del hmsIn, rDepth, img_trans
            cv2.putText(img, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            # cv2.putText(img, "human_id:  {}".format(0),(100, 200), cv2.ACCESS_MASK, 4, (0,0,255),4)
            cv2.imshow('Human Pose Estimation', img)
            
            # cv2.imshow('human',img_copy)
            
            key = cv2.waitKey(delay)
            if key == esc_code:
                break
            if key == p_code:
                if delay == 1:
                    delay = 0
                else:
                    delay = 1 
            rate.sleep()
        print(non_useful)    # 数据坏的
                         
def process_rgb_depth(model, refine_model, frame_provider, cfg, device):
    delay = 1
    esc_code = 27
    p_code = 112
    mean_time = 0
    
    model.eval()
    if refine_model is not None:
        refine_model.eval()
    
    kpt_num = cfg.DATASET.KEYPOINT.NUM
    idd = 0   
    
    # 除去不合理的人
    remove_thres = 0.5
    
    # 姿态跟踪使用
    pose_tracker = TRACK()
    track_maxFrames = 100
    last_pose_list = [] # global
    frame = 0
    last_id = 0
    thres = 0.5
    max_human = 10
    consec_list = []
    operator = [0]
    time_step = 54
    
    # 手势识别
    non_pose_frame = 0
    time_flag = 0
    initial_rec_size = 80  # 像素值
    rec_size_range = 200
    hand_center_ratio = 2
    
    # 手图
    resized_size = 244
    int_arr3 = ctypes.c_int * 3
    imgattr_para = int_arr3()
    imgattr_para[0] = resized_size   # width
    imgattr_para[1] = resized_size   # height
    imgattr_para[2] = 3              # channel
    lhand_img_name = 'left_hand_img'
    rhand_img_name = 'right_hand_img'
    
    # 手的检测框的点
    int_resPoints = ctypes.c_int * 8
    points_arr = int_resPoints()
    
    action_time_list = []
    total_time = 0.0
    for (img, depth, img_trans, scales) in frame_provider: 
        """
        img: 原图
        img_trans: 处理后送入网络中的图
        scales: 一些参数
        """
        
        img_trans = img_trans.to(device)
        with torch.no_grad():
            
            st = time.time()
            outputs_2d, outputs_3d, outputs_rd = model(img_trans)
            total_time += (time.time() - st)
            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()
            
            hmsIn = outputs_2d[0]  # 关节点热图 + 相对深度图
            hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255 #处理
            hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 #处理
            rDepth = outputs_rd[0][0]               #根节点深度图
            
            st = time.time()
            pred_bodys_2d = dapalib_light.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
            total_time += (time.time() - st)
            
            if len(pred_bodys_2d) > 0:
                pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # 将得到的2D坐标scale到输入的图像尺寸以读取对应的深度
                pred_bodys_2d = pred_bodys_2d.numpy()
                
                # 出去不合理的人
                need_remove_person_idx = []
                for i in range(pred_bodys_2d.shape[0]):
                    non_zero = pred_bodys_2d[i,:,3] != 0
                    useful_joints_idx = [i for i in range(len(non_zero)) if non_zero[i] == True]  # 有的关节点被遮挡了，就不需要遮挡的置信度
                    score = sum(pred_bodys_2d[i, :, 3][useful_joints_idx]) / len(useful_joints_idx)
                    
                    if score < remove_thres:
                        need_remove_person_idx.append(i)
                pred_bodys_2d = np.delete(pred_bodys_2d, need_remove_person_idx, axis=0)
                
                ori_resoulution_bodys_2D = recover_origin_resolution(copy.deepcopy(pred_bodys_2d), scales['scale'])
            
            
            if len(pred_bodys_2d) > 0:  # 除去不合理的人之后
                K = scales['K']  # 相机的参数
                pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  # 相对深度图
                root_d = outputs_rd[0][0].numpy()                   # 根节点深度图
                
                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales)
                pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, scales['pad_value'])  # 3D姿态
                
                """
                refine
                """
                if refine_model is not None:
                    st = time.time()
                    new_pred_bodys_3d = lift_and_refine_3d_pose(
                        pred_bodys_2d, pred_bodys_3d, refine_model, device=device, root_n=cfg.DATASET.ROOT_IDX)
                    total_time += (time.time() - st)
                else:
                    new_pred_bodys_3d = pred_bodys_3d
                    
                for i in range(new_pred_bodys_3d.shape[0]):
                    ori_resoulution_bodys_2D[i, 0, :2] = (np.array(ori_resoulution_bodys_2D[i,3,:2]) + np.array(ori_resoulution_bodys_2D[i,9,:2])) / 2
                    new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2                   
                
                
                current_frame_human = copy.deepcopy(new_pred_bodys_3d[:,:,:3])  #for image embedding
                
                # 姿态跟踪
                current_pose_list = []
                non_pose_frame = 0
                if frame == 0:
                    # 根据深度先进行简单的排序
                    root_depth_value = []
                    for i in range(len(current_frame_human)):
                        root_depth_value.append(current_frame_human[i,2,2])  
                    root_depth_value = np.array(root_depth_value)
                    sort_idx = np.argsort(root_depth_value)
                    current_frame_human = current_frame_human[sort_idx]   #初始化帧时根据深度确定人员的ID
                    
                    for i in range(len(current_frame_human)):
                        human = HumanPoseID(current_frame_human[i], i)  #根据顺序配置ID
                        current_pose_list.append(human)
                        last_id += 1   #第一帧的时候进行更新到最新的last_id 
                        
                    last_pose_list = current_pose_list     
                    consec_list.append(current_pose_list)
                    frame += 1
                elif frame > 0:
                    for i in range(len(current_frame_human)):
                        human = HumanPoseID(current_frame_human[i], -1)
                        current_pose_list.append(human)   
                    # 跟踪
                    last_id = pose_tracker.track_pose2(consec_list, last_pose_list, current_pose_list, last_id, thres, max_human)
                    
                    # smooth the 3d pose  用的是一欧元滤波
                    # for i, current_pose in enumerate(current_frame_human):
                    #     for j in range(15):
                    #         current_pose[j,0] = current_pose_list[i].filter[j][0](current_pose[j,0])
                    #         current_pose[j,1] = current_pose_list[i].filter[j][1](current_pose[j,1])
                    #         current_pose[j,2] = current_pose_list[i].filter[j][2](current_pose[j,2])

                    # smooth the 2d pose
                    # for i, current_2d_pose in enumerate(ori_resoulution_bodys_2D):
                    #     for j in range(15):
                    #         current_2d_pose[j,0] = current_pose_list[i].filter_2d[j][0](current_2d_pose[j,0])
                    #         current_2d_pose[j,1] = current_pose_list[i].filter_2d[j][1](current_2d_pose[j,1])
                    
                    last_pose_list = current_pose_list    # update 
                    # 跟踪匹配序列
                    consec_list.append(current_pose_list)
                    if len(consec_list) >= track_maxFrames:
                        del consec_list[0]
                    
                    frame += 1
                    
                    """
                    手势识别
                    """
                    for i in range(len(current_pose_list)):
                        lhand_is_detected = False
                        rhand_is_detected = False
                        cropped_lhand_img = None
                        cropped_rhand_img = None
                        # cv2.namedWindow(lhand_img_name, cv2.WINDOW_NORMAL)
                        # cv2.namedWindow(rhand_img_name, cv2.WINDOW_NORMAL)
                        # lhand_points = []
                        # rhand_points = []
                        
                        human_id = current_pose_list[i].human_id
                        current_human_pose = copy.deepcopy(current_frame_human[i])
                        
                        lelbow_y_3d = current_human_pose[4, 1]  # 左手肘的y值
                        lwrist_y_3d = current_human_pose[5, 1]  # 左手腕的y值
                        # lelbow_y_2d = ori_resoulution_bodys_2D[i, 4, 1] # 左手肘的y值(2d) 
                        
                        relbow_y_3d = current_human_pose[10, 1] # 右手肘的y值
                        rwrist_y_3d = current_human_pose[11, 1] # 右手腕的y值
                        # relbow_y_2d = ori_resoulution_bodys_2D[i, 10, 1] # 右手肘的y值(2d)
                        
                        # 判定手势框的检测
                        # 左手
                        frame_data = np.asarray(img, dtype = np.uint8)
                        frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)

                        depth_data = np.asarray(depth, dtype = np.float32)
                        depth_data = depth_data.ctypes.data_as(ctypes.c_char_p)

                        # embed()

                        img_shape = img.shape
                        if human_id == 0:
                            if lelbow_y_3d > lwrist_y_3d and ori_resoulution_bodys_2D[i, 4, 1] <= (1080 * 1 / 2):
                                lwrist_z_3d = current_human_pose[5, 2]
                                # print('left wrist depth value: ', lwrist_z_3d)
                                rec_size = initial_rec_size / lwrist_z_3d * rec_size_range  # 修正的框的大小
                                
                                # angle = (y2 - y1) / (x2 - x1)
                                angle = math.atan2(ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1],
                                                ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0]) * 180 / math.pi  # 转换成角度
                                lforearm_length = math.sqrt(pow(ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1], 2.0)
                                                            +
                                                            pow(ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0], 2.0))
                                lhand_center_x = ori_resoulution_bodys_2D[i, 5, 0] + (ori_resoulution_bodys_2D[i, 5, 0] - ori_resoulution_bodys_2D[i, 4, 0]) / lforearm_length * (lforearm_length / hand_center_ratio)
                                lhand_center_y = ori_resoulution_bodys_2D[i, 5, 1] + (ori_resoulution_bodys_2D[i, 5, 1] - ori_resoulution_bodys_2D[i, 4, 1]) / lforearm_length * (lforearm_length / hand_center_ratio)
                                
                                # encoded_cropped_hand_img = lib.cdraw_rectangle(img_shape[0], img_shape[1], frame_data, int(lhand_center_x), int(lhand_center_y), int(rec_size), angle)
                                # tmp = ctypes.string_at(encoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                # nparr = np.frombuffer(tmp, np.uint8)
                                # cropped_lhand_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                # lib.cget_rectangle_points(points_arr)
                                # lhand_points = list(points_arr)

                                # lib.csave_rgb_depth(img_shape[0], img_shape[1], frame_data, depth_data, int(lhand_center_x), int(lhand_center_y), int(rec_size), angle)                               
                                # draw hand bounding box
                                # cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
                                # cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
                                # cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
                                # cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)

                                lhand_is_detected = True

                            if relbow_y_3d > rwrist_y_3d and ori_resoulution_bodys_2D[i, 11, 1] <= (1080 * 1 / 2):
                                rwrist_z_3d = current_human_pose[11, 2]
                                # print('right wrist depth value: ', rwrist_z_3d)
                                rec_size = initial_rec_size / rwrist_z_3d * rec_size_range  # 修正的框的大小
                                
                                angle = math.atan2(ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1],
                                                ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0]) * 180 / math.pi  # 转换成角度    
                                
                                rforearm_length = math.sqrt(pow(ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1], 2.0)
                                                            +
                                                            pow(ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0], 2.0))                                                            
                                
                                rhand_center_x = ori_resoulution_bodys_2D[i, 11, 0] + (ori_resoulution_bodys_2D[i, 11, 0] - ori_resoulution_bodys_2D[i, 10, 0]) / rforearm_length * (rforearm_length / hand_center_ratio)
                                rhand_center_y = ori_resoulution_bodys_2D[i, 11, 1] + (ori_resoulution_bodys_2D[i, 11, 1] - ori_resoulution_bodys_2D[i, 10, 1]) / rforearm_length * (rforearm_length / hand_center_ratio)
                                
                                # encoded_cropped_hand_img = lib.cdraw_rectangle(img_shape[0], img_shape[1], frame_data, int(rhand_center_x), int(rhand_center_y), int(rec_size), angle)
                                # tmp = ctypes.string_at(encoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
                                # nparr = np.frombuffer(tmp, np.uint8)
                                # cropped_rhand_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                lib.cget_rectangle_points(points_arr)
                                rhand_points = list(points_arr) 
                                
                                lib.csave_rgb_depth(img_shape[0], img_shape[1], frame_data, depth_data, int(rhand_center_x), int(rhand_center_y), int(rec_size), angle)
                                # draw hand bounding box
                                cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
                                cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
                                cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
                                cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)                                
                                rhand_is_detected = True
                        
                        """
                        (1) img transform  --> to tensor
                        (2) img to cuda
                        (3) feed into network
                        (4) predict label
                        """
                        # if lhand_is_detected:
                        #     cv2.imshow(lhand_img_name, cropped_lhand_img)
                                                
                        # if rhand_is_detected:
                        #     cv2.imshow(rhand_img_name, cropped_rhand_img)
                                            
                    
                id_with_action = [[0, 0], [1, 1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]    
                    
                # 姿态的发布
                human_list = HumanList()
                # new_score = copy.deepcopy(new_pred_bodys_3d[:,:,3])[:,:,np.newaxis]
                # current_frame_human = np.concatenate([current_frame_human,new_score], axis=2)
                refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)  #current_frame_human :滤波后  new_pred_bodys_3d ：原始 在这里进行转化就可以了 
                for i in range(len(current_pose_list)):   #当前的帧
                    # human_id = id_with_action[i][0]
                    human_id = current_pose_list[i].human_id
                    if human_id == -1:   #这一步是：如果人员分配的id已经超过了设置的最大人数，那么超出的人员的姿态将不会被发布出去
                        continue
                    # if human_id != 0:  #测试，是为了只看到0的姿态信息
                    #     continue
                    
                    # 转换到像素坐标进行可视化
                    # refine_pred_2d    ori_resoulution_bodys_2D
                    if frame <= 99999:
                        if human.human_id == 0:
                            draw_lines_once_only_one(img, ori_resoulution_bodys_2D[i], cfg.SHOW.BODY_EADGES, color=pose_color(human_id, max_human))
                            # draw_cicles_once_only_one(bodys=refine_pred_2d[i], image=img, color=(255,0,0))
                        elif human_id < max_human and human_id != -1: # pose_color(human.human_id, max_human)
                            # color = (25 * human_id, 25 * human_id, 25 * human_id)
                            color = pose_color(human.human_id, max_human)
                            draw_lines_once_only_one(img, ori_resoulution_bodys_2D[i], cfg.SHOW.BODY_EADGES, color=color)
                    
                    if not np.isnan(refine_pred_2d[i][1][1]):
                        cv2.putText(img,"id: {}".format(human.human_id),(int(refine_pred_2d[i][1][0]-50),
                                                    int(refine_pred_2d[i][1][1]-50)),
                                                    cv2.ACCESS_MASK,1,(0,0,255),2)
            else:                                                 
                non_pose_frame += 1
                if non_pose_frame > time_step / 2:  # 为了防止在短时间内人突然跟踪失败，这个时候最好别立马就清空frame而重新排序
                    frame = 0
                    pose_dict_for_action = {}  
                if non_pose_frame >= int(track_maxFrames / 2):
                    consec_list = []
                    
        # current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        current_time = total_time
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        # print(mean_time)
        total_time = 0.0

        del hmsIn, rDepth, img_trans
        cv2.putText(img, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        # cv2.putText(img, "human_id:  {}".format(0),(100, 200), cv2.ACCESS_MASK, 4, (0,0,255),4)
        cv2.imshow('Human Pose Estimation', img)
        # cv2.imshow('depth', depth)
        
        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1 

def main():
    parser = argparse.ArgumentParser()
    # /home/xuchengjun/ZXin/smap/pretrained/main_model.pth
    # /media/xuchengjun/zx/human_pose/pth/main/1.4/train.pth
    # /media/xuchengjun/zx/human_pose/pth/main/12.16/train.pth
    # /media/xuchengjun/zx/human_pose/pth/main/20220328/train.pth
    parser.add_argument('--SMAP_path', type=str, 
                                       default='/home/xuchengjun/ZXin/smap/pretrained/main_model.pth')
    # /media/xuchengjun/zx/human_pose/pth/main/1.4/RefineNet_epoch_250.pth
    # /home/xuchengjun/ZXin/smap/pretrained/refine.pth
    parser.add_argument('--RefineNet_path', type=str, 
                                       default='/home/xuchengjun/ZXin/smap/pretrained/refine.pth') 
    parser.add_argument('--Action_path', type=str, default='/home/xuchengjun/ZXin/smap/pretrained/.pth')  # /home/xuchengjun/ZXin/smap/pretrained/action.pth  train_previous  trained_model
    parser.add_argument('--Hand_skel_path', type=str, default='/home/xuchengjun/ZXin/smap/pretrained/ReXNetV1-size-256-wingloss102-0.122.pth')
    parser.add_argument('--device',"-de", type=int, default=0) 
    # donot use mode 1 ..
    parser.add_argument('--mode', '-m', type=int, default=0, 
                        help='mode --> 0: process single iamge, 1:using CustomDataset, 2: process video, 3: real-time process')
    parser.add_argument('--internal_matrix','-i',type=int,default=1,help='是否使用自己的相机内参')
    
    # stage_root test.py  
    parser.add_argument("--test_mode", "-t", type=str, default="run_inference",
                    choices=['generate_train', 'generate_result', 'run_inference'],
                    help='Type of test. One of "generate_train": generate refineNet datasets, '
                            '"generate_result": save inference result and groundtruth, '
                            '"run_inference": save inference result for input images.')
    parser.add_argument("--data_mode", "-d", type=str, default="test",
                        choices=['test', 'generation'],
                        help='Only used for "generate_train" test_mode, "generation" for refineNet train dataset,'
                             '"test" for refineNet test dataset.')
    parser.add_argument("--batch_size", type=int, default=1,help='Batch_size of test')
    parser.add_argument("--do_flip", type=float, default=0,help='Set to 1 if do flip when test')

    # process img dir
    # /media/xuchengjun/datasets/CMU/170407_haggling_a1/hdImgs/00_16
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample
    # /media/xuchengjun/datasets/action/images/stand
    # /media/xuchengjun/datasets/coco_2017
    parser.add_argument("--dataset_path", type=str, default="/media/xuchengjun/datasets/action/images/walk",
                        help='Image dir path of "run_inference" test mode')
    
    # process video
    # /media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdVideos/hd_00_00.mp4
    # /media/xuchengjun/datasets/panoptic-toolbox/160422_ultimatum1/hdVideos
    # /media/xuchengjun/datasets/panoptic-toolbox/160906_pizza1/hdVideos/hd_00_00.mp4
    # /media/xuchengjun/datasets/panoptic-toolbox/161029_sports1/hdVideos
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample/hdVideos/hd_00_00.mp4
    # /media/xuchengjun/datasets/action/video/walk.avi
    # /home/xuchengjun/Videos/识别1.mp4
    # /home/xuchengjun/Videos/test.mp4  output.mp4
    # /media/xuchengjun/datasets/panoptic-toolbox/161029_sports1/hdVideos/hd_00_16.mp4
    # /media/xuchengjun/zx/videos/01.avi
    # /media/xuchengjun/datasets/JTA-Dataset/jta_dataset/videos/train
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1/hdVideos
    parser.add_argument('--video_path', type=str, default='/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/rgb_video.avi')
    parser.add_argument('--video_depth', type=str, default='/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/depth_video.avi')
    parser.add_argument("--json_name", type=str, default="final_json",
                        help='Add a suffix to the result json.')

    # process single img
    # 160422_ultimatum1--00_02--00014176.json
    # 160906_pizza1
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample/hdImgs/00_00/00_00_00000000.jpg
    # /media/xuchengjun/datasets/panoptic-toolbox/160422_ultimatum1/hdImgs/00_02/00_02_00014176.jpg
    # /media/xuchengjun/datasets/CMU/170407_haggling_a1/hdImgs/00_30/00_30_00003500.jpg
    # /media/xuchengjun/datasets/CMU/160906_pizza1/hdImgs/00_30/00_30_00003500.jpg
    # /media/xuchengjun/datasets/action/images/stand/0.jpg
    # /home/xuchengjun/ZXin/SMAP-master/results/current.jpg
    # /media/xuchengjun/datasets/CMU/160422_ultimatum1/hdImgs/00_08/00_08_00015350.jpg
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample/hdImgs/00_00
    # 161029_sports1
    parser.add_argument('--json_path', default="/media/xuchengjun/disk/datasets/CMU/CMU_val_json_file/160906_pizza1/160906_pizza1--00_16--00000200.json")
    parser.add_argument('--img_path', default="/media/xuchengjun/disk/datasets/panoptic-toolbox/160422_ultimatum1/hdImgs/00_08/00_08_00015350.jpg")
    
    # process real camera  kinectSDK/color  /kinect2_1/hd/image_color
    parser.add_argument('--camera_topic', default='kinectSDK/color')


    parser.add_argument('--gesture_num', default="00")
    
    args = parser.parse_args()

    # set device 
    device = torch.device("cuda:"+ str(args.device))

    # load main model
    # model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
    # model = SMAP_(cfg, run_efficient=cfg.RUN_EFFICIENT)  # mode_1 原来的带有root mask
    model = SMAP_new(cfg, run_efficient=cfg.RUN_EFFICIENT)
    model.to(device)
    # load refine model
    refine_model = RefineNet()
    refine_model.to(device)
    # load action model
    action_model = EARN(depth=28, num_classes=5, widen_factor=4, dropRate=0.3, nc=3)
    action_model.to(device) 

    # action_model = VSGCNN(5,3,5,0.4)
    # action_model.to(device)
    
    smap_model_path = args.SMAP_path
    refine_model_path = args.RefineNet_path
    action_model_path = args.Action_path
    hand_skel_path = args.Hand_skel_path
    hand_skel_model = handpose_x_model(model_path=hand_skel_path, device=device)
    if Path(smap_model_path).exists():
        # smap_state_dict = torch.load(smap_model_path, map_location=torch.device('cpu'))
        # smap_state_dict = smap_state_dict['state_dict']
        # model.load_state_dict(smap_state_dict)

        # smap
        state_dict = torch.load(smap_model_path, map_location=torch.device('cpu'))
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)  

        # refine
        if Path(refine_model_path).exists():
            print('using refine net ..')
            refine_state_dict = torch.load(refine_model_path)
            refine_model.load_state_dict(refine_state_dict)
        else:
            refine_model = None

        if Path(action_model_path).exists():
            print('using action net ..')
            action_state_dict = torch.load(action_model_path)
            # action_model.load_state_dict(action_state_dict['state_dict'])
            action_model.load_state_dict(action_state_dict)
        else:
            action_model = None

        if Path(hand_skel_path).exists():
            print("hand skeleton extractor loading : ", hand_skel_path)
            ck = torch.load(hand_skel_path)
            hand_skel_model.model_handpose.load_state_dict(ck)
        else:
            hand_skel_model = None

        if args.mode == 0:
            process_single_image(model, refine_model, cfg, device, args.img_path, args.json_path)

        elif args.mode == 1:
            # set params
            cfg.TEST_MODE = args.test_mode
            cfg.DATA_MODE = args.data_mode
            cfg.REFINE = len(args.RefineNet_path) > 0
            cfg.DO_FLIP = args.do_flip
            cfg.JSON_NAME = args.json_name
            cfg.TEST.IMG_PER_GPU = args.batch_size    # if just run_inference, set to 1
            if args.test_mode == "run_inference":
                test_dataset = CustomDataset(cfg, args.dataset_path)
                data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            else:
                data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)
            generate_3d_point_pairs(model, refine_model, data_loader, cfg, device, output_dir=cfg.TEST_PATH)
        elif args.mode == 2:
            # /media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdVideos
            # /media/xuchengjun/datasets/action/video/walk.avi
            # /home/xuchengjun/Videos
            file_name = args.video_path
            frame_provider = VideoReader(file_name, cfg, args.internal_matrix)
            process_video(model, refine_model, action_model, frame_provider, cfg, device)
            
        elif args.mode == 3:
            rospy.init_node('human_pose', anonymous=True)
            camera_topic = args.camera_topic
            frame_provider = CameraReader(args.camera_topic, cfg, args.internal_matrix)
            run_demo(model, refine_model, action_model, frame_provider, cfg, device)
        
        elif args.mode == 4:
            rospy.init_node('human_pose', anonymous=True)
            camera_topic = args.camera_topic
            # frame_provider = CameraReader(camera_topic, cfg, args.internal_matrix)
            # frame_provider = MultiModalSub("/kinectSDK/color", cfg, 1)
            person_num = "05"
            gesture_num = "02"
            file_name = os.path.join("/media/xuchengjun/disk/datasets/HAND/", gesture_num, person_num, "right.json")   # gesture_num  person_num

            lcrop_depth_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "left", "crop_depth") # gesture_num  person_num
            lcrop_rgb_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "left", "crop_rgb") # gesture_num  person_num
            lhand_skel_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "left", "skel")
            ensure_dir(lcrop_depth_dir)
            ensure_dir(lcrop_rgb_dir)
            ensure_dir(lhand_skel_dir)

            rcrop_depth_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "right", "crop_depth") # gesture_num  person_num
            rcrop_rgb_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "right", "crop_rgb") # gesture_num  person_num
            rhand_skel_dir = os.path.join("/media/xuchengjun/disk/datasets/HAND", gesture_num, person_num, "right", "skel")
            ensure_dir(rcrop_depth_dir)
            ensure_dir(rcrop_rgb_dir)
            ensure_dir(rhand_skel_dir)

            cfg.gesture_num = gesture_num
            frame_provider = ImageReader(file_name, cfg, 1)
            run_demo_with_hand(model, refine_model, hand_skel_model, frame_provider, cfg, device)

        elif args.mode == 5:
            rgb_file_name = args.video_path
            depth_file_name = args.video_depth
            frame_provider = VideoReaderWithDepth(rgb_file_name, depth_file_name, cfg, args.internal_matrix)
            process_rgb_depth(model, refine_model, frame_provider, cfg, device)

        elif args.mode == 6:
            # 这个是采集分好片段的
            image_dir_root = '/media/xuchengjun/zx/process/4'
            image_dir_list = Path(image_dir_root).dirs()

            # image_dir_list = ['/media/xuchengjun/zx/videos/52']  # image files root dir
            csv_dir = '/media/xuchengjun/datasets/action_zx/NEW/3'
            action_label = '3'  # 0-stand  1-walk  2-wave_arm  3-T-pose 4-raise-arm
            total_data_num = 50  #0,1,2中每一个都有200个
            consec_frames = 54    #设定连续动作的帧数，设为54， 32
            
            if not os.path.exists(csv_dir):
                os.mkdir(csv_dir)
                print('have created a new dir ..') 
            print('begin ..')  
            csv_id = 0

            for image_dir in image_dir_list:
                all_image_list = os.listdir(image_dir)

                #这一部主要是对另外两种动作的起始帧数进行一个排序
                image_idx_list_ori = [int(img_name.split(".")[0]) for img_name in all_image_list]
                sort_list = np.array(image_idx_list_ori).argsort()
                image_idx_list = list(np.array(image_idx_list_ori)[sort_list])  #int的列表

                # 2, 3
                print(image_dir)
                angles = [5 * angle_step for angle_step in range(1, 73)]  #5（for (1,61)） 以及 3(for (1,121))
                pbarAngle = tqdm(angles,desc="Angle")
                for single_angle in pbarAngle:
                    none_frame = 0
                    pose_frame_list = []
                    for i in range(len(image_idx_list)):
                        img_path = os.path.join(image_dir, str(image_idx_list[i]) + '.jpg')
                        pose_3d = process_single_image(model, refine_model, cfg, device, img_path, single_angle)
                        if pose_3d is None:
                            none_frame += 1
                            continue
                        pose_frame_list.append(pose_3d)

                    if none_frame >= 10:
                        continue
                    csv_file = os.path.join(csv_dir, action_label + '_' + str(csv_id) + '.csv')
                    with open(csv_file, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(headers)
                        csv_writer.writerows(pose_frame_list)
                    csv_id += 1
                print("csv_id --> ", csv_id)
    else:
        print('no model !')


if __name__ == "__main__":
    # rospy.init_node('human_pose', anonymous=True)
    main()
