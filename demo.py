import sys
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

import dapalib_light
import dapalib
from exps.stage3_root2.config import cfg
from path import Path
from IPython import embed
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from lib.utils.tools import *
from lib.utils.camera_wrapper import CustomDataset, VideoReader, CameraReader, CameraInfo
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

    # [1633.6,0,943.253],
	# [0,1628.87,556.807],
	# [0,0,1] haggling00_00
    # scales['f_x'] = 1633.6
    # scales['f_y'] = 1628.87
    # scales['cx'] = 943.253
    # scales['cy'] = 556.807

    # ultim00_08
    # scales['f_x'] = 1643.03
    # scales['f_y'] = 1639.25
    # scales['cx'] = 946.877
    # scales['cy'] = 562.09  
  
    # ultim00_03
    # scales['f_x'] = 1397.21
    # scales['f_y'] = 1394.06
    # scales['cx'] = 950.563
    # scales['cy'] = 565.694  

    # myself
    scales['f_x'] = 1059.95
    scales['f_y'] = 1053.93
    scales['cx'] = 954.88
    scales['cy'] = 523.74
    stride = cfg.dataset.STRIDE

    # print("pad_value", pad_value)
    img_trans = transform(img_scale)
    img_trans = img_trans.unsqueeze(0).to(device)
    # time_ = 0
    with torch.no_grad():
        st = time.time()
        outputs_2d, outputs_3d, outputs_rd = model(img_trans)
        et = time.time()
        print(f'times -> {et - st}')
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
        st = time.time()
        pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)  # --> tensor, shape:(person_num, 15, 4)
        et = time.time()
        print(f'times -> {et - st}')
        # time_ += et-st

        if pred_bodys_2d.shape[0] == 0:
            print('here is no people ..')
            # zero_pose = [0] * 45
            # return zero_pose
            return None
        # if pred_bodys_2d.shape[0] >= 2:
        #     print('误检测 ..')
        #     return None
        else:
            print(len(pred_bodys_2d))
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
            embed()
            # new_pred_bodys_3d --> numpy()
            if refine_model is not None:
                st = time.time()
                new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                            device=device, root_n=cfg.DATASET.ROOT_IDX)
                et = time.time()
                print(f'times -> {et - st}')
            else:
                new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4) 
            
            # time_ += et - st
            # print(time_)

            # for i in range(new_pred_bodys_3d.shape[0]):
            #     new_pred_bodys_3d[i,0,:3] = (np.array(new_pred_bodys_3d[i,3,:3]) + np.array(new_pred_bodys_3d[i,9,:3])) / 2 

            # # show_3d_results(new_pred_bodys_3d, cfg.SHOW.BODY_EADGES)
            # aug_pose_3d = augment_pose(pred_3d_bodys=new_pred_bodys_3d, angles=angle) 
            # norm_pose_3d = vector_pose_normalization(aug_pose_3d[0])[:,:3]  # 每一帧都要进行norm
            # pose_3d = change_pose(norm_pose_3d)
            embed()
            return new_pred_bodys_3d
            
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

    time_list1 = []
    time_list2 = []
    time_num = []
    frames = 0
    for (img, img_trans, scales) in frame_provider:   # img have processed
        current_time = cv2.getTickCount()
        img_trans = img_trans.to(device)
        st_1 = st_2 = st_3 = 0
        et_1 = et_2 = et_3 = 0
        with torch.no_grad():
            st_1 = time.time()
            outputs_2d, outputs_3d, outputs_rd = model(img_trans)
            et_1 = time.time()
            
            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()

            hmsIn = outputs_2d[0]
            hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255 
            hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 
            rDepth = outputs_rd[0][0]
            # show_map(rDepth)
            
            st_2 = time.time()
            pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
            et_2 = time.time()
            time_list1.append((et_1 - st_1 + et_2 - st_2))

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
            # ids = []
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
                    st_3 = time.time()
                    new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                               device=device, root_n=cfg.DATASET.ROOT_IDX)
                    et_3 = time.time()
                    time_list2.append(et_3 - st_3)
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
                    refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                    # color = pose_color(current_pose_list[i].human_id, max_human)
                    # draw_lines(img, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                    # draw_cicles(refine_pred_2d, img)
                    for i in range(len(new_pred_bodys_3d)):
                        color = pose_color(current_pose_list[i].human_id, max_human)
                        draw_lines_once_only_one(img, refine_pred_2d[i], cfg.SHOW.BODY_EADGES, color=color)
                        cv2.putText(img, 'ID: {}'.format(current_pose_list[i].human_id),(int(refine_pred_2d[i][0,0]-50),int(refine_pred_2d[i][0,1]-200)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255))                        
                else:
                    refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                    draw_lines(img, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                    draw_cicles(refine_pred_2d, img) 
                frames += 1
                print(f'current frames: {frames}')
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
    avg_time1 = np.sum(np.array(time_list1)) / len(time_list1)
    avg_time2 = np.sum(np.array(time_list2)) / len(time_list1)
    print(f'avg_time1: {avg_time1}')
    print(f'avg_time2: {avg_time2}')

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
    rate = rospy.Rate(50)
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
        for (img, img_trans, scales, depth) in frame_provider:   # img have processed
            # current_time = cv2.getTickCount()
            # total = 0

            # depth的测试
            # depth_sum = 0.0
            # depth_trans = depth / 255 * 4096.0
            # for i in range(540 - 25, 540 + 25):
            #     for j in range(960 - 25, 960 + 25):
            #         depth_sum += depth_trans[i,j]
            # depth_avg = depth_sum / 2500
            # print(f'depth: {depth_avg}')
            rospy.loginfo('processing ..')
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
                        # 跟踪匹配序列
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
                                                             

                    # print(id_with_action)
                    # #将当前帧中的坐标转换到世界坐标系  不用发布了，因为得到的结果是错的,在这里发布在世界坐标系中的
                    # world_pose[:,:,:3] /= 100
                    # for pose_data in world_pose:
                    #     pose_data = camera2world(pose_data, T)


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
                            human_identity = "no-ident"
                            color = (0, 0, 0)
                            if human.human_id == 0:
                                color = pose_color(human_id, max_human)
                                draw_lines_once_only_one(img, refine_pred_2d[i], cfg.SHOW.BODY_EADGES, color=color)
                                # draw_cicles_once_only_one(bodys=refine_pred_2d[i], image=img, color=(255,0,0))
                                human_identity = "operator"  
                            elif human_id < max_human and human_id != -1: # pose_color(human.human_id, max_human)
                                # color = (25 * human_id, 25 * human_id, 25 * human_id)
                                color = pose_color(human.human_id, max_human)
                                draw_lines_once_only_one(img, refine_pred_2d[i], cfg.SHOW.BODY_EADGES, color=color)
                                human_identity = "non-operator"
                            if not np.isnan(ori_resoulution_bodys_2D[i][1][1]):
                                cv2.putText(img,"id: {}".format(human.human_id),(int(refine_pred_2d[i][1][0]-100),
                                                            int(refine_pred_2d[i][1][1]-100)),
                                                            cv2.ACCESS_MASK,1,color,2)
                                cv2.putText(img,"iden: {}".format(human_identity),(int(refine_pred_2d[i][1][0]-100),
                                                            int(refine_pred_2d[i][1][1]-50)),
                                                            cv2.ACCESS_MASK,1,color,2) 

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
                    if non_pose_frame >= int(track_maxFrames / 2):
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
            # cv2.imshow("depth", depth / 255)
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
    parser.add_argument('--device',"-de", type=int, default=0) 
    # donot use mode 1 ..
    parser.add_argument('--mode', '-m', type=int, default=0, 
                        help='mode --> 0: process single iamge, 1:using CustomDataset, 2: process video, 3: real-time process')
    parser.add_argument('--internal_matrix','-i', type=int, default=0, help='是否使用自己的相机内参')
    
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

    """
    -----------------------------------   process video -----------------------------------------------
    """
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
    # /media/xuchengjun/disk/datasets/panoptic-toolbox/160906_pizza1/hdVideos
    # /media/xuchengjun/disk/datasets/panoptic-toolbox/170221_haggling_b1
    parser.add_argument('--video_path', type=str, default='/media/xuchengjun/disk/datasets/JTA-Dataset/videos/train/seq_31.mp4')
    parser.add_argument("--json_name", type=str, default="final_json",
                        help='Add a suffix to the result json.')
    """
    -----------------------------------   process single img  -----------------------------------------------
    """
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
    # /media/xuchengjun/disk/datasets/panoptic-toolbox/160422_ultimatum1/hdImgs/00_08/00_08_00015350.jpg
    # /media/xuchengjun/disk/datasets/JTA-Dataset/frames/train/seq_195/47.jpg
    # /media/xuchengjun/disk/datasets/panoptic-toolbox/170407_haggling_a1/hdImgs/00_00/00_00_00006985.jpg
    # 12_00_r_1104.jpg
    parser.add_argument('--json_path', default="/media/xuchengjun/disk/datasets/CMU/CMU_val_json_file/160906_pizza1/160906_pizza1--00_16--00000200.json")
    parser.add_argument('--img_path', default="/home/xuchengjun/ZXin/smap/lib/visualize/1043.jpg")
    
    """
    -----------------------------------   process real camera  -----------------------------------------------
    """
    # kinectSDK/color /kinect2_1/hd/image_color
    parser.add_argument('--camera_topic', default='kinectSDK/color')   # /kinect2_1/hd/image_color  kinectSDK/color
    parser.add_argument('--depth_topic', default='kinectSDK/depth')
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
            frame_provider = CameraReader(camera_topic, cfg, args.internal_matrix)
            run_demo(model, refine_model, action_model, frame_provider, cfg, device)
            
        elif args.mode == 4:
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

                # have_appeared = [] 
                # total_frames = len(image_idx_list)
                # print(total_frames)

                # 0, 1, 4
                # for i in range(0, total_data_num):
                #     consec_frames = random.randrange(50,55)
                #     start_frame = random.sample(image_idx_list, 1)[0]   #在存在的图片中随机选择一张作为动作的起始， int
                    
                #     print(f'start_frame --> {start_frame}  consec_frames --> {consec_frames}')
                #     while True:
                #         if start_frame in have_appeared or start_frame not in image_idx_list:
                #             start_frame = random.sample(image_idx_list, 1)[0]
                #         else:
                #             break
                #     have_appeared.append(start_frame)

                #     angles = [5 * angle_step for angle_step in range(1,73)]  #5（for (1,73)） 以及 3(for (1,121))
                #     pbarAngle = tqdm(angles,desc="Angle")

                #     # aug pose
                #     for single_angle in pbarAngle:

                #         pose_frame_list = []
                #         useful_frame = 0
                #         for j in range(0, consec_frames):

                #             if (start_frame + j) not in image_idx_list:  #不在图片列表中,直接break掉
                #                 break
                #             else:
                #                 img_path = os.path.join(image_dir, str(start_frame + j) + '.jpg')
                #                 pose_3d = process_single_image(model, refine_model, cfg, device, img_path, single_angle)  
                #                 if pose_3d is None:
                #                     # none_frame += 1
                #                     continue
                #                 pose_frame_list.append(pose_3d)
                #                 useful_frame += 1 
                #         if useful_frame < (consec_frames / 2):
                #             continue    
                #         csv_file = os.path.join(csv_dir, action_label + '_' + str(csv_id) + '.csv')
                #         with open(csv_file, 'w', newline='') as csvfile:
                #             csv_writer = csv.writer(csvfile)
                #             csv_writer.writerow(headers)
                #             csv_writer.writerows(pose_frame_list)
                #         csv_id += 1   

                #     print(f"第{i}个完成 ..")  

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
