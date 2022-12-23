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
# from human_pose_msg.msg import HumanList, Human, PointCoors
from human_hg_msg.msg import HumanList, Human, PointCoors
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
from model.hand.mynet import MyNet
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
from lib.utils.camera_wrapper import CustomDataset, MultiModalSub, VideoReader, VideoReaderWithDepth, CameraReader, MultiModalSubV2
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
import pandas as pd

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
lib.ccrop_rgb_depth.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int]

# 同时，python也看不到函数返回什么，默认情况下,python认为函数返回了一个c中的int类型
# 如果函数返回别的类型，就需要用到retype命令
lib.cdraw_rectangle.restype = ctypes.c_void_p
# lib.ccrop_rgb_depth.restype = POINTER(Result_process)  # 读取不到缓冲区的数据
lib.cget_cropped_rgb.restype = ctypes.c_void_p
lib.cget_cropped_depth.restype = ctypes.c_void_p
lib.cdraw_rectangle_depth.restype = ctypes.c_void_p
lib.cdelete.restype = ctypes.c_void_p
# human_id_hg_dict = {}
int_arr3 = ctypes.c_int * 3
imgattr_para = int_arr3()
imgattr_para[0] = 160   # width
imgattr_para[1] = 160   # height
imgattr_para[2] = 3              # channel
file_root_path = "/media/xuchengjun/disk1/zx/dataset/HAND/00/00/0-563"

center_ = [[int(100), int(100)], [int(200), int(200)], [int(300), int(300)], [int(400), int(400)], [int(500), int(500)]]

person_num = 2
for i in range(5):
    rgb_path = os.path.join(file_root_path, 'rgb', '00_00_r_' + str(i) + '.jpg')
    depth_path = os.path.join(file_root_path, 'depth', '00_00_d_' + str(i) + '.png')
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path)[:, :, 0]
    frame_data = np.asarray(rgb, dtype = np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
    frame_data_depth = np.asarray(depth, dtype = np.uint8)
    frame_data_depth = frame_data_depth.ctypes.data_as(ctypes.c_char_p)
    img_shape = rgb.shape
    for j in range(person_num):
        lib.ccrop_rgb_depth(rgb.shape[0], rgb.shape[1], frame_data, frame_data_depth, center_[j][0] + j * 50, center_[j][1] + j * 50, int(80), 0, i, person_num)
        
        rencoded_cropped_hand_img = lib.cget_cropped_rgb()
        rtmp_rgb = ctypes.string_at(rencoded_cropped_hand_img, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
        rnparr = np.frombuffer(rtmp_rgb, np.uint8)
        cropped_rhand_img = cv2.imdecode(rnparr, cv2.IMREAD_COLOR)

        rencoded_cropped_depth_img = lib.cget_cropped_depth()
        rtmp_depth = ctypes.string_at(rencoded_cropped_depth_img, imgattr_para[0] * imgattr_para[1])
        rnparr_depth = np.frombuffer(rtmp_depth)
        cropped_rhand_depth = cv2.imdecode(rnparr_depth, cv2.IMREAD_GRAYSCALE)

        lib.ccrop_rgb_depth(rgb.shape[0], rgb.shape[1], frame_data, frame_data_depth, center_[j][0] + j * 100, center_[j][1] + j * 100, int(80), 0, i, person_num)
        lib.cdelete()  # 销毁堆中的内存   
        cv2.imshow('img', cropped_rhand_img)
        cv2.imshow('depth', cropped_rhand_depth)
        cv2.waitKey(0)