import numpy as np
from IPython import embed
from time import time
import torch
import math
import random
import cv2
import ctypes
import os
import yaml
# from pykinect2 import PyKinectRuntime
# from pykinect2 import PyKinectV2
from lib.utils.tools import *

# xx = np.zeros(shape=(2,15,4), dtype=np.float)

# xx[0, 2, 2] = 1
# xx[1, 2, 2] = 0.5

# yy = xx[:,2,2].argsort()
# xx = xx[yy]
# embed()

# xx = np.ones(shape=(75,45))
# xx = xx.tolist()
# # xx.pop(0,20)
# del xx[0:20]
# embed()







# xx = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,
#         '17':0,'18':0,'19':0,'20':0,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0,'27':0,'28':0,'29':0,'30':0,'31':0,}

# st = time()
# if "0" in xx.keys():
#     et = time()
#     print(f"total {(et - st)}")

# def change_pose(pred_3d_bodys):
#     """[summary]

#     Args:
#         pred_3d_bodys ([type]): [description]
#         not original 

#     Returns:
#         [type]: [description]
#     """
#     pose_3d = []
#     for i in range(0,1):   # 默认都是1个人
#         for j in range(15):
#             pose_3d.append(pred_3d_bodys[i][j][0])  # x
#             pose_3d.append(pred_3d_bodys[i][j][1])  # y
#             pose_3d.append(pred_3d_bodys[i][j][2])  # z
#     return pose_3d
# xx = np.eye(3)
# yy = np.random.rand(1, 15,3)
# yy =yy.transpose(0,2,1)
# zz = xx @ yy
# zz[0,1] += 1
# zz[0,2] += 1
# embed()

# a = [1,2,3]
# b = [1,2,3]
# c = max(b)
# print(c)

# a = [[1,2,3],[1,2,3]]
# a = np.array(a)

# b = [[1,5,3],[0,0,0]]
# b = np.array(b)

# c = np.array([1,2,3])
# # print(sum(c))
# print(c)
# print(c.argmax(0))

# a = torch.tensor([1,2,3])
# a = 0


# xx = (1 / math.sqrt(2 * math.pi)) * math.exp((-1 / 2) * 0.13)
# xx = math.exp((-1 / 2) * 0.13)
# pri

# xx = random.randrange(30,54)
# print(xx)


# xx = np.array([[1,2,3],[1,2,3]])
# yy = np.delete(xx[:,:],1)
# embed()

# xx = np.array([[ -91.24533081,   -9.77925491,  267.06481934,    1.        ],
#        [ -82.04265594,  -31.73023224,  271.43804932,    1.        ],
#        [ -89.02472687,   40.83181763,  284.30203247,    1.        ],
#        [-104.65914917,  -12.89662933,  276.2901001 ,    1.        ],
#        [-109.66155243,    9.82787323,  289.22158813,    1.        ],
#        [ -85.39533997,    6.52565002,  293.66082764,    1.        ],
#        [ -97.42415619,   39.70267487,  290.0920105 ,    1.        ],
#        [-100.76938629,   71.71859741,  304.46191406,    1.        ],
#        [-110.30347443,  106.38193512,  312.63577271,    1.        ],
#        [ -77.77825165,   -7.06853485,  257.95681763,    1.        ],
#        [ -70.11280823,   17.82071495,  265.47302246,    1.        ],
#        [ -69.68502808,   12.14739037,  288.32354736,    1.        ],
#        [ -80.57032013,   41.87945175,  278.51196289,    1.        ],
#        [ -77.47626495,   75.3965683 ,  291.89306641,    1.        ],
#        [ -78.98562622,  109.38594055,  302.17233276,    1.        ]])

# yy = np.array([[ 195.16946411,  240.30853271,  270.02520752,    2.        ,
#          -97.80656433,   -8.6984005 ,  270.02520752, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 222.62481689,  187.98979187,  273.14260864,    2.        ,
#          -85.97328186,  -32.63896561,  273.14260864, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 213.34616089,  348.8364563 ,  293.92376709,    2.        ,
#          -97.53807831,   44.09518433,  293.92376709, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 173.70863342,  233.23698425,  280.52893066,    2.        ,
#         -112.50686646,  -12.4541378 ,  280.52893066, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 176.00830078,  279.80993652,  292.9100647 ,    2.        ,
#         -116.22911072,   10.05602646,  292.9100647 , 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 216.99163818,  286.08230591,  300.94900513,    2.        ,
#          -97.40164948,   13.34723759,  300.94900513, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 197.83638   ,  343.52972412,  299.24221802,    2.        ,
#         -107.50037384,   42.39523315,  299.24221802, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 207.59408569,  397.00366211,  315.77627563,    2.        ,
#         -108.87758636,   73.62127686,  315.77627563, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 202.59803772,  455.28015137,  322.55981445,    2.        ,
#         -115.69550323,  108.71553802,  322.55981445, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 215.68159485,  247.08103943,  260.17868042,    2.        ,
#          -84.7667923 ,   -5.39123869,  260.17868042, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 234.49308777,  303.28533936,  262.46777344,    2.        ,
#          -77.00579834,   19.09913254,  262.46777344, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 250.81388855,  287.89248657,  284.92578125,    2.        ,
#          -75.51580811,   13.37642765,  284.92578125, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 229.62341309,  354.34918213,  288.60528564,    2.        ,
#          -87.57569885,   45.7951622 ,  288.60528564, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 252.86502075,  412.93890381,  304.94400024,    2.        ,
#          -81.10929108,   78.66136169,  304.94400024, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904],
#        [ 265.61761475,  467.32778931,  317.81060791,    2.        ,
#          -78.63576508,  112.31228638,  317.81060791, 1427.33996582,
#         1423.13000488,  949.61798096,  548.13201904]])


# xx = xx[:,:3]
# yy = yy[:,4:7]
# error = np.linalg.norm(np.abs(xx - yy), axis=1)
# embed()

# def get_last_depth():
#     frame = kinect.get_last_depth_frame()
#     frame = frame.astype(np.uint8)
#     dep_frame = np.reshape(frame,[424,512])
#     return cv2.cvtColor(dep_frame, cv2.COLOR_GRAY)

# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

# frame_type ='depth'
# while True:
#     if frame_type == 'rgb':

#     # if kinect.has_new_color_frame():
#         last_frame = get_last_rbg()
#     else:
#     # if kinect.has_new_depth_frame():
#         last_frame = get_last_depth()

#     cv2.imshow('test', last_frame)
#     cv2.waitKey(1)



# =======================================================================
# ll = ctypes.cdll.LoadLibrary
# lib = ll("./lib/CPPlibs/libcppLib.so")
# # python并不会直接读取到.so的源文件，需要使用.argtypes告诉python在c函数中需要什么参数
# # 这样，在后面使用c函数时pytho会自动处理你的参数，从而达到像调用python参数一样
# lib.cdraw_rectangle.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
# # 同时，python也看不到函数返回什么，默认情况下,python认为函数返回了一个c中的int类型
# # 如果函数返回别的类型，就需要用到retype命令
# lib.cdraw_rectangle.restype = ctypes.c_void_p

# x, y

# l_wrist = [1190, 204] 
# l_elbow = [1277, 307]
# l_wrist = [962, 674] 
# l_elbow = [1012, 628]
# # l_wrist = [1366, 192] 
# # l_elbow = [1294, 312]
# depth_val = 100  # mm
# rec_size = 75 / depth_val * 100
# print(rec_size)
# angle = math.atan2(l_wrist[1] - l_elbow[1], l_wrist[0] - l_elbow[0]) * 180 / math.pi
# # print(angle)
# forearm_length = math.sqrt(pow(l_elbow[0] - l_wrist[0], 2.0) + pow(l_elbow[1] - l_wrist[1], 2.0))
# hand_center_x = l_wrist[0] + (l_wrist[0] - l_elbow[0]) / forearm_length * (forearm_length / 2)
# hand_center_y = l_wrist[1] + (l_wrist[1] - l_elbow[1]) / forearm_length * (forearm_length / 2)

# img = cv2.imread("/home/xuchengjun/ZXin/00_00_00000073.jpg")   # 00_08_00015350.jpg  00_00_00000078.jpg
# # img = torch.tensor(img).to('cuda')
# # img = np.array(img.cpu())
# frame_data = np.asarray(img, dtype = np.uint8)
# frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
# res = lib.cdraw_rectangle(img.shape[0], img.shape[1], frame_data, int(hand_center_x), int(hand_center_y), int(rec_size), angle)   # 后面的参数都是要计算的
#                                                                                                                        # 原始图像的height, width;; 原始图像的数据；； 手的中心（width, height）；； 边界框的信息；； 旋转量 
# print("have cropped hand image ..")

# int_arrPoint = ctypes.c_int * 2
# point_arr = int_arrPoint()
# lib.cadd_twoNum(point_arr)
# print("结果为: ", point_arr[0], point_arr[1])

# int_resPoints = ctypes.c_int * 10
# points_arr = int_resPoints()
# lib.cget_rectangle_points(points_arr)
# print("结果为: ", points_arr[0], points_arr[1])

# 划线
# cv2.line(img, (points_arr[0], points_arr[1]), (points_arr[2], points_arr[3]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[2], points_arr[3]), (points_arr[4], points_arr[5]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[4], points_arr[5]), (points_arr[6], points_arr[7]), (0, 255, 0), 3)
# cv2.line(img, (points_arr[6], points_arr[7]), (points_arr[0], points_arr[1]), (0, 255, 0), 3)
# cv2.circle(img, (int(hand_center_x), int(hand_center_y)), 4, (255, 0, 0), 2)
# cv2.circle(img, (int(hand_center_x) + 45, int(hand_center_y) - 45), 2, (255, 0, 0), 2)

# cv2.circle(img, (l_wrist[0], l_wrist[1]), 6, (255, 0, 0), -1)
# cv2.circle(img, (l_elbow[0], l_elbow[1]), 6, (255, 0, 0), -1)
# cv2.line(img, (l_wrist[0], l_wrist[1]), (l_elbow[0], l_elbow[1]), (0, 200, 255), 1)

# ctypes重载了*, 因此可以使用类型 *n 来表示n个该类型的元素在一起组成一个整体
# int_arr3 = ctypes.c_int * 3
# imgattr_para = int_arr3()
# imgattr_para[0] = 224
# imgattr_para[1] = 224
# imgattr_para[2] = 3
# tmp = ctypes.string_at(res, imgattr_para[0] * imgattr_para[1] * imgattr_para[2])
# nparr = np.frombuffer(tmp, np.uint8)
# img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# # cv2.imshow('source img', img)
# # cv2.imshow("crop img", img_decode)
# # cv2.waitKey(0)
# print('done ..')

# print(1080 * 3 / 4)
# =====================================================================


# =====================================================================

# def yaml_parser(config_base_path, file_name):
#     """
#     YAML file parser.
#     Args:
#         file_name (str): YAML file to be loaded
#         config_base_path (str, optional): Directory path of file
#                                           Default to '../modeling/config'.
        
#     Returns:
#         [dict]: Parsed YAML file as dictionary.
#     """
    
#     cur_dir = os.path.dirname(os.path.abspath(__file__))   # 定位到程序总文件夹的这一级
#     config_base_path = os.path.normpath(os.path.join(cur_dir, config_base_path))
    
#     file_path = os.path.join(config_base_path, file_name + '.yaml')
#     with open(file_path, 'r') as yaml_file:
#         yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
#     return yaml_dict
    
#     print(config_base_path)
    
    
# if __name__ == '__main__':
#     import os
    # gesture_dict =  yaml_parser("lib", "hand_gesture")
    # cur_dir = os.path.dirname(os.path.abspath(__file__))   # 定位到程序总文件夹的这一级
    # gesture_dict = yaml_parser('config', 'hand_gesture', cur_dir)
    # gesture_dict['LEFTHAND']['ONE'] = 13
    # yaml_storer('config', 'hand_gesture_2', gesture_dict, cur_dir)
    # embed()
    # /home/xuchengjun/ZXin/smap/depth_and_rgb
    # /media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/frame_depth/33.jpg
    # depth_img = cv2.imread("/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/depth_frame/133.png")
    # bb, gg, rr = cv2.split(depth_img)
    # bb, gg, rr = cv2.split(depth_img)
    # depth = rr
    # depth_sum = 0.0
    # depth_trans = depth / 255 * 4096.0
    # for i in range(540 - 25, 540 + 25):
    #     for j in range(960 - 25, 960 + 25):
    #         depth_sum += depth_trans[i,j]
    # depth_avg = depth_sum / 2500
    # print(f'depth: {depth_avg}')
    # embed()
    # img = cv2.imread("/media/xuchengjun/disk/datasets/SaveImagesfromKinectV2-master/build/062422/062422_videos/rgb_frame/120.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.circle(img, (960, 540), 10, (255, 0, 0), 2)
    # embed()
    # cv2.imshow('img', depth_img)
    # cv2.waitKey(0)
    # embed()
    
    
    







    # class MultiModalSub(object):
    # def __init__(self, topic_name1, cfg, my_cam):
    #     # ------ img -------
    #     self.color = None
    #     self.depth = None
    #     self.color_convert = None
    #     self.img_header = None

    #     # ------ emg -------
    #     self.emg_sample1 = None
    #     self.emg_sample2 = None
    #     self.emg_list = []


    #     self.process_time = None
    #     self.res_emg = None

    #     self.cv_bridge = CvBridge()
    #     self.color_topic = "/kinectSDK/color"
    #     self.depth_topic = 'kinectSDK/depth'
    #     self.header_topic = '/header'
    #     self.emg_topic = '/myo1/emg'

    #     self.color_sub = message_filters.Subscriber(self.color_topic, Image)  # 对齐时间戳
    #     self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
    #     self.header_sub = message_filters.Subscriber(self.header_topic, Header)
    #     self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.header_sub], queue_size=20, slop=0.1)   # 近似同步
    #     self.syc.registerCallback(self.callback)
        
    #     self.emg_sub = rospy.Subscriber(self.emg_topic, Emg, self.emgCallback)

    #     self.is_accept = False
    #     self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
    #     #图片预处理
    #     normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    #     transform = transforms.Compose([transforms.ToTensor(), normalize])
    #     self.cam = my_cam
    #     if self.cam:
    #         print('using myself cam')
    #         cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
    #         self.K = np.array(cam_data['kinect_1'])
    #     else:
    #         print('using dataset cam')
    #         cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
    #         self.K = np.array(cam_data['K'])

    #     self.transform = transform
    #     print('setting done ..')

    # def callback(self, msg_color, msg_depth, msg_header):

    #     self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
    #     self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
    #     self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')
    #     self.img_header = msg_header.stamp.to_sec()
    #     self.is_accept = True

    #     # 因为在c++端中初始化了一个header,并同时赋给了color和depth中的header
    #     # print(f'color: {msg_color.header.stamp.to_sec()}')
    #     # print(f'depth: {msg_depth.header.stamp.to_sec()}')
        
    # # 暂时没有用
    # # def headerCallback(self, msg):
    # #     print(msg.stamp.to_sec())

    # def emgCallback(self, msg_emg):
    #     """
    #     这个is_accept的标志位:是当图像被接收到时,立马置为true,
    #         此时emg_list应清空,但需要保存这一步的值(近似图像和肌电传感器信号对齐)
    #         is_accept值为false
    #         当is_accept为false的时候,会一直保存当前图像段内的肌电信号
    #         直到接收到下一帧的图像

    #     手势识别时,只有当is_accept为true时,表明
    #     """

    #     # if not self.is_accept:
    #     #     self.emg_list.append(list(msg_emg.sample1))
    #     # elif self.is_accept:
    #     #     self.is_accept = False
    #     #     self.emg_list = []
    #     #     self.emg_list.append(list(msg_emg.sample1))   # 当前的也需要保存 .. 
    #     #     if self.is_first:
    #     #         print('<<<<  <<<<  <<<<  <<<<  <<<<  <<<<  <<<<  First Loop !!!  >>>>   >>>>  >>>>  >>>>  >>>>  >>>>  >>>>')   // 第一次就不用了， 程序刚打开的时候，肯定不会是
    #     #         self.is_first = False
        
    #     self.emg_list.append(list(msg_emg.sample2))
    #     if self.is_accept:
    #         self.is_accept = False
    #         self.res_emg = self.emg_list
    #         self.emg_list = []
    #         self.emg_list.append(list(msg_emg.sample2))
        
    #     # 时间戳
    #     # print(f'emg: {msg_emg.header.stamp.to_sec()}')
    
    # def __iter__(self):
    #     return self

    # def __next__(self):

    #     # print(self.color_convert)
    #     if self.color is None or self.color_convert is None or self.depth is None: 
    #         raise StopIteration

    #     # transfrom the img
    #     net_input_image, scale = self.aug_croppad(self.color_convert)
    #     scale['K'] = self.K
    #     scale['f_x'] = self.K[0,0]
    #     scale['f_y'] = self.K[1,1]
    #     scale['cx'] = self.K[0,2]
    #     scale['cy'] = self.K[1,2]
    #     net_input_image = self.transform(net_input_image)
    #     net_input_image = net_input_image.unsqueeze(0)

    #     # if len(self.emg_list) >= 100:
    #     #     del self.emg_list[0:20]
    #     # print(len(self.emg_list))

    #     return self.color_convert, net_input_image, scale, self.depth, self.res_emg, self.is_accept

    # def aug_croppad(self, img):
    #     scale = dict()                    #创建字典
    #     crop_x = self.net_input_shape[0]  # width 自己设定的
    #     crop_y = self.net_input_shape[1]  # height 512
    #     scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
    #     img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
    #     scale['img_width'] = img.shape[1]
    #     scale['img_height'] = img.shape[0]
    #     scale['net_width'] = crop_x
    #     scale['net_height'] = crop_y
    #     pad_value = [0,0]  # left,up

    #     center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
    #     if img_scale.shape[1] < crop_x:    # pad left and right
    #         margin_l = (crop_x - img_scale.shape[1]) // 2
    #         margin_r = crop_x - img_scale.shape[1] - margin_l
    #         pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
    #         pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
    #         pad_value[0] = margin_l
    #         img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
    #     elif img_scale.shape[0] < crop_y:  # pad up and down
    #         margin_u = (crop_y - img_scale.shape[0]) // 2
    #         margin_d = crop_y - img_scale.shape[0] - margin_u
    #         pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
    #         pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
    #         pad_value[1] = margin_u
    #         img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
    #     scale['pad_value'] = pad_value
        
    #     return img_scale, scal


hand_skel_edge = [[0, 1], [1, 2], [2, 3], [3, 4],
                  [0, 5], [5, 6], [6, 7], [7, 8],
                  [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16],
                  [0, 17], [17, 18], [18, 19], [19, 20]]

def putVecMaps3D(centerA, centerB, accumulate_vec_map, thre):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    stride = 1
    crop_size_y = 160
    crop_size_x = 160
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA  # x,y
    limb_z = 1.0
    norm = np.linalg.norm(limb_vec)
    if norm < 1.0:  # limb is too short, ignore it
        return accumulate_vec_map

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
    vec_map[0, yy, xx] = np.repeat(mask[np.newaxis, :, :], 1, axis=0)
    vec_map[0, yy, xx] *= limb_z
    mask = np.logical_or.reduce(
        (np.abs(vec_map[0, :, 0]) != 0))
    
    accumulate_vec_map += vec_map
    
    return accumulate_vec_map


if __name__ == "__main__":
    import pandas as pd
    # csv_file = "./hand_data.csv"
    # data = read_csv(csv_file)
    # hand_numpy = np.array(data, dtype=np.float)[:, 1:]
    # labels = np.zeros((1, 160, 160))
    # for i in range(hand_numpy.shape[0] - 1):
    #     centerA = np.array(hand_numpy[hand_skel_edge[i][0]], dtype=int)
    #     centerB = np.array(hand_numpy[hand_skel_edge[i][1]], dtype=int)
    #     labels += putVecMaps3D(centerA, centerB, labels, 1)
    
    # labels[labels > 1] = 1
    # labels *= 255
    # cv2.imshow('img', labels[0])
    # cv2.waitKey(0)
    # cv2.imwrite('./test.jpg', labels[0])
    # print('ok')


    # human_id_hg_dict = {}
    # human_id = 0
    # lhg_num = 11
    # rhg_num = 15
    # accum_frame = 6
    # human_id_hg_dict[str(human_id)] = [np.array([0] * lhg_num), np.array([0] * rhg_num), accum_frame, accum_frame, np.array([0] * lhg_num), np.array([0] * rhg_num)]
    # human_id_hg_dict[str(human_id)][0][0] += 1
    # # print(np.where(human_id_hg_dict[str(human_id)][0] > 3))
    # human_id_hg_dict[str(human_id)][0] *= 0
    # print(human_id_hg_dict[str(human_id)][0])

    # xx = [0, 2, 3]
    # if 1 in xx:
    #     print("123")
    # # xx.remove(1)
    # print(xx)

    # xx = {'0':[1,2], '1':[0, 2]}
    # xx.pop(str(0))
    # if str(0) in xx.key():
    #     print(xx['0'])
    


















