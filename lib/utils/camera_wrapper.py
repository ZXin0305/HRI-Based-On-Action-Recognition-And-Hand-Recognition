from doctest import FAIL_FAST
import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
sys.path.append('/home/xuchengjun/catkin_ws/src')   # 必须要导入，不然找不到对应的msg文件
import os.path as osp
import glob
import sys
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from path import Path
from IPython import embed
from lib.utils.tools import read_json, transform, show_map
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import message_filters
import tf
from autolab_core import RigidTransform
import copy
import message_filters
# from sub_myo.msg import Emg         # 可以支持两个MYO设备
from ros_myo_cpp.msg import EmgArray  # 可以扩展成两个MYO设备
from time import time
import os

class CustomDataset(Dataset):
    def __init__(self, cfg, dataset_path, dataset_name='CMU'):
        """
        this is to set the image dataset ..
        """
        # self.calibration_json_file = None
        self.dataset_name = dataset_name
        # if self.dataset_name == 'CMU':
        #     self.calibration_json_file = '/home/zx/panoptic-toolbox/170407_haggling_a1/calibration_170407_haggling_a1.json'
        #     self.cali_file = read_json(self.calibration_json_file)

        self.dataset_path = dataset_path
        #把该路径下的所有图片都加在进来（只是说的图片的路径）
        self.image_list = glob.glob(osp.join(dataset_path, '**/*.jpg'), recursive=True)    #**/*.jpg会在路径下进行迭代  返回的是路径的list
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.png'), recursive=True))  #该方法没有返回值，但会在已存在的列表中添加新的列表内容。
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.jpeg'), recursive=True))
        # self.image_list = self.image_list[0:5]

        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = transform

    def get_cam_(self, cali_file, cam_id):
        cameras = {(cam['panel'], cam['node']): cam for cam in cali_file['cameras']}
        cam = cameras[cam_id]
        return cam['K']


    def __getitem__(self, index):
        image_path = self.image_list[index].rstrip()
        image_name = image_path.replace(self.dataset_path, '').lstrip('/')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.image_shape = (image.shape[1], image.shape[0])  # --> (width, heght)

        net_input_image, scale, pad_value = self.aug_croppad(image)
        net_input_image = self.transform(net_input_image)

        # if self.calibration_json_file is not None:
        #     # cam_id
        #     cam_id = image_name.split('/')[-1].split('_')
        #     lnum, rnum = int(cam_id[0]), int(cam_id[1])
        #     cam = self.get_cam_(self.cali_file, (lnum, rnum))
        #     scale['cam'] = cam

        return net_input_image, image_name, scale

    def __len__(self):
        return len(self.image_list)

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
        scale['pad_value'] = pad_value
        return img_scale, scale, pad_value


class VideoReader:
    def __init__(self, file_name, cfg, my_cam):
        """_summary_

        Args:
            file_name (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            读取正常的视频
        """
        self.file_name = file_name
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, image = self.cap.read()
        if not was_read:
            raise StopIteration
        
        # transfrom the img
        net_input_image, scale = self.aug_croppad(image)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return image, net_input_image, scale

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

class VideoReaderWithDepth:
    def __init__(self, rgb_file_name, depth_file_name, cfg, my_cam):
        """_summary_

        Args:
            rgb_file_name (_type_): _description_
            depth_file_name (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            同时读取保存直接由kinect得到的rgb以及depth形成的视频
        """
        self.rgb = rgb_file_name
        self.depth = depth_file_name
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        
        # try:
        #     self.file_name = int(file_name)
        # except ValueError:
        #     pass

    def __iter__(self):
        self.cap_rgb = cv2.VideoCapture(self.rgb)
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap_depth = cv2.VideoCapture(self.depth)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_depth.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)       
        if not self.cap_rgb.isOpened() or not self.cap_depth.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.rgb))
        return self

    def __next__(self):
        rgb_is_read, image = self.cap_rgb.read()
        depth_is_read, depth = self.cap_depth.read()
        if not rgb_is_read or not depth_is_read:
            raise StopIteration
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)   # 20220714 不确定， 在从bin文件中提取出来的时候，RGB的编码格式是BGRA8, 所以在这里制作数据集的时候需要进行转换， 以便和实际的对上
                                                          # 但是实际上不能用BGR转换，读取之后直接用RGBA到RGB就可以，显示出来是一样的
        # transfrom the img
        net_input_image, scale = self.aug_croppad(image)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return image, depth, net_input_image, scale

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

class CameraReader(object):
    def __init__(self, topic_name1, cfg, my_cam):
        """_summary_

        Args:
            topic_name1 (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            也是同步获取由kinect/SDK发布的图像数据
            和MultimodalSubV2的功能一样
        """
        self.color = None
        self.depth = None
        self.color_convert = None
        self.cv_bridge = CvBridge()
        self.color_topic = topic_name1
        self.depth_topic = '/kinectSDK/depth'
        self.color_sub = message_filters.Subscriber(self.color_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.syc.registerCallback(self.callback)
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform

    def callback(self, msg_color, msg_depth):
        # rospy.loginfo('Image has received...')
        self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
        self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
        self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')


    def __iter__(self):
        return self

    def __next__(self):

        # if self.image is None or self.image_convert is None:
        if self.color is None or self.color_convert is None or self.depth is None: 
            raise StopIteration

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.color_convert)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return self.color_convert, net_input_image, scale, self.depth, None, None

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

class MultiModalSub(object):
    def __init__(self, topic_name1, cfg, my_cam):
        """_summary_

        Args:
            topic_name1 (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            接收rgb, depth, emg
            但是emg最后没有用了，效果不太好
        """
        # ------ img -------
        self.color = None
        self.depth = None
        self.color_convert = None
        self.img_header = None
        self.img_time = time()

        # ------ emg -------
        self.emg_sample1 = None
        self.emg_sample2 = None
        self.emg_list = []
        self.emg_header = []


        self.process_time = None
        self.res_emg = None

        self.cv_bridge = CvBridge()
        self.color_topic = "/kinectSDK/color"
        self.depth_topic = '/kinectSDK/depth'
        self.header_topic = '/header'
        self.emg_topic = '/myo_raw/myo_emg'  #  /myo1/emg   /myo_raw/myo_emg

        self.color_sub = message_filters.Subscriber(self.color_topic, Image)  # 对齐时间戳
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        # self.header_sub = message_filters.Subscriber(self.header_topic, Header)
        self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=20, slop=0.1)   # 近似同步
        self.syc.registerCallback(self.callback)
        
        self.emg_sub = rospy.Subscriber(self.emg_topic, EmgArray, self.emgCallback)  # emg  EmgArray

        self.is_accept = False
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        print('setting done ..')

    def callback(self, msg_color, msg_depth):

        self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
        self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
        self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')
        self.img_header = msg_color.header.stamp.to_sec()
        # print('img header:  ', self.img_header)
        self.is_accept = True
        # self.pre.append(self.img_header)
        # print(self.pre)

        # 因为在c++端中初始化了一个header,并同时赋给了color和depth中的header
        # print(f'color: {msg_color.header.stamp.to_sec()}')
        # print(f'depth: {msg_depth.header.stamp.to_sec()}')

    def emgCallback(self, msg_emg):
        """
        这个is_accept的标志位:是当图像被接收到时,立马置为true,
            此时emg_list应清空,但需要保存这一步的值(近似图像和肌电传感器信号对齐)
            is_accept值为false
            当is_accept为false的时候,会一直保存当前图像段内的肌电信号
            直到接收到下一帧的图像

        手势识别时,只有当is_accept为true时,表明
        """

        self.emg_list.append(list(msg_emg.data))
        self.emg_header.append(msg_emg.header.stamp.to_sec())
        if self.is_accept:
            # print(f'time: {time() - self.img_time}')
            # self.img_time = time()
            self.is_accept = False
            idx = np.where(np.array(self.emg_header) <= self.img_header)
            # print('emg stamp:  ', self.emg_header[0:10])
            self.res_emg = np.array(self.emg_list)[idx]  # --> array
            del self.emg_list[0:len(idx[0])]
            del self.emg_header[0:len(idx[0])]
            # self.emg_list.append(list(msg_emg.data))
            # self.emg_header.append(msg_emg.header.stamp.to_sec())
        
        # 时间戳
        # print(f'emg: {msg_emg.header.stamp.to_sec()}')
    
    def __iter__(self):
        return self

    def __next__(self):

        # print(self.color_convert)
        if self.color is None or self.color_convert is None or self.depth is None: 
            raise StopIteration

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.color_convert)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        # if len(self.emg_list) >= 100:
        #     del self.emg_list[0:20]
        # print(len(self.emg_list))

        return self.color_convert, net_input_image, scale, self.depth, self.res_emg, self.is_accept

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale


# 数据采集时候用
class DatasetWrapper(object):
    def __init__(self, topic_name1, cfg, my_cam):
        """_summary_

        Args:
            topic_name1 (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            采集RGB和depth数据的时候使用
        """
        # ------ img -------
        self.color = None
        self.depth = None
        self.color_convert = None
        self.img_header = None

        self.cv_bridge = CvBridge()
        self.color_topic = "/kinectSDK/color"
        self.depth_topic = '/kinectSDK/depth'
        self.header_topic = '/header'

        self.color_sub = message_filters.Subscriber(self.color_topic, Image)  # 对齐时间戳
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=20, slop=0.1)   # 近似同步
        self.syc.registerCallback(self.callback)
        print('setting done ..')

    def callback(self, msg_color, msg_depth):

        self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
        self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
        self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')
        self.img_header = msg_color.header.stamp.to_sec()

    def __iter__(self):
        return self

    def __next__(self):

        if self.color is None or self.color_convert is None or self.depth is None: 
            raise StopIteration
        return self.color_convert, self.depth


class ImageReader(object):
    def __init__(self, file_names, cfg, my_cam):
        """_summary_

        Args:
            file_names (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            用来提取手部区域 -- 制作数据集
        """
        self.color = None
        self.depth = None
        self.file_names = file_names
        # self.max_idx = len(file_names)
        print(file_names)
        self.json_file = read_json(file_names)
        self.img_list = self.json_file['data']
        self.max_idx = len(self.img_list)
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0])
        self.root = '/media/xuchengjun/disk/datasets'

        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img_name = self.img_list[self.idx][0]
        depth_name = self.img_list[self.idx][1]
        self.color = cv2.imread(os.path.join(self.root, img_name), cv2.IMREAD_COLOR)
        self.depth = cv2.imread(os.path.join(self.root, depth_name), cv2.IMREAD_GRAYSCALE)
        # self.depth = self.depth[:, :,]
        if self.color is None:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.color)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)
        self.idx = self.idx + 1

        return self.color, net_input_image, scale, self.depth, img_name, depth_name, self.root

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale
    

class ImageReaderWoTrans(object):
    """_summary_

    Args:
        object (_type_): _description_
        
        这个是用来增强数据集时读取数据
    """
    def __init__(self, file_names, cfg):
        self.color = None
        self.depth = None
        self.file_names = file_names
        # self.max_idx = len(file_names)
        print(f'current used original json ---->>> {file_names}')
        self.json_file = read_json(file_names)
        self.img_list = self.json_file['data']
        self.max_idx = len(self.img_list)
        # print(len(self.img_list))
        self.root = cfg.json1_root

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return len(self.img_list)

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img_name = self.img_list[self.idx][0]
        # print(img_name)
        depth_name = self.img_list[self.idx][1]
        skel_name = self.img_list[self.idx][2]

        self.color = cv2.imread(os.path.join(self.root, img_name), cv2.IMREAD_COLOR)
        self.depth = cv2.imread(os.path.join(self.root, depth_name))
        if self.color is None:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))

        self.idx = self.idx + 1

        return self.color, self.depth, img_name, depth_name, skel_name, self.root

"""
===================================================================================================
===================================================================================================
===================================================================================================
"""

class MultiModalSubV2(object):
    def __init__(self, topic_name1, cfg, my_cam):
        """_summary_

        Args:
            topic_name1 (_type_): _description_
            cfg (_type_): _description_
            my_cam (_type_): _description_
            
            多模态--仅仅接收rgb和depth
        """
        # ------ img -------
        self.color = None
        self.depth = None
        self.color_convert = None
        self.img_header = None

        self.cv_bridge = CvBridge()
        self.color_topic = "/kinectSDK/color"
        self.depth_topic = '/kinectSDK/depth'
        self.header_topic = '/header'

        # marker ..
        self.marker_topic = "/marker_detect"
        self.marker_disappear = 0
        self.have_marker = False
        self.marker_id = 1
        self.is_accept = False

        self.marker_sub = rospy.Subscriber(self.marker_topic, Int32MultiArray, self.markerCallback)

        self.color_sub = message_filters.Subscriber(self.color_topic, Image)  # 对齐时间戳
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        # self.header_sub = message_filters.Subscriber(self.header_topic, Header)
        self.syc = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=20, slop=0.1)   # 近似同步
        self.syc.registerCallback(self.callback)

        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0]) # (width, height)
        
        #图片预处理
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.cam = my_cam
        if self.cam:
            print('using myself cam')
            cam_data = read_json("/home/xuchengjun/ZXin/smap/cam_data/myCam.json")
            self.K = np.array(cam_data['kinect_1'])
        else:
            print('using dataset cam')
            cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
            self.K = np.array(cam_data['K'])

        self.transform = transform
        print('setting done ..')

    def callback(self, msg_color, msg_depth):

        self.color = self.cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')     # 有时间试试可不可以使用[:, :, :3], 这样速度是不是快一些
        self.color_convert = cv2.cvtColor(self.color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
        self.depth = self.cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')
        # self.img_header = msg_color.header.stamp.to_sec()

        self.is_accept = True

    def markerCallback(self, marker_msg):
        
        if self.is_accept:
            self.is_accept = False
            marker_ids = marker_msg.data
            if self.marker_id in marker_ids:
                self.marker_disapper = 0
                self.have_marker = True
            else:
                self.marker_disappear += 1
                if self.marker_disappear > 3:
                    self.have_marker = False
    

    def __iter__(self):
        return self

    def __next__(self):

        if self.color is None or self.color_convert is None or self.depth is None: 
            raise StopIteration

        # transfrom the img
        net_input_image, scale = self.aug_croppad(self.color_convert)
        scale['K'] = self.K
        scale['f_x'] = self.K[0,0]
        scale['f_y'] = self.K[1,1]
        scale['cx'] = self.K[0,2]
        scale['cy'] = self.K[1,2]
        net_input_image = self.transform(net_input_image)
        net_input_image = net_input_image.unsqueeze(0)

        return self.color_convert, net_input_image, scale, self.depth, self.have_marker

    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale

class GetEmg(object):
    def __init__(self):
        """_summary_
           将接收emg单独分开了
        """
        self.emg_list = []
        self.emg_header = []

        self.process_time = None
        self.res_emg = []

        self.emg_topic = '/myo_raw/myo_emg'   # /myo_raw/myo_emg   /myo1/emg

        self.emg_sub = rospy.Subscriber(self.emg_topic, EmgArray, self.emgCallback)  # EmgArray  Emg
        print('setting done ..')

    def emgCallback(self, msg_emg):
        self.emg_list.append(list(msg_emg.data))
        self.emg_header.append(msg_emg.header.stamp.to_sec())
        # print(len(self.emg_list))
        self.res_emg = [np.array(self.emg_list), np.array(self.emg_header)] # --> array
        if len(self.emg_list) > 50:
            self.emg_list = []
            self.emg_header = []
    
    def __iter__(self):
        return self

    def __next__(self):
        if len(self.emg_list) <= 0:
            raise StopIteration
        return self.res_emg


"""
===================================================================================================
===================================================================================================
===================================================================================================
"""

class CameraInfo:
    def __init__(self):
        self.child_frame = "camera_base_1"
        self.parent_frame = "marker_0"

    def get_tf(self):  # camera to world
        listener = tf.TransformListener()
        rate = rospy.Rate(30)
        rot_matrix = None
        trans = None
        while not rospy.is_shutdown():
            try:
                (t, q) = listener.lookupTransform(self.child_frame, self.parent_frame, rospy.Time(0))
                # embed()
            except:
                rospy.loginfo("wait for camera info ..")
                rate.sleep()
                continue

            #得到相应的旋转矩阵和平移向量
            trans = self.t2trans(t)
            ts = trans
            rot_matrix = self.q2rot(q)
            break

        rospy.loginfo_once("have get camera2world info ..")
        print(rot_matrix)
        print(trans)
        return rot_matrix, ts

    def q2rot(self, q): 
        """四元数到旋转矩阵
        
        """
        w, x, y, z= q[0], q[1], q[2], q[3]
        # qq = np.array([w,x,y,z])
        # rotMat = RigidTransform(qq, trans)
        rotMat = np.array(
            [[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
             [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
             [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]]
        )
        # rotMat = np.array([[ 0.977503 ,  0.210436 ,-0.0143229],[0.0273426 , -0.193757,  -0.980668],[-0.209143,   0.958214 , -0.195152]])
        return rotMat
    
    def t2trans(self, t):
        """转换成numpy形式的平移向量
        """
        trans = np.zeros((3, 1), dtype=np.float32)
        for i in range(len(t)):
            trans[i] = t[i]
        # trans = np.array([-0.414375,0.68425,2.33348])
        return trans

if __name__ == '__main__':
    rospy.init_node("test_get_camera_info", anonymous=True)
    # rot_matrix, trans = CameraInfo().listen_tf() 
    # print(rot_matrix, trans)

    data = ImageReader('/media/xuchengjun/disk/datasets/HAND/right/00/00/right.json')
    embed()
