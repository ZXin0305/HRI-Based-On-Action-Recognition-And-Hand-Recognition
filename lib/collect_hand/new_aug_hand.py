import sys
sys.path.append("/home/xuchengjun/ZXin/smap")
from curses import meta
from select import select
from sys import meta_path
from turtle import shape
import cv2
import ctypes
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from path import Path
import math
import random
from skimage import data, exposure, img_as_float
import argparse
from lib.utils.tools import *
from lib.utils.camera_wrapper import *

#  ==========================  #
#  给三种方式  #
#  1. depth mask
#  2. cv2.grabCut
#  3. depth mask + cv2.grabCut
#  其中:2的分割效果较好,1因为depth mask会有一定的shallow
#  但是如果1的mask比较好的话,效果也是可以的,同时比较快
rgb_suffix = 'cropped_img_'
depth_suffix = 'cropped_depth_'

def segment_hand(img, depth, bg, mode, resize_shape=(160, 160), H_ = 15, W_ = 5):
    img_shape = img.shape
    bg_ = cv2.resize(bg, resize_shape)
    if mode == 1:
        size_ = 4
        ratio_ = 0.18
        max_val = 1 + ratio_
        min_val = 1 - ratio_
        img_copy = img.copy()
        img_center = [int(img_shape[0] / 2 + H_ + 0.5), int(img_shape[1] / 2 + 0.5 + W_)]
        depth = depth / 255.0 * 4096.0 / 10 # --> cm
        total_depth = np.sum(depth[img_center[0] - size_ : img_center[0] + size_,
                                   img_center[1] - size_ : img_center[1] + size_,
                                   0], dtype=np.float)          
        mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
        depth_mask = depth.copy()
        # depth_mask = depth_mask[:,:,0]
        depth_mask[depth_mask > mean_depth_val * max_val] = 0
        depth_mask[depth_mask < mean_depth_val * min_val] = 0
        depth_mask[depth_mask != 0] = 1 
        depth_mask = np.array(depth_mask, dtype=np.uint8)

        # 膨胀
        # 使用mask增强数据的时候，使用膨胀加上中值滤波, 单纯处理的时候，就直接使用中值滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        depth_mask = cv2.dilate(depth_mask, kernel)
        depth_mask = cv2.medianBlur(depth_mask, 3)  # 中值滤波

        for i in range(3):
            img_copy[:,:,i] = depth_mask[:, :, i] * img_copy[:,:,i]

        depth_mask_inverse = depth_mask.copy()
        depth_mask_inverse[depth_mask == 1] = 0
        depth_mask_inverse[depth_mask == 0] = 1
        for i in range(3):
            bg_[:,:,i] = depth_mask_inverse[:, :, i] * bg_[:,:,i]
        img_mask = img_copy + bg_      
        return img_mask 
        # cv2.imshow('depth_mask', depth_mask)
        # cv2.imshow('img_new', img_mask)
        # cv2.imwrite('./depth_mask.jpg', depth_mask * 255)
        # cv2.imwrite('./depth_mask_in.jpg', depth_mask_inverse * 255)
        # cv2.imwrite('./ori_img.jpg', img)
        # cv2.imwrite('./mask_bg.jpg', bg_)
        # cv2.imwrite('./mask_img.jpg', img_mask)
        # cv2.imwrite('./tmp_img.jpg', img_copy)
 
    elif mode == 2:
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img.shape[1], img.shape[0])
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_copy = img * mask2[:, :, np.newaxis]
        depth_mask_inverse = mask2.copy()
        depth_mask_inverse[mask2 == 1] = 0
        depth_mask_inverse[mask2 == 0] = 1
        for i in range(3):
            bg_[:,:,i] = depth_mask_inverse * bg_[:,:,i]  
        img_copy = img_copy + bg_      
        return img_copy  
        # cv2.imshow('mask', mask2 * 255)
        # cv2.imshow('img_new', img_copy)
        # cv2.imwrite('./depth_mask_grab.jpg', mask2 * 255)
        # cv2.imwrite('./img_grab.jpg', img_copy)

    elif mode == 3:
        size_ = 3
        ratio_ = 0.15
        max_val = 1 + ratio_
        min_val = 1 - ratio_
        img_copy = img.copy()
        img_center = [int(img_shape[0] / 2 + 0.5), int(img_shape[1] / 2 + 0.5)]
        depth = depth / 255.0 * 4096.0 / 10 # --> cm
        total_depth = 0.0
        for i in range(img_center[0] - size_, img_center[0] + size_):
            for j in range(img_center[1] - size_, img_center[1] + size_):
                total_depth += depth[i,j]  
        mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
        depth_mask = depth.copy()
        depth_mask = depth_mask[:,:,0]
        depth_mask[depth_mask > mean_depth_val[0] * max_val] = 0
        depth_mask[depth_mask < mean_depth_val[0] * min_val] = 0
        depth_mask[depth_mask != 0] = 1 
        for i in range(3):
            img_copy[:,:,i] = depth_mask * img_copy[:,:,i]
        mask = np.zeros(img_copy.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img_copy.shape[1], img_copy.shape[0])
        cv2.grabCut(img_copy, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_copy = img_copy * mask2[:, :, np.newaxis]
        # cv2.imshow('img_new', img_copy)
        return img_copy

    key = cv2.waitKey(0)  
    if key == 27:
        return 'break'  

def process_bg(bg):
    """
    处理背景图
    """
    bg_shape = bg.shape
    crop_size = 350
    if crop_size > bg_shape[0] or crop_size > bg_shape[1]:
        crop_size = min(bg_shape[0], bg_shape[1])
    crop_bg_x = random.randint(10, int(bg_shape[1]/2))
    crop_bg_y = random.randint(10, int(bg_shape[0]/2))
    cropped_bg = bg[crop_bg_y:crop_bg_y+crop_size, crop_bg_x:crop_bg_x+crop_size, :]
    # cv2.imwrite('./cropped_bg.jpg', cropped_bg)
    return cropped_bg

#  其余的离线数据增强的方式
#  高斯模糊
def AugGaussianFilter(img, ksize=[7,7]):
    img_copy = img.copy()
    img_copy = cv2.GaussianBlur(img_copy, ksize, 0, 0)
    return img_copy

# 高斯噪声
def AugGaussianNoise(img, loc=0.0, sigma=0.1):
    img_copy = img.copy()
    img_copy = np.array(img_copy / 255, dtype=np.float)
    noise = np.random.normal(loc, sigma, img_copy.shape)    # 正态分布函数
    gaussian_noise = img_copy + noise
    gaussian_noise = np.clip(gaussian_noise, 0, 1)
    gaussian_noise_img = np.uint8(gaussian_noise * 255)

    # size_ = 5
    # gaussian_noise_img = gaussian_noise_img / 255.0 * 4096.0 / 10 # --> cm
    # img_center = [int(gaussian_noise_img.shape[0] / 2 + 15 + 0.5), int(gaussian_noise_img.shape[1] / 2 + 0.5)]
    # gaussian_noise_img = gaussian_noise_img / 255.0 * 4096.0 / 10 # --> cm
    # # embed()
    # total_depth = np.sum(gaussian_noise_img[img_center[0] - size_ : img_center[0] + size_,
    #                             img_center[1] - size_ : img_center[1] + size_,
    #                             0], dtype=np.float)          
    # mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
    # depth_mask = gaussian_noise_img.copy()
    # # depth_mask = depth_mask[:,:,0]
    # depth_mask[depth_mask > mean_depth_val * 1.15] = 0
    # depth_mask[depth_mask < mean_depth_val * 0.85] = 0
    # depth_mask[depth_mask != 0] = 1 
    # depth_mask = np.array(depth_mask, dtype=np.uint8)
    # cv2.imshow('gau', depth_mask * 255)
    # cv2.waitKey(0)

    return gaussian_noise_img

# 椒盐噪声
def AugSaltNoise(img, s_vs_p=0.5, amount = 0.02):
    """
    :param img: 原图
    :param s_vs_p: 椒盐噪声中椒 ：盐比例
    :param amount: 实施椒盐噪声的元素的数量
    :return:
    """

    img_copy = img.copy()

    # 添加salt噪声
    num_salt = np.ceil(amount * img_copy.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_copy.shape[:2]]
    img_copy[coords] = 255

    # 添加pepper噪声
    num_pepper = np.ceil(amount * img_copy.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_copy.shape[:2]]
    img_copy[coords] = 0
    return img_copy

# 直方图均衡化
def AugHistEqua(img, mask=None, L=256):
    img_copy = img.copy()
    h, w = img_copy.shape[0], img_copy.shape[1]
    hist = cv2.calcHist([img_copy], [0], mask, [256], [0, 255])  # 计算图像的直方图，即存在的每个灰度值的像素点数量
    # plt.plot(hist)
    # plt.show()
    hist[0:255] = hist[0:255] / (h * w)  # 计算灰度值的像素点的概率，除以所有像素点个数，就归一化
    # 设置si
    sum_hist = np.zeros(hist.shape)
    # 开始计算si的一部分值，i每一次增大，si都是对前i个灰度值的分布概率进行累加
    for i in range(256):
        sum_hist[i] = sum(hist[0:i+1])
    equal_hist = np.zeros(sum_hist.shape)
    # si再乘以灰度级，再四舍五入
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    img_equal = img_copy.copy()
    # 新图片的创建
    for i in range(h):
        for j in range(w):
            img_equal[i, j, 0] = equal_hist[img_copy[i, j, 0]]
            img_equal[i, j, 1] = equal_hist[img_copy[i, j, 1]]
            img_equal[i, j, 2] = equal_hist[img_copy[i, j, 2]]
    return img_equal

    # equal_hist = cv2.calcHist([img_equal], [0], mask, [256], [0, 255])
    # plt.plot(equal_hist)
    # plt.show()
    
# 三通道中方图均衡化
def AugHistEqual2(img):
    img_copy = img.copy()
    (b, g, r) = cv2.split(img_copy)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    result = cv2.merge((bH, gH, rH))
    return result

# 图像亮度增强
def AugBright(img, bright_score=0.3):
    img_copy = img.copy()
    # np.random.seed(0)
    # bright_score = np.random.rand(1, 1)[0, 0]
    bright_score = random.uniform(0.15, 0.5)
    # print(bright_score)
    img_copy = exposure.adjust_gamma(img_copy, bright_score)
    return img_copy

# 图像亮度变暗
def AugDark(img, dark_score=2):
    img_copy = img.copy()
    dark_score = random.uniform(1.5, 3.5)
    img_copy = exposure.adjust_gamma(img_copy, dark_score)
    return img_copy

def AugHSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cv2.imshow('hsv', img_hsv)
    cv2.waitKey(0)
    
def AugGray(img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(img_gray.shape)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)
    img_one_channel = cv2.split(img)[1]
    img_gray = cv2.merge((img_one_channel, img_one_channel, img_one_channel))
    cv2.imshow('gray', img_gray)
    cv2.waitKey(0)

def main(args, base_dir, prob_list):
    meta_data= read_json(args.json2_path)  #  增强后的文件路径
    aug_bg = read_json(args.aug_json)  # 背景路径
    img_provider = ImageReaderWoTrans(args.json1_path, args)
    
    first = 1
    for idx, (img, depth, img_name, depth_name, emg_name, root) in enumerate(img_provider):
        aug_path = None
        aug_img = img.copy()

        aug_img_name = None
        img_name_split = img_name.split("/")
        aug_img_name = os.path.join(img_name_split[0], img_name_split[1], \
                img_name_split[2], base_dir, img_name_split[-1]) 

        if first == 1:      
            aug_path = os.path.join(root, img_name_split[0], img_name_split[1], \
                        img_name_split[2], base_dir)
            ensure_dir(aug_path)
            print("have created a new img dir ..............")
            first = 0

        # 混合增强的概率
        # segment_hand, AugGaussianNoise, AugSaltNoise, AugHistEqual2, AugBright, AugDark
        probs = [random.random() for i in range(6)]

        flag1 = False
        flag2 = False
        flag3 = False

        # aug_mask
        if probs[0] > prob_list[0]:
            selec_idx = random.randint(1, len(aug_bg['data']) - 10)
            bg = cv2.imread(os.path.join(args.aug_root, aug_bg['data'][selec_idx]))
            bg = process_bg(bg)
            aug_img = segment_hand(aug_img, depth, bg, 1, (160, 160), 15)

        # AugHistEqual2
        if probs[1] > prob_list[1]:
            aug_img = AugHistEqual2(aug_img)
            flag1 = True
            flag2 = True

        # AugBright
        if not flag1 and probs[2] > prob_list[2]:
            aug_img = AugBright(aug_img)
            flag2 = True
        
        # AugDark
        if not flag2 and probs[3] > prob_list[3]:
            aug_img = AugDark(aug_img)

        # AugGaussian
        if probs[4] > prob_list[4]:
            aug_img = AugGaussianNoise(aug_img)
            flag3 = True

        # AugSalt
        if not flag3 and probs[5] > prob_list[5]:
            aug_img = AugSaltNoise(aug_img)

        cv2.imwrite(os.path.join(root, aug_img_name), aug_img)
        meta_data["data"].append([aug_img_name, depth_name, emg_name, args.gesture_num, args.person_num])  # RGB 
        # meta_data["data"].append([img_name, aug_img_name, emg_name, args.gesture_num, args.person_num])  # depth 
        # cv2.imshow("aug_img", aug_img)
        # cv2.waitKey(1000)
        print(f'working {idx + 1}/{len(img_provider)}') 

    write_json(args.json2_path, meta_data)     
           

if __name__ == "__main__":
    # ------------------------------------------------------------------------------
    gesture_num = "15"
    person_num = "00"    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default = "")
    parser.add_argument('--gesture_num', default = gesture_num)
    parser.add_argument('--person_num', default = person_num)
    parser.add_argument('--hand_type', default = 'right')
    parser.add_argument('--relative_path', default="")
    # /media/xuchengjun/disk1/zx/tmp/HAND/right
    # "/media/xuchengjun/disk1/zx/right/HAND", gesture_num, person_num, "right.json"
    parser.add_argument('--json1_path', default=os.path.join("/media/xuchengjun/disk1/zx/right/HAND", gesture_num, person_num, "right.json"))  # 原来的
    parser.add_argument('--json1_root', default='/media/xuchengjun/disk1/zx/right')


    base_dir = "aug_data"
    MIX = 1
    # "/media/xuchengjun/disk1/zx/right/HAND", gesture_num, person_num + "_" + base_dir + ".json"
    parser.add_argument('--json2_path', default=os.path.join("/media/xuchengjun/disk1/zx/right/HAND", gesture_num, person_num + "_" + base_dir + ".json"))   # 新的，会将增强后的数据保存起来
    parser.add_argument('--aug_json', default='/media/xuchengjun/disk/datasets/bg.json')     # aug json data
    parser.add_argument('--aug_root', default="/media/xuchengjun/disk/datasets")
    args = parser.parse_args()
    
    output_json = dict()
    output_json['data'] = []
    write_json(args.json2_path, output_json)
    print("have created a new json file ................")

    # segment_hand, AugHistEqual2, AugBright, AugDark, AugGaussianNoise, AugSaltNoise
    prob_list = [0.3, 0.7, 0.7, 0.7, 0.6, 0.6]              
 
    main(args, base_dir, prob_list)
    