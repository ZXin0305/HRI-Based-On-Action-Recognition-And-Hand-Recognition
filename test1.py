#!/usr/bin/env python
import cv2
import ctypes
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import sys
sys.path.append('/home/xuchengjun/catkin_ws/src') 
from ros_myo_cpp.msg import EmgArray

# img = cv2.imread("/home/xuchengjun/ZXin/00_08_00015350.jpg")
# h, w, c = img.shape
# if c > 1:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_data = np.array(img, dtype=np.uint8)
# img_data = img_data.ctypes.data_as(ctypes.c_char_p)

# # cv2.imshow("img", img)
# # cv2.waitKey(0)

# # size_of_initial_crop = (200,200)
# # hand_center = (540, 960)
# test_pyc = ctypes.cdll.LoadLibrary("./lib/process_cpp/test_1.so")
# test_pyc.cdraw_rectangle(50, 55, (540, 960))

# depth = cv2.imread("./depth_and_rgb/cropped_depth_25.png")

# depth_mask = depth.copy()
# depth_mask[depth_mask > 35] = 0
# depth_mask[depth_mask < 10] = 0
# depth_mask[depth_mask != 0] = 1
# img = cv2.imread('./depth_and_rgb/cropped_img_25.jpg')
# # img_new = depth_mask * img
# # cv2.imshow('img_new', img_new)
# cv2.imshow('depth_mask', depth_mask * 255)
# cv2.waitKey(0)
# # embed()

# img = cv2.imread('./depth_and_rgb/cropped_img_0.jpg')
# img_copy = img.copy()
# cv2.imshow('img', img)
# img_shape = img.shape
# img_center = [int(img_shape[0] / 2 + 0.5), int(img_shape[1] / 2 + 0.5)]
# depth = cv2.imread("./depth_and_rgb/cropped_depth_0.png")
# depth = depth / 255.0 * 4096.0 / 10  # --> cm
# total_depth = 0.0
# for i in range(img_center[0] - 3, img_center[0] + 3):
#     for j in range(img_center[1] - 3, img_center[1] + 3):
#         total_depth += depth[i,j]

# mean_depth_val = total_depth / 36
# depth_mask = depth.copy()
# depth_mask = depth_mask[:,:,0]

# depth_mask[depth_mask > mean_depth_val[0] * 1.18] = 0
# depth_mask[depth_mask < mean_depth_val[0] * 0.82] = 0
# depth_mask[depth_mask != 0] = 1

# for i in range(3):
#     img_copy[:,:,i] = depth_mask * img_copy[:,:,i]

# cv2.imshow('depth_mask', depth_mask)
# cv2.imshow('img_new', img_copy)
# cv2.waitKey(0)

# 细化
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (1, 1, img.shape[1], img.shape[0])
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
# img = img * mask2[:, :, np.newaxis]

# cv2.imshow('img_copy', img)
# cv2.waitKey(0)

# plt.subplot(121)
# plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)) 
# plt.title("grabcut")
# plt.xticks([])
# plt.yticks([])
# plt.show()


# if __name__ == "__main__":
#     img = cv2.imread("/home/xuchengjun/Desktop/hand.jpg")
#     OLD_IMG = img.copy()
#     # print(img.shape[0], img.shape[1])
#     mask = np.zeros(img.shape[:2], np.uint8)

#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)

#     rect = (1, 1, img.shape[1], img.shape[0])
#     print("123 ..")
#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50, cv2.GC_INIT_WITH_RECT)

#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
#     img = img * mask2[:, :, np.newaxis]

#     plt.subplot(121)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("grabcut")
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(122)
#     plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
#     plt.title("original")
#     plt.xticks([])
#     plt.yticks([])

#     plt.show()

# zz = [1, 2, 3, 4, 5, 6, 7]
# xx = np.array([1, 2, 3, 4, 5, 6, 7])
# idx = np.where(xx <= 4)
# need = xx[idx]
# print(idx[0])
# xx = np.delete(xx, idx)
# del zz[0:len(idx[0])]
# print(zz)


# xx = np.array([])
# yy = [[1, 2]]
# zz = [[3, 4]]
# xx = np.append(xx, yy)
# xx = np.append(xx, zz)
# xx = np.insert(xx, yy)
# print(xx)

color = None
color_convert = None
depth = None
emg_list = []
emg_header = []
res_emg = None
img_header = None
cv_bridge = CvBridge()
is_accept = False
def callback1(msg_color, msg_depth):
    global color_convert, img_header, is_accept, depth
    color = cv_bridge.imgmsg_to_cv2(msg_color, 'rgba8')
    color_convert = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)  # 这里是从 RGBA到RGB是因为在cpp端转化成ROS信息的时候就变成了  'rgba8'
    img_header = msg_color.header.stamp.to_sec()
    print(img_header)
    is_accept = True
    depth = cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')

def callback2(msg_depth):
    global depth
    depth = cv_bridge.imgmsg_to_cv2(msg_depth, 'mono8')


def callback3(msg_emg):
    global is_accept, res_emg, emg_list, emg_header
    emg_list.append(list(msg_emg.data))
    emg_header.append(msg_emg.header.stamp.to_sec())
    if is_accept:
        is_accept = False
        idx = np.where(np.array(emg_header) <= img_header)
        res_emg = np.array(emg_list)[idx]  # --> array
        del emg_list[0:len(idx[0])]
        del emg_header[0:len(idx[0])]
        # print(len(res_emg))

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    color_sub = message_filters.Subscriber("/kinectSDK/color", Image)  # 对齐时间戳
    depth_sub = message_filters.Subscriber("/kinectSDK/depth", Image)
    syc = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=20, slop=0.1)
    syc.registerCallback(callback1)

    rospy.Subscriber("/myo_raw/myo_emg", EmgArray, callback3)
    while not rospy.is_shutdown():
        #此处添加另外一个线程的代码
        if color_convert is not None and depth is not None:
            cv2.imshow("color", color_convert)
            cv2.imshow("depth", depth)
            cv2.waitKey(1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()