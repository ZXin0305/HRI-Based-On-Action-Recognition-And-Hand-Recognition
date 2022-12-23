import sys
from unicodedata import is_normalized
# sys.path.append('/home/xuchengjun/ZXin/smap')
# sys.path.append('/home/xuchengjun/catkin_ws/src')
from exps.stage3_root2.config import cfg
import rospy
from lib.utils.camera_wrapper import *
from time import time
import cv2

if __name__ == "__main__":
    rospy.init_node('multi_modal_sub', anonymous=True)
    # frame_provider = DatasetWrapper("/kinectSDK/color", cfg, 1)
    frame_provider = MultiModalSubV2("/kinectSDK/color", cfg, 1)

    rate = rospy.Rate(60)
    st = time()
    count = 0
    total = 0
    while not rospy.is_shutdown():
        for (img, img_trans, scales, depth, accept_flag) in frame_provider:
            print(accept_flag)
            