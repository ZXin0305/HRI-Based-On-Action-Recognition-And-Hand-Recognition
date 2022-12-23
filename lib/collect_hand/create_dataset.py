from curses import delay_output
from email.policy import default
import sys
from turtle import color, delay
sys.path.append('/home/xuchengjun/ZXin/smap')
from unicodedata import is_normalized
from exps.stage3_root2.config import cfg
import rospy
from lib.utils.camera_wrapper import *
from time import time
import cv2
import pandas as pd
import argparse
from lib.utils.tools import *
import os


# def main():
#     rospy.init_node('multi_modal_sub', anonymous=True)
#     frame_provider = DatasetWrapper("/kinectSDK/color", cfg, 1)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root_path', default = "/media/xuchengjun/disk/datasets/HAND")
#     parser.add_argument('--color_name', default = "rgb")
#     parser.add_argument('--depth_name', default = "depth")
#     parser.add_argument('--emg_name', default = "emg")
#     parser.add_argument('--person_num', type = str, default = "00")
#     parser.add_argument('--gesture_num', type = str, default = "00")
#     parser.add_argument('--hand_type', type = str, default = "right")
#     parser.add_argument('--store_name', type = str, default = "r")

#     args = parser.parse_args()
    
#     data_dir = os.path.join(args.root_path, args.hand_type, args.gesture_num, args.person_num)
#     color_dir = os.path.join(data_dir, args.color_name)
#     depth_dir = os.path.join(data_dir, args.depth_name)
#     emg_dir = os.path.join(data_dir, args.emg_name)

#     ensure_dir(data_dir)
#     ensure_dir(color_dir)
#     ensure_dir(depth_dir)
#     ensure_dir(emg_dir)
    
#     rate = rospy.Rate(60)
#     count = 169

#     color_file = os.path.join(color_dir, (args.person_num + "_" + args.gesture_num + "_" + args.store_name + "r"))
#     depth_file = os.path.join(depth_dir, (args.person_num + "_" + args.gesture_num + "_" + args.store_name + "d"))
#     emg_file = os.path.join(emg_dir, (args.person_num + "_" + args.gesture_num + "_" + args.store_name + "e"))

#     begin_st = time()
#     recore_time = 20   # 真正的recored_time 是这两个参数的差
#     delay_time = 3
#     tmp = delay_time
#     tt = 35
#     non_useful = 0
#     while not rospy.is_shutdown():
#         for (img, depth, emg_list) in frame_provider:
#             # if len(emg_list) != 0:
#             #     print(len(emg_list))
#             place_time = time() - begin_st
#             # print(place_time)
#             if place_time >= delay_time:
#                 save_color = color_file + "_" + str(count) + ".jpg"  #  eg. 00_00_rr_0.jpg   --> person_num, gesture_num, 'r' -- right hand, 'r' -- rgb, 0 -- img nums
#                 save_depth = depth_file + "_" + str(count) + ".png"
#                 # save_emg = emg_file + "_" + str(count) + ".csv"

#                 # save
#                 # if len(emg_list) <= 3:
#                 #     non_useful += 1
#                 #     pass
#                 # else:
#                 #     emg_numpy = np.array(emg_list)
#                 #     pd.DataFrame(emg_numpy).to_csv(save_emg)
#                 #     cv2.imwrite(save_color, img)
#                 #     cv2.imwrite(save_depth, depth)
#                 #     count += 1
#                 # rate.sleep()

#                 cv2.imwrite(save_color, img)
#                 cv2.imwrite(save_depth, depth)
#                 count += 1
#                 if place_time > recore_time:
#                     print(f'current_count is: {count} .....................')
#                     print(non_useful)
#                     return
#             else:
#                 cv2.putText(img,"time: {:.0f}".format(tmp),(int(500),int(500)),cv2.ACCESS_MASK,10,(0,0,255),10)
#                 tmp = delay_time - (time() - begin_st)
#             cv2.imshow('img', img)
#             # cv2.imshow('depth', depth)
#             key = cv2.waitKey(tt)
#             if key == 27:
#                 break


def main():
    rospy.init_node('multi_modal_sub', anonymous=True)
    frame_provider = DatasetWrapper("/kinectSDK/color", cfg, 1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default = "/media/xuchengjun/disk/datasets/HAND")
    parser.add_argument('--color_name', default = "rgb")
    parser.add_argument('--depth_name', default = "depth")
    parser.add_argument('--person_num', type = str, default = "04")
    parser.add_argument('--gesture_num', type = str, default = "02")


    args = parser.parse_args()
    
    data_dir = os.path.join(args.root_path, args.gesture_num, args.person_num)
    color_dir = os.path.join(data_dir, args.color_name)
    depth_dir = os.path.join(data_dir, args.depth_name)

    ensure_dir(data_dir)
    ensure_dir(color_dir)
    ensure_dir(depth_dir)
    
    rate = rospy.Rate(60)
    count = 0
    color_file = os.path.join(color_dir, (args.gesture_num + "_" + args.person_num + "_" + "r"))
    depth_file = os.path.join(depth_dir, (args.gesture_num + "_" + args.person_num + "_" + "d"))

    begin_st = time()
    recore_time = 1000  # 真正的recored_time 是这两个参数的差
    delay_time = 5
    tmp = delay_time
    tt = 12  # 1 15
    non_useful = 0
    while not rospy.is_shutdown():
        for (img, depth) in frame_provider:
            place_time = time() - begin_st
            if place_time >= delay_time:
                save_color = color_file + "_" + str(count) + ".jpg"  #  eg. 00_00_r_0.jpg  
                save_depth = depth_file + "_" + str(count) + ".png"

                # cv2.imwrite(save_color, img)
                # cv2.imwrite(save_depth, depth)
                count += 1
                if place_time > recore_time:
                    print(f'current_count is: {count} .....................')
                    print(non_useful)
                    return
            else:
                cv2.putText(img,"time: {:.0f}".format(tmp),(int(500),int(500)),cv2.ACCESS_MASK,10,(0,0,255),10)
                tmp = delay_time - (time() - begin_st)
            cv2.imshow('img', img)
            cv2.imshow('depth', depth)
            key = cv2.waitKey(tt)
            if key == 27:
                break
            
                

if __name__ == "__main__":
    main()