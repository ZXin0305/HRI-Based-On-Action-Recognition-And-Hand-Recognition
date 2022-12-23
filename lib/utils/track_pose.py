import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import numpy as np
from track import *
from IPython import embed
from lib.utils.one_euro_filter import OneEuroFilter

color_list = [(255,0,255),(255,255,0),(0,255,255),(0,200,200),(100,0,200),(150,250,200),
              (120,120,40),(200,255,120),(155,120,0),(150,200,200),(255,255,170),(180,147,250),
              (255,120,40),(170,165,110),(200,120,200),(200,200,200),(180,255,104),(100,200,100),
              (70,80,90),(255,200,137),(98,30,24),(100,200,255),(165,168,200),(240,125,250)]
len_color_list = len(color_list)

def pose_color(human_id, max_human):
    color = (200,200,0)
    if human_id >= 0 and human_id < max_human and human_id < len_color_list: 
        color = color_list[human_id]
    return color

class HumanPoseID:
    num_kpts = 15
    def __init__(self, human_pose, human_id):
        self.human_id = human_id
        self.human_pose = self.change_pose_to_mat(human_pose)
        self.filter = [[OneEuroFilter(),OneEuroFilter(),OneEuroFilter()] for _ in range(HumanPoseID.num_kpts)]
        self.filter_2d = [[OneEuroFilter(),OneEuroFilter()] for _ in range(HumanPoseID.num_kpts)]

    def change_pose_to_mat(self, human_pose):
        pose_mat = np.zeros(shape=(3,15,15),dtype=np.float)
        for i in range(3):
            row, col = np.diag_indices_from(pose_mat[i])
            pose_mat[i][row, col] = human_pose[:, i]
        return pose_mat

# def track_pose(last_pose_list, current_pose_list, last_id, thres=0.75, max_human = 10):
#     if len(last_pose_list) >= len(current_pose_list):  # 前面一帧的姿态大于或者等于当前帧的
#         is_occupy = [0] * len(last_pose_list)
#         for current_human_pose in current_pose_list:
#             similar_list = []
#             for i, last_human_pose in enumerate(last_pose_list):
#                 dis_mat = current_human_pose.human_pose - last_human_pose.human_pose
#                 numera = np.sum(dis_mat ** 2, axis=(1, 2))
#                 denom = np.sum(last_human_pose.human_pose ** 2, axis=(1, 2))
#                 similar = np.sum([1.0,1.0,1.0] - (numera / denom)) / 3 # 一共三个通道 x/y/z
#                 similar_list.append((i,similar))
#             similar_list.sort(key=lambda x: x[1],reverse=True)         
#             if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
#                 is_occupy[similar_list[0][0]] = 1
#                 current_human_pose.human_id = last_pose_list[similar_list[0][0]].human_id  #注意这里，如果是前面一帧的姿态大于当前帧的，那么这里的是直接用对应的current_human_pose
#     elif len(last_pose_list) < len(current_pose_list): # 前面一帧的姿态小于当前帧的
#         is_occupy = [0] * len(current_pose_list)
#         for last_human_pose in last_pose_list:
#             similar_list = []
#             for i, current_human_pose in enumerate(current_pose_list):
#                 dis_mat = current_human_pose.human_pose - last_human_pose.human_pose
#                 numera = np.sum(dis_mat ** 2, axis=(1, 2))
#                 denom = np.sum(last_human_pose.human_pose ** 2, axis=(1, 2))
#                 similar = np.sum([1.0,1.0,1.0] - (numera / denom)) / 3
#                 similar_list.append((i,similar))
#             similar_list.sort(key=lambda x: x[1],reverse=True)  #利用相似度进行排序
#             #选取相似度最大的进行排序
#             if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
#                 is_occupy[similar_list[0][0]] = 1
#                 current_pose_list[similar_list[0][0]].human_id = last_human_pose.human_id 
#     # 可以匹配的已经完成了匹配
#     # 对于未匹配的，赋予全新的id， 不过是根据last_id
#     for current_human_pose in current_pose_list:
#         if current_human_pose.human_id == -1 and last_id < max_human:
#             current_human_pose.human_id = last_id
#             last_id += 1          
#     return last_id

class TRACK:
    def __init__(self):
        pass
    def track_pose(self, consec_list, last_pose_list, current_pose_list, last_id, thres=0.75, max_human = 10):
        have_paired = []
        #姿态列表中的数量小于最大搜索帧
        #当前帧中的值会与上一帧中的序列进行逐一匹配
        is_occupy = [0] * len(last_pose_list)
        for current_human_pose in current_pose_list:
            similar_list = []  #每次循环都初始化这个相似性列表
            for i, last_human_pose in enumerate(last_pose_list):
                similar = self.cal_sim(last_human_pose.human_pose, current_human_pose.human_pose)
                similar_list.append((i,similar))

            similar_list.sort(key=lambda x: x[1],reverse=True)

            if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
                is_occupy[similar_list[0][0]] = 1
                current_human_pose.human_id = last_pose_list[similar_list[0][0]].human_id

                # if current_human_pose.human_id == 1:
                #     print(similar_list)

                current_human_pose.filter = last_pose_list[similar_list[0][0]].filter
                current_human_pose.filter_2d = last_pose_list[similar_list[0][0]].filter_2d
                have_paired.append(current_human_pose.human_id)  #当前帧中的这个已经匹配了
                                                                 #告诉后面搜索的范围排除掉这些已经经过匹配的  
        # 进入姿态列表中进行向前搜索
        for current_human_pose in current_pose_list:
            if current_human_pose.human_id == -1:  #如果当前帧中还存在未匹配的帧数，那么就会进行向前搜索
                frame_ptr = -2                    #定义搜索的帧，值会越来越小

                #搜索的主体
                while frame_ptr >= -len(consec_list):
                    last_pose_list = consec_list[frame_ptr] #更新上一帧的姿态列表元素
                    is_occupy = [0] * len(last_pose_list)
                    similar_list = []  #对当前的id仍为-1的进行搜索
                    for i, last_human_pose in enumerate(last_pose_list):
                        tmp_id = last_human_pose.human_id
                        if tmp_id in have_paired:  #如果匹配过了，会继续与last中别的姿态进行匹配
                            is_occupy[i] = 1
                            continue
                        similar = self.cal_sim(last_human_pose.human_pose, current_human_pose.human_pose)
                        similar_list.append((i,similar))

                    #如果last中的所有姿态都是已经匹配了的，那么similar_list的长度就会为0
                    #此时就直接往前搜索了
                    if len(similar_list) > 0:
                        similar_list.sort(key=lambda x: x[1],reverse=True)
                        
                        if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
                            is_occupy[similar_list[0][0]] = 1
                            current_human_pose.human_id = last_pose_list[similar_list[0][0]].human_id
                            current_human_pose.filter = last_pose_list[similar_list[0][0]].filter
                            current_human_pose.filter_2d = last_pose_list[similar_list[0][0]].filter_2d
                            have_paired.append(current_human_pose.human_id)     
                    frame_ptr -= 1 
                    
        for current_human_pose in current_pose_list:
            if current_human_pose.human_id == -1 and last_id < max_human:
                current_human_pose.human_id = last_id
                last_id += 1
        return last_id

    def track_pose2(self, consec_list, last_pose_list, current_pose_list, last_id, thres=0.75, max_human = 10):

        """
        Form 1 in tracking process
        """
        have_paired = []
        is_occupy = [0] * len(last_pose_list)
        similar_list = []
        for i, current_human_pose in enumerate(current_pose_list):
            for j, last_human_pose in enumerate(last_pose_list):
                similar = self.cal_sim(last_human_pose.human_pose, current_human_pose.human_pose)
                similar_list.append((i,j,similar))
        # 全部配对计算结束之后再进行排序，这样可以进行全面的匹配
        similar_list.sort(key=lambda x: x[2], reverse=True)

        tmp1 = []  # 当当前姿态列表中的某一个姿态对上一帧中多个姿态都有相似度时，需要锁定当前姿态，只能匹配一个
        for i in range(len(similar_list)):
            sim = similar_list[i][2]
            if sim >= thres and is_occupy[similar_list[i][1]] == 0:
                if similar_list[i][0] not in tmp1:
                    is_occupy[similar_list[i][1]] = 1
                    tmp1.append(similar_list[i][0])
                    current_human_pose = current_pose_list[similar_list[i][0]]
                    last_human_pose = last_pose_list[similar_list[i][1]]
                    current_human_pose.human_id = last_human_pose.human_id
                    current_human_pose.filter = last_human_pose.filter
                    current_human_pose.filter_2d = current_human_pose.filter_2d
                    have_paired.append(current_human_pose.human_id)
        """
        Form 2 in tracking process
        """
        # 进入姿态列表中向前搜索
        frame_ptr = -2 
        while frame_ptr >= -len(consec_list):
            last_pose_list = consec_list[frame_ptr]
            is_occupy = [0] * len(last_pose_list)
            similar_list = []

            # 全部配对计算结束之后再进行排序，这样可以进行全面的匹配
            for i, current_human_pose in enumerate(current_pose_list):
                if current_human_pose.human_id == -1:
                    for j, last_human_pose in enumerate(last_pose_list):
                        tmp_id = last_human_pose.human_id
                        if tmp_id in have_paired:
                            is_occupy[j] = 1
                            continue
                        similar = self.cal_sim(last_human_pose.human_pose, current_human_pose.human_pose)
                        similar_list.append((i,j,similar))

            if len(similar_list) > 0:
                similar_list.sort(key=lambda x: x[2], reverse=True)
                tmp2 = []
                for i in range(len(similar_list)):
                    sim = similar_list[i][2]
                    if sim >= thres and is_occupy[similar_list[i][1]] == 0:
                        if similar_list[i][0] not in tmp2:
                            is_occupy[similar_list[i][1]] = 1
                            tmp2.append(similar_list[i][0])
                            current_human_pose = current_pose_list[similar_list[i][0]]
                            last_human_pose = last_pose_list[similar_list[i][1]]
                            current_human_pose.human_id = last_human_pose.human_id
                            current_human_pose.filter = last_human_pose.filter
                            current_human_pose.filter_2d = current_human_pose.filter_2d
                            have_paired.append(current_human_pose.human_id)
            frame_ptr -= 1

        """
        Form 3 in tracking process
        """
        for current_human_pose in current_pose_list:
            if current_human_pose.human_id == -1 and last_id < max_human:
                current_human_pose.human_id = last_id
                last_id += 1
        return last_id

    def cal_sim(self, last_human_pose, current_human_pose, sigma=2): 
        #计算相似性
        dis_mat = current_human_pose - last_human_pose
        numera = np.sum(dis_mat ** 2, axis=(1, 2))
        denom = np.sum(last_human_pose ** 2, axis=(1, 2))
        similar = np.exp( (-1) * np.sum(numera / denom) / sigma** 2) # 一共三个通道 x/y/z
        # similar = np.sum([1.0,1.0,1.0] - (numera / denom)) / 3 # 一共三个通道 x/y/z
        # temp = [1.0,1.0,1.0] - (numera / denom)
        # similar = 0.2 * temp[0] + 0.2 * temp[1] + 0.6 * temp[2]  
        return similar


def track_pose(consec_list, last_pose_list, current_pose_list, last_id, thres=0.75, max_human = 10):

    have_paired = []
    #姿态列表中的数量小于最大搜索帧
    #当前帧中的值会与上一帧中的序列进行逐一匹配
    is_occupy = [0] * len(last_pose_list)
    for current_human_pose in current_pose_list:
        similar_list = []  #每次循环都初始化这个相似性列表
        for i, last_human_pose in enumerate(last_pose_list):

            #计算相似性
            dis_mat = current_human_pose.human_pose - last_human_pose.human_pose
            numera = np.sum(dis_mat ** 2, axis=(1, 2))
            denom = np.sum(last_human_pose.human_pose ** 2, axis=(1, 2))
            similar = np.sum([1.0,1.0,1.0] - (numera / denom)) / 3 # 一共三个通道 x/y/z
            # temp = [1.0,1.0,1.0] - (numera / denom)
            # similar = 0.2 * temp[0] + 0.2 * temp[1] + 0.6 * temp[2]
            similar_list.append((i,similar))

        similar_list.sort(key=lambda x: x[1],reverse=True)
        print(similar_list)
        if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
            is_occupy[similar_list[0][0]] = 1
            current_human_pose.human_id = last_pose_list[similar_list[0][0]].human_id
            have_paired.append(current_human_pose.human_id)  #当前帧中的这个已经匹配了
                                                             #告诉后面搜索的范围排除掉这些已经经过匹配的
        
    # 进入姿态列表中进行向前搜索
    for current_human_pose in current_pose_list:
        if current_human_pose.human_id == -1:  #如果当前帧中还存在未匹配的帧数，那么就会进行向前搜索
            frame_ptr = -3                     #定义搜索的帧，值会越来越小

            #搜索的主体
            while frame_ptr >= -len(consec_list):
                last_pose_list = consec_list[frame_ptr] #更新上一帧的姿态列表元素
                is_occupy = [0] * len(last_pose_list)
                similar_list = []  #对当前的id仍为-1的进行搜索
                for i, last_human_pose in enumerate(last_pose_list):
                    tmp_id = last_human_pose.human_id
                    if tmp_id in have_paired:  #如果匹配过了，会继续与last中别的姿态进行匹配
                        is_occupy[i] = 1
                        continue
                    #相似性
                    dis_mat = current_human_pose.human_pose - last_human_pose.human_pose
                    numera = np.sum(dis_mat ** 2, axis=(1, 2))
                    denom = np.sum(last_human_pose.human_pose ** 2, axis=(1, 2))
                    similar = np.sum([1.0,1.0,1.0] - (numera / denom)) / 3 # 一共三个通道 x/y/z
                    # temp = [1.0,1.0,1.0] - (numera / denom)
                    # similar = 0.2 * temp[0] + 0.2 * temp[1] + 0.6 * temp[2]
                    similar_list.append((i,similar))

                #如果last中的所有姿态都是已经匹配了的，那么similar_list的长度就会为0
                #此时就直接往前搜索了
                if len(similar_list) > 0:
                    similar_list.sort(key=lambda x: x[1],reverse=True)
                    
                    if similar_list[0][1] >= thres and is_occupy[similar_list[0][0]] == 0:  # 相似度 和 是否已经匹配
                        is_occupy[similar_list[0][0]] = 1
                        current_human_pose.human_id = last_pose_list[similar_list[0][0]].human_id
                        have_paired.append(current_human_pose.human_id)     
                frame_ptr -= 1


    for current_human_pose in current_pose_list:
        if current_human_pose.human_id == -1 and last_id < max_human:
            current_human_pose.human_id = last_id
            last_id += 1
                
    return last_id

if __name__ == "__main__":
    #这里是得到的每次的pose列表 --> numpy
    #运行demo的时候，这里记得
    last_id = 0
    thres = 0.75
    frame = 1
    last_frame_human = [test, last_frame]  #这两个到时候只用到一个就可以了  new_pred_bodys_3d
    current_frame_human = [first_frame, test_1]

    last_pose_list = []  #这个是作为全局变量的,在进行所有的排序之后，用current_pose_list进行更新
    current_pose_list = [] #这个是用来作为局部变量的,会在每一次循环的时候先进行更新

    if frame == 0:
        for i in range(len(current_frame_human)):
            human = HumanPoseID(current_frame_human[i], i)
            current_pose_list.append(human)
            last_id += 1  
        last_pose_list = current_pose_list      
    else:

        for i in range(len(last_frame_human)):
            human = HumanPoseID(last_frame_human[i], i)
            last_pose_list.append(human)
            last_id += 1

        for i in range(len(current_frame_human)):
            human = HumanPoseID(current_frame_human[i], -1)
            current_pose_list.append(human)

        track_pose(last_pose_list, current_pose_list, last_id, thres)
        last_pose_list = current_pose_list