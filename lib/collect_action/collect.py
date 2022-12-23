import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
from model.main_model.smap import SMAP
from model.refine_model.refinenet import RefineNet
from exps.stage3_root2.config import cfg
import os
import random
import cv2
from lib.utils.tools import *
import dapalib
import csv
import h5py
import numpy as np
from IPython import embed
# from demo import *

headers = ['neck_x','neck_y','neck_z',
           'nose_x','nose_y','nose_z',
           'BodyCenter_x','BodyCenter_y','BodyCenter_z',
           'lShoulder_x','lShoulder_y','lShoulder_z',
           'lElbow_x','lElbow_y','lElbow_z',
           'lWrist_x','lWrist_y','lWrist_z',
           'lHip_x','lHip_y','lHip_z',
           'lKnee_x','lKnee_y','lKnee_z',
           'lAnkle_x','lAnkle_y','lAnkle_z',
           'rShoulder_x','rShoulder_y','rShoulder_z',
           'rElbow_x','rElbow_y','rElbow_z',
           'rWrist_x','rWrist_y','rWrist_z',
           'rHip_x','rHip_y','rHip_z',
           'rKnee_x','rKnee_y','rKnee_z',
           'rAnkle_x','rAnkle_y','rAnkle_z']

# def pose_normalization(pred_3d_bodys):
#     """[summary]
#     original
#     """

#     for i in range(len(pred_3d_bodys)):
#         origin_x = np.min(pred_3d_bodys[i,:,0])
#         origin_y = np.min(pred_3d_bodys[i,:,1])
#         origin_z = np.min(pred_3d_bodys[i,:,2])
#         max_x = np.max(pred_3d_bodys[i,:,0])
#         max_y = np.max(pred_3d_bodys[i,:,1])
#         max_z = np.max(pred_3d_bodys[i,:,2])
#         len_x = max_x - origin_x
#         len_y = max_y - origin_y
#         len_z = max_z - origin_z
        
#         # pred_3d_bodys[i,:,0] = float(format((pred_3d_bodys[i,:,0] - origin_x) / len_x, "0.3f"))
#         # pred_3d_bodys[i,:,1] = float(format((pred_3d_bodys[i,:,1] - origin_y) / len_y, "0.3f"))
#         # pred_3d_bodys[i,:,2] = float(format((pred_3d_bodys[i,:,2] - origin_z) / len_z, "0.3f"))

#         pred_3d_bodys[i,:,0] = np.round((pred_3d_bodys[i,:,0] - origin_x) / len_x, 3)
#         pred_3d_bodys[i,:,1] = np.round((pred_3d_bodys[i,:,1] - origin_y) / len_y, 3)
#         pred_3d_bodys[i,:,2] = np.round((pred_3d_bodys[i,:,2] - origin_z) / len_z, 3)
        
#     return pred_3d_bodys

# def change_pose(pred_3d_bodys):
#     """[summary]

#     Args:
#         pred_3d_bodys ([type]): [description]
#         not original 
#         可以用flatten

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
        

def pose_normalization(pred_3d_bodys):
    """[summary]
    using the maximum 3D bounding box to normlize the pose in a single frame of one person
    但你开始的数据是以cm为单位进行计算的
    """
    origin_x = np.min(pred_3d_bodys[:,0])  #x
    origin_y = np.min(pred_3d_bodys[:,1])  #y
    origin_z = np.min(pred_3d_bodys[:,2])  #z
    max_x = np.max(pred_3d_bodys[:,0])
    max_y = np.max(pred_3d_bodys[:,1])
    max_z = np.max(pred_3d_bodys[:,2])
    len_x = max_x - origin_x
    len_y = max_y - origin_y
    len_z = max_z - origin_z

    pred_3d_bodys[:,0] = np.round((pred_3d_bodys[:,0] - origin_x) / len_x, 3)
    pred_3d_bodys[:,1] = np.round((pred_3d_bodys[:,1] - origin_y) / len_y, 3)
    pred_3d_bodys[:,2] = np.round((pred_3d_bodys[:,2] - origin_z) / len_z, 3)

    # pred_3d_bodys[:,0] = (pred_3d_bodys[:,0] - origin_x) / len_x
    # pred_3d_bodys[:,1] = (pred_3d_bodys[:,1] - origin_y) / len_y
    # pred_3d_bodys[:,2] = (pred_3d_bodys[:,2] - origin_z) / len_z
        
    return pred_3d_bodys

def vector_pose_normalization(pred_3d_bodys):
    """
    用相对于根节点的姿态，这里也可以进行归一化，但是别只保留三位数了，感觉不太准确
    以m为单位
    """
    pred_3d_bodys = (pred_3d_bodys[2] - pred_3d_bodys) / 100  #化成m
    origin_x = np.min(pred_3d_bodys[:,0])
    origin_y = np.min(pred_3d_bodys[:,1])
    origin_z = np.min(pred_3d_bodys[:,2])

    max_x = np.max(pred_3d_bodys[:,0])
    max_y = np.max(pred_3d_bodys[:,1])
    max_z = np.max(pred_3d_bodys[:,2])   

    len_x = max_x - origin_x
    len_y = max_y - origin_y
    len_z = max_z - origin_z

    pred_3d_bodys[:,0] = (pred_3d_bodys[:,0] - origin_x) / len_x
    pred_3d_bodys[:,1] = (pred_3d_bodys[:,1] - origin_y) / len_y
    pred_3d_bodys[:,2] = (pred_3d_bodys[:,2] - origin_z) / len_z

    return pred_3d_bodys


def change_pose(pred_3d_bodys):
    """[summary]

    Args:
        pred_3d_bodys ([type]): [description]
        not original 

    Returns:
        [type]: [description]
    """ 
    # pose_3d = []
    # for j in range(15):
    #     pose_3d.append(pred_3d_bodys[j][0])  # x
    #     pose_3d.append(pred_3d_bodys[j][1])  # y
    #     pose_3d.append(pred_3d_bodys[j][2])  # z
    pose_3d = pred_3d_bodys.flatten().tolist()
    return pose_3d  


# if __name__ == '__main__':

    # model_path = '/media/xuchengjun/zx/human_pose/pth/main/12.16/train.pth'
    # refine_path = '/media/xuchengjun/zx/human_pose/pth/main/12.16/refine/RefineNet_epoch_200.pth'
    # device = 'cuda:0'
    # image_dir = '/home/xuchengjun/Videos/test'  # image files root dir
    # csv_dir = '/home/xuchengjun/Videos/csv'
    # hy5file_path = '/home/xuchengjun/Videos/final.h5' #绝对路径
    # action_label = '0'
    # total_data_num = 5
    # have_appeared = []
    
    # if not os.path.exists(csv_dir):
    #     os.mkdir(csv_dir)
    #     print('have created a new dir ..')
    
    # model = SMAP(cfg)
    # model.to(device)
    
    # refine_model = RefineNet()
    # refine_model.to(device)
    
    # model.eval()
    # refine_model.eval()
    # print('have loaded model ..')
    # # all_image_list = os.listdir(image_dir)
    # # sort_image_list = all_image_list.sort(key=lambda x:int(x.split('.')[0]))
    # stride = cfg.dataset.STRIDE
    # print('begin ..')
    
    # for i in range(total_data_num):
    #     # start_frame = random.randint(500, 550)
    #     start_frame = 500
    #     print(start_frame)
    #     while True:
    #         if start_frame in have_appeared:
    #             start_frame = random.randint(50, 700)
    #         else:
    #             break
            
    #     have_appeared.append(start_frame)
    #     pose_frame_list = []
        
    #     for j in range(0, 30):
    #         print(f'process img .. {j}')
    #         img_path = os.path.join(image_dir, str(start_frame + j) + '.jpg')
    #         embed()
    #         image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #         img_scale, scales, pad_value = croppad_img(image, cfg)
    #         scales['f_x'] = 1059.95
    #         scales['f_y'] = 1053.93
    #         scales['cx'] = 954.88
    #         scales['cy'] = 523.74
    #         img_trans = transform(img_scale, cfg)
    #         img_trans = img_trans.unsqueeze(0).to(device)
            
    #         with torch.no_grad():
    #             outputs_2d, outputs_3d, outputs_rd = model(img_trans)
    #             outputs_3d = outputs_3d.cpu()
    #             outputs_rd = outputs_rd.cpu()
                
    #             hmsIn = outputs_2d[0]
                
    #             hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255   # keypoint  maps
    #             hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127   # paf maps
    #             rDepth = outputs_rd[0][0]
                
    #             pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)  # --> tensor, shape:(person_num, 15, 4)
                
    #             if pred_bodys_2d.shape[0] == 0:
    #                 print('no human ..')
    #                 zero_pose = [0] * 45
    #                 pose_frame_list.append(zero_pose)
    #                 continue
    #             else:
    #                 pred_bodys_2d[:,:,:2] *= stride
    #                 pred_bodys_2d = pred_bodys_2d.numpy()    
                    
    #             if len(pred_bodys_2d) > 0:
    #                 pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
    #                 root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
    #                 #　upsample the outputs' shape to obtain more accurate results
    #                 #　--> (256, 456)
    #                 paf_3d_upsamp = cv2.resize(
    #                     pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
    #                 root_d_upsamp = cv2.resize(
    #                     root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
    #                 pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales)
    #                 pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, pad_value)
                    
    #             # new_pred_bodys_3d --> numpy()
    #             if refine_model is not None:
    #                 new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
    #                                                             device=device, root_n=cfg.DATASET.ROOT_IDX)
    #             else:
    #                 new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4)  
    #             print(len(new_pred_bodys_3d))  
    #             pose_3d = change_pose(new_pred_bodys_3d)
    #             pose_frame_list.append(pose_3d)
                
    #     csv_file = os.path.join(csv_dir, action_label + '_' + str(i) + '.csv')
        
    #     with open(csv_file, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(headers)
    #         csv_writer.writerows(pose_frame_list)
        
    #     # data = read_csv(csv_file)
    #     # f = h5py.File(hy5file_path, 'w')
    #     # name = action_label + '_' + str(i)
    #     # f[name] = data
        
    # print('done ..')
        
            
if __name__ == "__main__":
    # /media/xuchengjun/datasets/action_zx/4(previous)
    # /media/xuchengjun/datasets/action_zx/data_train.h5
    # /media/xuchengjun/datasets/UTD-MAD/cs2/test_csv
    # /media/xuchengjun/datasets/MSRAction3D/cs2/train_csv
    import random

    # csv_dir =  '/media/xuchengjun/datasets/action_zx/NEW/test_4'   # /media/xuchengjun/datasets/action_zx
    # hy5file_data = '/media/xuchengjun/datasets/action_zx/NEW/data_test.h5'   #不能用纯数字的名字 /media/xuchengjun/datasets/UTD-MAD/cs2/data_test.h5
    # hy5file_label = '/media/xuchengjun/datasets/action_zx/NEW/label_test.h5'
    
    csv_dir =  '/media/xuchengjun/datasets/MSRAction3D/cs2/test_csv'   # /media/xuchengjun/datasets/action_zx  
    hy5file_data = '/media/xuchengjun/datasets/MSRAction3D/cs2/data_test.h5'   #不能用纯数字的名字 /media/xuchengjun/datasets/UTD-MAD/cs2/data_test.h5
    hy5file_label = '/media/xuchengjun/datasets/MSRAction3D/cs2/label_test.h5'
    csv_list = os.listdir(csv_dir)
    random.shuffle(csv_list)

    # csv_list.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
    f_3 = h5py.File(hy5file_data, 'w')
    f_4 = h5py.File(hy5file_label, 'w')
    print(f'process total num : {len(csv_list)}')
    # embed()
    num = 1
    for csv_file in csv_list:
        csv_name = csv_file.split('.')[0]
        path = os.path.join(csv_dir, csv_file)
        data = read_csv(path)
        f_3[csv_name] = data                    #数据
        f_4[csv_name] = csv_name.split('_')[0]  #标签
        if num == 1:
            print(f"label --> {csv_name.split('_')[0]}")
        print(f'working .. {len(csv_list)} / {num}')
        num += 1
        # if num>=2000:
        #     break
    f_3.close()
    f_4.close()
   
   

                                             
            
        
        
    
    
    
    