"""
using for refine
"""
import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import json
import os.path as osp
from path import Path
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from IPython import embed
from lib.utils.tools import read_json


class P2PDataset(Dataset):
    def __init__(self, stage='train', dataset_path='', root_idx=2):
        self.root_idx = root_idx
        self.dataset_path = dataset_path
        if len(dataset_path) == 0:
            print('please input the refine-dataset-path !!')
            return
        self.data_list = Path(self.dataset_path).files()

    def __getitem__(self, index):
        json_path = self.data_list[index]
        json_file = read_json(json_path)
        input_point_3d = np.asarray(json_file['pred_3d'], dtype=np.float)
        input_point_2d = np.asarray(json_file['pred_2d'], dtype=np.float)
        gt_point_3d = np.asarray(json_file['gt_3d'], dtype=np.float)

        input_point = np.zeros((15, 5), dtype=np.float)  
        gt_point = np.zeros((15, 3), dtype=np.float) 

        # print(gt_point_3d[2])
        # root ..
        gt_point[self.root_idx] = 0  # relative to the root joint
        # gt_point[self.root_idx, 2] = gt_point_3d[self.root_idx, 2] - input_point_3d[self.root_idx, 2]  #差值
        # embed()
        input_point[self.root_idx, :2] = input_point_2d[self.root_idx, :2]
        input_point[self.root_idx, 2:] = input_point_3d[self.root_idx, :3]

        # 关节点数
        for i in range(0, len(input_point_2d)):
            if i != self.root_idx:
                gt_point[i] = gt_point_3d[i] - gt_point_3d[self.root_idx]   # 最终得到的也是相对的值
                if input_point_3d[i, 3] > 0:                                #　只有这个点在预测的时候就有值才会计算这个相对数值，但是这种并没有对root进行修正啊
                    input_point[i, :2] = input_point_2d[i, :2] - input_point_2d[self.root_idx, :2]
                    input_point[i, 2:] = input_point_3d[i, :3] - input_point_3d[self.root_idx, :3]
        
      
        inp = input_point.flatten()
        gt = gt_point.flatten()
        inp = torch.from_numpy(inp).float()
        gt = torch.from_numpy(gt).float()
        return inp, gt
        # return input_point, gt_point

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    dataset = P2PDataset(dataset_path='/media/xuchengjun/disk/datasets/CMU/refine')

    for i in range(len(dataset)):
        print(dataset[i][0])
        # embed()

