import copy
import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import cv2
import json
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
import random
from path import Path
from dataset.ImageAugmentation import (aug_croppad, aug_croppad_for_test, aug_flip, aug_rotate)
from dataset.representation import generate_heatmap, generate_paf, generate_rdepth, generate_rdepth_map
from lib.utils.tools import read_json
from IPython import embed
from exps.stage3_root2.config import cfg


class JointDataset(Dataset):
    def __init__(self, cfg, stage, transform=None, with_augmentation=False, with_mds=False):

        self.stage = stage
        self.data_list = []
        """
        train: provide training data for training
        test: provide test data for test
        generation: provide training data for inference --> the input to RefineNet
        """
        assert self.stage in ('train', 'test', 'generation')

        self.transform = transform  #判断是否进行转换
        # self.train_data = list()
        # self.val_data = list()
        self.cfg = cfg
        DATASET = cfg.dataset

        # ----------- get data list --------------
        if self.stage == 'train':
            print(f'current mode is {self.stage}, loading {self.stage} dataset ..')
            # choose coco + specific 3d dataset for training together

            data = []
            _3d_data_list = []
            # for convenience, save json for single img format ..
            data = Path(DATASET.COCO_JSON_PATH).files()      # len(data) 
            print(f'using COCO dataset --> {DATASET.COCO_JSON_PATH}')
            for data_name in DATASET.USED_3D_DATASETS: # 'MUCO', 'CMU', 'H36M', default is 'CMU'
                _3d_json_path = eval('DATASET.%s_JSON_PATH'%(data_name))
                # set the total 3d imgs' num
                if type(_3d_json_path) == list:
                    for i in range(len(_3d_json_path)):
                        _3d_data_list += Path(_3d_json_path[i]).files()[:DATASET.CMU_TRAIN_IMGS]
                # random.shuffle(_3d_data_list)
                # sample_3d_data = random.sample(_3d_data_list, DATASET.CMU_TRAIN_IMGS)  # 小样本学习不用sample
                data = _3d_data_list + data
                
            self.data_list = data

        elif self.stage == 'generation':
            # 还是用train的数据集
            print(f'current mode is {self.stage}, loading {self.stage} dataset ..')
            _3d_data_list = []
            for data_name in DATASET.USED_3D_DATASETS: 
                _3d_json_path = eval('DATASET.%s_JSON_PATH'%(data_name))
                if type(_3d_json_path) == list:
                    for i in range(len(_3d_json_path)):
                        _3d_data_list += Path(_3d_json_path[i]).files()
                data = _3d_data_list

            self.data_list = data
        else:
            # test mode
            # 可以用val的数据集
            print(f'current mode is {self.stage}, loading {self.stage} dataset ..')
            for data_name in DATASET.USED_3D_DATASETS: 
                _3d_json_path = eval('DATASET.%s_VAL_JSON_PATH'%(data_name))
                _3d_data_list = Path(_3d_json_path).files()
                # random.shuffle(_3d_data_list)
                # data = random.sample(_3d_data_list, DATASET.CMU_TEST_IMGS)
                data = _3d_data_list[0:1000]
            self.data_list = data
            data = None
            # with open(cfg.TEST.JSON_PATH) as data_file:
            #     data_this = json.load(data_file)
            #     data = data_this['root']
            #     for i in range(len(data)):
            #         self.data_list += data[i]

        print('load dataset sucessfully ..')

        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.stride = DATASET.STRIDE

        # initial parameters
        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.stride = DATASET.STRIDE

        # data root path
        self.test_root_path = cfg.TEST.ROOT_PATH
        self.root_path = {}
        for dname in (['COCO'] + DATASET.USED_3D_DATASETS): # 'MUCO', 'CMUP', 'H36M'
            self.root_path[dname] = eval('DATASET.%s_ROOT_PATH'%(dname))

        # keypoints information
        self.root_idx = DATASET.ROOT_IDX
        self.keypoint_num = DATASET.KEYPOINT.NUM
        self.gaussian_kernels = DATASET.TRAIN.GAUSSIAN_KERNELS
        self.paf_num = DATASET.PAF.NUM
        self.paf_vector = DATASET.PAF.VECTOR
        self.paf_thre = DATASET.PAF.LINE_WIDTH_THRE            # default is 1

        # augmentation information
        # 图像增强用的参数
        self.with_augmentation = with_augmentation
        self.params_transform = dict()
        self.params_transform['crop_size_x'] = DATASET.INPUT_SHAPE[1]  #832
        self.params_transform['crop_size_y'] = DATASET.INPUT_SHAPE[0]  #512
        self.params_transform['center_perterb_max'] = DATASET.TRAIN.CENTER_TRANS_MAX
        self.params_transform['max_rotate_degree'] = DATASET.TRAIN.ROTATE_MAX
        self.params_transform['flip_prob'] = DATASET.TRAIN.FLIP_PROB
        self.params_transform['flip_order'] = DATASET.KEYPOINT.FLIP_ORDER       
        self.params_transform['stride'] = DATASET.STRIDE               #4
        self.params_transform['scale_max'] = DATASET.TRAIN.SCALE_MAX
        self.params_transform['scale_min'] = DATASET.TRAIN.SCALE_MIN

        self.with_mds = with_mds                 # default is false 
        self.max_people = cfg.DATASET.MAX_PEOPLE # 20

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data = read_json(self.data_list[index])
        meta_data = self.get_anno(data)

        if self.stage not in ['train', 'generation']:
            # test
            root_path = self.root_path[meta_data['dataset']]
        else:
            # train / generation
            root_path = self.root_path[meta_data['dataset']]

        img_ori = cv2.imread(osp.join(root_path, data['img_paths']), cv2.IMREAD_COLOR)  #读取图像

        #随机旋转
        if self.with_augmentation:
            meta_data, img = aug_rotate(meta_data, img_ori, self.params_transform)
        else:
            self.params_transform['center_perterb_max'] = 0

        # crop
        # this step must operate
        if meta_data['dataset'] == 'COCO':
            meta_data, img, pad_value = aug_croppad(meta_data, img, self.params_transform, self.with_augmentation)
        else:
            if self.with_augmentation:
                meta_data, img, pad_value = aug_croppad(meta_data, img, self.params_transform, False)
            else:
                # test for cal metrics
                meta_data, img, pad_value = aug_croppad(meta_data, img_ori, self.params_transform, False)


        #图像翻转
        # if self.with_augmentation:
            # meta_data, img = aug_flip(meta_data, img, self.params_transform)

        #去除增强后不再图像范围中的点
        meta_data = self.remove_illegal_joint(meta_data)

        # transform img
        if self.transform and self.cfg.INPUT.NORMALIZE:
            img = self.transform(img)
        else:
            # pass
            img = img.transpose((2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).float()
        
        # test or generation
        if self.stage in ['test', 'generation']:
            bodys = np.zeros((self.max_people, self.keypoint_num, len(meta_data['bodys'][0][0])), np.float)
            bodys[:len(meta_data['bodys'])] = np.asarray(meta_data['bodys'])
            img_path = data['img_paths']
            return torch.from_numpy(img_ori).float(), img, torch.from_numpy(bodys).float(), img_path, {'scale': meta_data['scale'],
                                                                    'img_width': meta_data['img_width'],
                                                                    'img_height': meta_data['img_height'],
                                                                    'net_width': self.params_transform['crop_size_x'],
                                                                    'net_height': self.params_transform['crop_size_y'],
                                                                    'dataset_name':meta_data['dataset'],
                                                                    'pad_value':pad_value}
        # generate labels
        # valid's function like mask
        valid = np.ones((self.keypoint_num + self.paf_num*3, 1), np.float)
        if meta_data['dataset'] == 'COCO':
            # coco has no headtop annotation
            # valid[1, 0] = 0  

            # pafs of headtop and neck
            # valid[self.keypoint_num, 0] = 0   
            # valid[self.keypoint_num+1, 0] = 0

            # relative depth
            valid[self.keypoint_num + self.paf_num*2:, 0] = 0

        labels_num = len(self.gaussian_kernels)  #5, using different kernel size to crose-to-fine
        labels = np.zeros((labels_num, self.keypoint_num + self.paf_num*3, *self.output_shape))
        #crose to fine
        for i in range(labels_num):
            # heatmaps
            labels[i][:self.keypoint_num] = generate_heatmap(meta_data['bodys'], self.output_shape, self.stride, \
                                                             self.keypoint_num, kernel=self.gaussian_kernels[i])
            # pafs + relative depth
            labels[i][self.keypoint_num:] = generate_paf(meta_data['bodys'], self.output_shape, self.params_transform, \
                                                         self.paf_num, self.paf_vector, max(1, (3-i))*self.paf_thre, self.with_mds)
        # root depth
        labels_rdepth = generate_rdepth(meta_data, self.stride, self.root_idx, self.max_people, self.output_shape)
        # 12.7
        # rdepth_map, rdepth_mask, count = generate_rdepth_map(labels_rdepth, self.output_shape)
        
        # 训练的时候必须的
        labels = torch.from_numpy(labels).float()
        labels_rdepth = torch.from_numpy(labels_rdepth).float()
        valid = torch.from_numpy(valid).float() 

        # 12.7
        # rdepth_map = torch.from_numpy(rdepth_map).float().unsqueeze(0)
        # rdepth_mask = torch.from_numpy(rdepth_mask).float().unsqueeze(0)   

        return img, valid, labels, labels_rdepth        
        # return img, valid, labels, rdepth_map, rdepth_mask  # 12.7

    def get_anno(self, meta_data):
        anno = dict()
        anno['dataset'] = meta_data['dataset'].upper()
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['bodys'] = np.asarray(meta_data['bodys'])
        anno['center'] = np.array([anno['img_width']//2, anno['img_height']//2])
        return anno

    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        for i in range(len(meta['bodys'])):
            #遍历该图像中的所有人
            mask_ = np.logical_or.reduce((meta['bodys'][i][:, 0] >= crop_x,
                                          meta['bodys'][i][:, 0] < 0,
                                          meta['bodys'][i][:, 1] >= crop_y,
                                          meta['bodys'][i][:, 1] < 0))
            #制作这个mask,mask为1的地方值为0
            meta['bodys'][i][mask_ == True, 3] = 0
        return meta

def show_map(center_map):
    center_map = np.array(center_map)

    # center_map = center_map * 255
    plt.subplot(111)
    plt.imshow(center_map)
    plt.axis('off')
    plt.show()

    # center_map = center_map * 255
    # for i in range(9):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(center_map[i])
    #     plt.axis('off')
    #     # plt.savefig(fname="/home/xuchengjun/Desktop/zx/SPM_Depth/results/center_map/" + img_name + '_offset.jpg')
    # plt.show()

if __name__ == '__main__':
    from exps.stage3_root2.config import cfg
    import random
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = JointDataset(cfg, 'train', transform, with_augmentation=True, with_mds=False)

    idx_list = [i for i in range(len(dataset))]
    random.shuffle(idx_list)

    for idx in idx_list:
        print('working ..')
        data = dataset[idx]
        embed()
        # meta_data = data[-1]
        # if meta_data['dataset'] != 'CMU':
        #     continue
        # embed()
        # rdepth_map = data[3]
        # rdepth_mask = data[4]
        # show_map(rdepth_map)
        # show_map(rdepth_mask)

        # heat = data[2]
        # embed()