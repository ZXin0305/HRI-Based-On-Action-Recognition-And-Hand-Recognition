from easydict import EasyDict as edict
import os
import os.path as osp

class MIX:
    NAME = 'MIX'              # using dataset include COCO & other 3d dataset
    KEYPOINT = edict()
    KEYPOINT.NUM = 15
    '''The order in this work:
        (0-'neck'  1-'head'  2-'pelvis'  
        3-'left_shoulder'  4-'left_elbow'  5-'left_wrist'
        6-'left_hip'  7-'left_knee'  8-'left_ankle'
        9-'right_shoulder'  10-'right_elbow'  11-'right_wrist'
        12-'right_hip'  13-'right_knee'  14-'right_ankle')
    '''
    KEYPOINT.FLIP_ORDER = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]
    ROOT_IDX = 2  # pelvis idx

    # PAF
    # total keypoint_num is 15, so the paf_vector is 14
    # please attention the [0,1], [0,2]
    # link order : right --> left
    PAF = edict()
    PAF.VECTOR = [[0, 1], [0, 2],
                  [0, 9], [9, 10], [10, 11],
                  [0, 3], [3, 4], [4, 5],
                  [2, 12], [12, 13], [13, 14],
                  [2, 6], [6, 7], [7, 8]]

    PAF.FLIP_CHANNEL = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 
                        22, 23, 24, 25, 26, 27, 16, 17, 18, 19, 20, 21]

    PAF.NUM = len(PAF.VECTOR)        #14
    PAF.LINE_WIDTH_THRE = 1          #线的宽度

    # ************* net's input size *************
    INPUT_SHAPE = (512,832)  # height, width 原始的是512,832  256, 456
    STRIDE = 4               # will be used to upsample the output
    OUTPUT_SHAPE = (INPUT_SHAPE[0] // STRIDE, INPUT_SHAPE[1] // STRIDE)  # last upsample block's output shape
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]                 # only appear here

    # 2d dataset
    # COCO
    # /media/xuchengjun/datasets/coco_2017 
    # /media/xuchengjun/datasets/COCO
    COCO_ROOT_PATH = '/media/xuchengjun/disk/datasets/coco_2017'
    COCO_JSON_PATH = '/media/xuchengjun/disk/datasets/COCO'    # COCO json path

    # 3d dataset
    USED_3D_DATASETS = ['MUCOPTS']  # 3D数据集  CMU
    # MUCO
    MUCO_ROOT_PATH = '/media/xuchengjun/disk/datasets/mupots-3d-eval/result'
    MUCO_JSON_PATH = osp.join(MUCO_ROOT_PATH, "annotations/MuCo.json")
    MUCOPTS_ROOT_PATH = '/media/xuchengjun/disk/datasets/mupots-3d-eval/result'
    MUCOPTS_JSON_PATH = osp.join(MUCO_ROOT_PATH, 'M3E_gt.json')
    # CMU  .. 160422_ultimatum1/hdImgs/00_03/00_03_00000266.jpg
    # /media/xuchengjun/datasets/CMU/train/160422_ultimatum1/00
    # 160906_pizza1  170221_haggling_b1
    # new_CMU_json_file
    CMU_ROOT_PATH = '/media/xuchengjun/disk/datasets/panoptic-toolbox'
    CMU_JSON_PATH = ['/media/xuchengjun/disk/datasets/CMU/train']  #  train or generation
    CMU_VAL_JSON_PATH = '/media/xuchengjun/disk/datasets/CMU/160422_ultimatum1/val' # for test
    CMU_TRAIN_IMGS = 160000
    CMU_GENERATION_IMGS = 160000    # refine net train dataset
    CMU_TEST_IMGS = 1000          # val, some metrics
    # H36M
    H36M_ROOT_PATH = ''
    H36M_JSON_PATH = osp.join(H36M_ROOT_PATH, "annotations/H36M.json")

    # TRAIN
    TRAIN = edict()
    # dataset augmentation
    TRAIN.CENTER_TRANS_MAX = 40
    TRAIN.ROTATE_MAX = 10
    TRAIN.FLIP_PROB = 0.5
    TRAIN.SCALE_MAX = 1.1
    TRAIN.SCALE_MIN = 0.8
    # 高斯核大小  高斯模糊
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]


def load_dataset(name):
    if 'MIX' in name:
        dataset = MIX        #数据集
        return dataset
    return None
