# encoding: utf-8
import os, getpass
import os.path as osp
import argparse

from easydict import EasyDict as edict  #easydict的作用：可以使得以属性的方式去访问字典的值！
from dataset.data_settings import load_dataset
from cvpack.utils.pyt_utils import ensure_dir
from IPython import embed
from path import Path


class Config:
    # -------- Directory Config -------- #
    ROOT_DIR = '/home/xuchengjun/ZXin'
    LOG_FILE_NAME = 'log0520.txt'
    OUTPUT_DIR = Path(ROOT_DIR) / 'smap/exps/stage3_root2/model_logs'
    TEST_DIR = OUTPUT_DIR / 'log_dir'     # log dir
    TENSORBOARD_DIR = Path('/media/xuchengjun/disk/zx/human_pose/pth/main/20220520')  # tensorboard

    # -------- Data Config(数据配置) -------- #
    #这里是对这个“字典”中的属性进行设置
    #DATALOADER
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 1
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0
    #DATASET
    DATASET = edict()
    DATASET.NAME = 'MIX'
    dataset = load_dataset(DATASET.NAME)      #
    DATASET.KEYPOINT = dataset.KEYPOINT       #KEYPOINT = edict()
    DATASET.PAF = dataset.PAF                 #PAF = edict()
    DATASET.ROOT_IDX = dataset.ROOT_IDX       # pelvis or neck    ROOT_IDX=2
    DATASET.MAX_PEOPLE = 20                   #预设置最大人数

    #INPUT
    INPUT = edict()
    INPUT.NORMALIZE = True                    #规范化
    INPUT.MEANS = [0.406, 0.456, 0.485]       # bgr 平均值 按照这三个通道
    INPUT.STDS = [0.225, 0.224, 0.229]        # 标准差
    INPUT_SHAPE = dataset.INPUT_SHAPE         #(512,832)
    OUTPUT_SHAPE = dataset.OUTPUT_SHAPE       #(128,208)

    # -------- Model Config -------- #
    #模型的参数设置
    MODEL = edict()
    MODEL.STAGE_NUM = 3                       #训练阶段的数　　３
    MODEL.UPSAMPLE_CHANNEL_NUM = 256          #上采样通道数　　256
    MODEL.DEVICE = 'cuda'                     # choose cuda
    MODEL.GPU_IDS = [0,1,2]                     # set useful gpu-ids
    MODEL.SAVE_PATH = '/media/xuchengjun/disk/zx/human_pose/pth/main/20220520'                      # the path to save model
    MODEL.CK_PATH = '/media/xuchengjun/disk/zx/human_pose/pth/main/20220520'                        # the path to save/load model checkpoint  
    # /home/xuchengjun/ZXin/human_pose/pretrained/SMAP_model.pth
    # /media/xuchengjun/zx/human_pose/pth/main/20220328/train.pth
    MODEL.WEIGHT = "/media/xuchengjun/zx/human_pose/pth/main/20220416/train.pth"                       # osp.join(ROOT_DIR, 'lib/models/resnet-50_rename.pth') /media/xuchengjun/zx/human_pose/pretrained

    # -------- Training Config -------- #
    #训练参数设置
    SOLVER = edict()
    SOLVER.IMG_PER_GPU = 2                    #训练时，每个ＧＰＵ处理多少图片
    SOLVER.BASE_LR = 2e-4                      #学习率
    SOLVER.CHECKPOINT_PERIOD = 5000            #保存中间的训练参数
    SOLVER.MAX_ITER = 600000                  #总的ite
    SOLVER.WEIGHT_DECAY = 8e-6                 #权重的decay
    SOLVER.WARMUP_FACTOR = 0.1                 #这两个是调整学习率用的
    SOLVER.WARMUP_ITERS = 3400 

    #LOSS参数设置
    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True

    WITH_MDS = True
    RUN_EFFICIENT = False 
    IS_TEST = 1
    USING_CURRENT_STAGE = [0,0,1]
    # USING_CURRENT_STAGE = [1,1]
    
    # -------- Test Config -------- #
    TEST = edict()
    TEST.IMG_PER_GPU = 1
    TEST.ROOT_PATH = '/media/xuchengjun/disk/datasets/mupots-3d-eval/MultiPersonTestSet'  # '/data/datasets/mupots-3d-eval/MultiPersonTestSet'
    TEST.JSON_PATH = Path(TEST.ROOT_PATH) / 'M3E_gt.json'
    TEST_DIR = 'main'
    TEST_PATH = Path('/media/xuchengjun/disk/datasets/CMU/refine')      # path to save the final-json file

    # ------- show results -------- #
    SHOW = edict()
    SHOW.BODY_EADGES = [[0,1], [0,2], [0,9], [0,3], 
                        [3,4], [4,5], 
                        [9,10], [10,11], 
                        [2,6], [6,7], [7,8], 
                        [2,12], [12,13], [13,14]]
    SHOW.COLOR_LIB = []

    SHOW.HAND_EADGES = [[0, 1], [1, 2], [2, 3], [3, 4],
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        [0, 17], [17, 18], [18, 19], [19, 20]]

    SHOW.HAND_CENTER_BORDER = [0, 2, 5, 9, 13, 17]

config = Config()
cfg = config      #cfg是创建好的类对象


def link_log_dir():
    if not osp.exists('./log'):
        ensure_dir(config.OUTPUT_DIR)
        cmd = 'ln -s ' + config.OUTPUT_DIR + ' log'
        os.system(cmd)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-log', '--linklog', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
