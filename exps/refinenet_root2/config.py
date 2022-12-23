# encoding: utf-8
import os, getpass
import os.path as osp
import argparse

from easydict import EasyDict as edict
from dataset.data_settings import load_dataset
from cvpack.utils.pyt_utils import ensure_dir


class Config:
    # -------- Directoy Config -------- #
    DATA_DIR = '/media/xuchengjun/disk/datasets/CMU/refine'

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 8
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0

    DATASET = edict()
    DATASET.ROOT_IDX = 2  # pelvis
    DATASET.MAX_PEOPLE = 20

    # -------- Model Config -------- #
    MODEL = edict()
    MODEL.DEVICE = 'cuda'
    MODEL.GPU_IDS = [0,1,2]

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 0.002
    SOLVER.BATCH_SIZE = 1024
    SOLVER.NUM_EPOCHS = 250
    SOLVER.LR_STEP_SIZE = 30
    SOLVER.GAMMA = 0.5
    SOLVER.DROP_STEP = [100, 150, 200, 250, 300, 350, 400, 425, 450, 475, 500, 525, 550, 575, 600]

    # --------- Checkpoint Config -------- #
    PRINT_FREQ = 1
    CHECK_FREQ = 2000
    SAVE_FREQ = 1
    SAVE_PATH = '/media/xuchengjun/disk/zx/human_pose/pth/refine'
    CHECK_PATH = '/media/xuchengjun/disk/zx/human_pose/pth/refine_ck.pth'
    PRETRAINED_PATH = ''

    # --------- Testing Config ----------- #
    TEST = edict()
    TEST.BATCH_SIZE = 1


cfg = Config()

def link_log_dir():
    if not osp.exists('./log'):
        ensure_dir(cfg.OUTPUT_DIR)
        cmd = 'ln -s ' + cfg.OUTPUT_DIR + ' log'
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
