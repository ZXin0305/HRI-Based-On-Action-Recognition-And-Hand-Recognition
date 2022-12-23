import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import argparse
import time

import torch
from tensorboardX import SummaryWriter

from cvpack.torch_modeling.engine.engine import Engine
from cvpack.utils.pyt_utils import ensure_dir
import numpy as np
from path import Path
from config import cfg
# from model.main_model.smap import SMAP
from model.main_model.new_model import SMAP_new as SMAP
# from model.main_model.model_tmp import SMAP_tmp as SMAP
from lib.utils.dataloader import get_train_loader
from lib.utils.solver import make_lr_scheduler, make_optimizer
from lib.utils.tools import load_state
from IPython import embed
import random
import os
from DNN_printer import DNN_Printer


def main():
    parser = argparse.ArgumentParser()

    with Engine(cfg, custom_parser=parser) as engine:
        logger = engine.setup_log(
            name='train', log_dir=cfg.OUTPUT_DIR, file_name=cfg.LOG_FILE_NAME)
        args = engine.args
        ensure_dir(cfg.OUTPUT_DIR)

        model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

        num_gpu = len(engine.devices) 
        #  default num_gpu: 8, adjust iter settings
        # cfg.SOLVER.CHECKPOINT_PERIOD = \
        #         int(cfg.SOLVER.CHECKPOINT_PERIOD * 8 / num_gpu)
        cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER)
        print(f'current MAX ITER --> {cfg.SOLVER.MAX_ITER}')

        optimizer = make_optimizer(cfg, model, num_gpu)
        scheduler = make_lr_scheduler(cfg, optimizer)

        # 将模型等数据也放进engine中
        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        # DDP mode
        if engine.distributed:
            print('using DDP mode ..')
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                broadcast_buffers=True, find_unused_parameters=True)

        # if engine.continue_state_object:
        #     print('continue_state_object .. ')
        #     engine.restore_checkpoint(is_restore=False)
        # else:
        # 导入断点
        if len(cfg.MODEL.WEIGHT)>0 and Path(cfg.MODEL.WEIGHT).exists():
            engine.load_checkpoint(cfg.MODEL.WEIGHT, is_restore=False)

        data_loader = get_train_loader(cfg, num_gpu=num_gpu, is_dist=engine.distributed,
                                       use_augmentation=True, with_mds=cfg.WITH_MDS)

        # -------------------- do training -------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(
            torch.__version__))

        max_iter = len(data_loader)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD     # default is 500
        if engine.local_rank == 0:
            tb_writer = SummaryWriter(cfg.TENSORBOARD_DIR)

        model.train()

        time1 = time.time()
        # (images, valids, labels, rdepth_map, rdepth_mask) # 12.7   (images, valids, labels, rdepth)
        for iteration, (images, valids, labels, rdepth) in enumerate(data_loader, engine.state.iteration):
            iteration = iteration + 1
            images = images.to(device)
            valids = valids.to(device)
            labels = labels.to(device)
            rdepth = rdepth.to(device)
            # rdepth_map = rdepth_map.to(device)  #12.7
            # rdepth_mask = rdepth_mask.to(device) #12.7
            # loss_dict = model(imgs=images, valids=valids, labels=labels, rdepth_map=rdepth_map, rdepth_mask=rdepth_mask) #12.7
            loss_dict = model(imgs=images, valids=valids, labels=labels, rdepth=rdepth)  
            
            losses = loss_dict['total_loss']

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()   #这里根据iteration来调整学习率

            if cfg.RUN_EFFICIENT:
                del images, valids, labels, losses

            if engine.local_rank == 0:
                if iteration % 20 == 0 or iteration == max_iter:
                    log_str = 'Iter:%d, LR:%.1e, ' % (
                        iteration, optimizer.param_groups[0]["lr"] / num_gpu)
                    for key in loss_dict:
                        tb_writer.add_scalar(
                            key, loss_dict[key], global_step=iteration)
                        log_str += key + ': %.3f, ' % float(loss_dict[key])

                    time2 = time.time()
                    elapsed_time = time2 - time1
                    time1 = time2
                    required_time = elapsed_time / 20 * (max_iter - iteration)  # 距离当前训练完的时间
                    hours = required_time // 3600
                    mins = required_time % 3600 // 60
                    log_str += 'To Finish: %dh%dmin,' % (hours, mins) 

                    logger.info(log_str)   # 读写日志

            if iteration % checkpoint_period == 0 or iteration == max_iter:
                engine.update_iteration(iteration)
                # save iteration .pth
                if not engine.distributed or engine.local_rank == 0:
                    engine.save_and_link_checkpoint(cfg.MODEL.CK_PATH, is_ckpt=True)

            if iteration % 5000 == 0 and iteration >= 20000:
                engine.update_iteration(iteration) 
                # save train.pth
                if not engine.distributed or engine.local_rank == 0:
                    engine.save_and_link_checkpoint(cfg.MODEL.CK_PATH, is_ckpt=False)

            if iteration >= max_iter:
                logger.info('Finish training process!')
                break

def set_seed(seed=2022):
    # seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    # set_seed()
    main()
