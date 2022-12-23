import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import os
import os.path as osp
import numpy as np
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from path import Path
from dataset.p2p_dataset import P2PDataset
from model.refine_model.refinenet import RefineNet
from config import cfg
from datetime import datetime
from lib.utils.tools import load_state

def main():
    train_dataset = P2PDataset(dataset_path=cfg.DATA_DIR, root_idx=cfg.DATASET.ROOT_IDX)
    print(f'using refine dataset --> {cfg.DATA_DIR}')
    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)
    loader_len = len(train_loader)
    check_path = Path(cfg.CHECK_PATH)
    pretrained_path = Path(cfg.PRETRAINED_PATH)
    
    model = RefineNet()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    """
    load checkpoint
    """
    # if check_path.exists():
    #     print(f'load checkpoint --> {check_path}')
    #     checkpoint = torch.load(check_path)
    #     load_state(model, checkpoint)
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     iteration = checkpoint['iter']
    #     epoch = checkpoint['current_epoch']
    #     print('have load the model ..')

    # if pretrained_path.exists():
    #     print(f'load pretrained --> {pretrained_path}')
    #     pretrained = torch.load(pretrained_path)
    #     model.load_state_dict(pretrained)
    #     print('have load pretrained model ..')

    if len(cfg.MODEL.GPU_IDS) > 1:
        model = nn.parallel.DataParallel(model, device_ids=cfg.MODEL.GPU_IDS)
        print('using DP mode ..')
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999))
    # 1.
    # drop_after_epoch = cfg.SOLVER.DROP_STEP
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.5)
    # 2.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.LR_STEP_SIZE, gamma=cfg.SOLVER.GAMMA, last_epoch=-1)
    
    criterion = nn.MSELoss()

    model.train()
    
    epoch = 1
    iteration = 1

    for i in range(epoch, cfg.SOLVER.NUM_EPOCHS+1):
        total_loss = 0
        count = 0
        for (inp, gt) in (train_loader):
            iteration += 1
            count += 1
            inp = inp.to(device)
            gt = gt.to(device)

            preds = model(inp)
            loss = criterion(preds, gt)
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % cfg.CHECK_FREQ == 0 or iteration == loader_len and iteration != 0:
                ck = {
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter': iteration,
                    'current_epoch': epoch
                }
                
                torch.save(ck, cfg.CHECK_PATH)

            print('\r[{}] Epoch:{} progress:{}/{} Loss:{} LR:{}'.format(datetime.now().strftime("%m-%d@%H:%M"), epoch, count, len(train_loader), total_loss / count, optimizer.param_groups[0]["lr"]),end='')
            del inp, gt
        scheduler.step()

        # avg_loss = total_loss / count
        # if epoch % cfg.PRINT_FREQ == 0:
        #     print("epoch: {} | loss: {}.".format(epoch, avg_loss))
        if epoch % cfg.SAVE_FREQ == 0 or epoch == cfg.SOLVER.NUM_EPOCHS:
            torch.save(model.module.state_dict(), osp.join(cfg.SAVE_PATH, "RefineNet_epoch_%03d.pth" % epoch))
        epoch += 1
        iteration = 0


# def set_seed(seed=2022):
#     # seed = int(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    # set_seed()
    main()
