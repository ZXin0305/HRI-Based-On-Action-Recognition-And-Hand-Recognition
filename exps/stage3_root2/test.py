import os
import argparse
import json
from posixpath import join
import cv2
from tqdm import tqdm
import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
from torch.utils.data import DataLoader

# from model.main_model.smap import SMAP     
from model.main_model.new_model import SMAP_new         
# from model.main_model.mode_1 import SMAP_
# from model.main_model.model_tmp import SMAP_tmp
from model.refine_model.refinenet import RefineNet

from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process
from exps.stage3_root2.test_util import *
from lib.utils.camera_wrapper import CustomDataset
from exps.stage3_root2.config import cfg
from exps.stage3_root2.test_util import save_result, save_result_for_train_refine
import dapalib_light
import dapalib
from IPython import embed
import copy
from time import time
from matplotlib import pyplot as plt
from lib.utils.tools import *
from lib.utils.test_metric import *
from IPython import embed


def generate_3d_point_pairs(model, refine_model, data_loader, cfg, device, output_dir=''):

    model.eval()
    if refine_model is not None:
        refine_model.eval()

    # foe cal f1
    tp = []
    fn = []
    fp = []
    pr = []
    re = []
    f1 = []
    mpjpe = []
    root_err = []

    # 3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 'gt_2d':[[x,y,Z,visual_type]...],
    #                         'pred_3d':[[X,Y,Z,score]...], 'gt_3d':[[X,Y,Z]...],
    #                         'root_d': (abs depth of root (float value) pred by network),
    #                         'image_path': relative image path}

    kpt_num = cfg.DATASET.KEYPOINT.NUM
    data = data_loader

    dataset_name = None
    count = 0
    data_len = len(data)
    if data_len > 0:
        count = 1
    for idx, batch in enumerate(data):
        if cfg.TEST_MODE == 'run_inference':
            # Custom Dataset
            ori_imgs, imgs, img_paths, scales, dataset_name = batch
            meta_data = None
        else:
            # Base Dataset
            ori_imgs, imgs, meta_data, img_paths, scales = batch  # meta_data:(gt-pose) imgs, keypoints, img_path, a dict --> some information like img height/width, net input size
            dataset_name = scales[0]['dataset_name']
            pad_value = scales[0]['pad_value']
            # print(pad_value)
        imgs = imgs.to(device)
        ori_imgs = ori_imgs[0].numpy()  # for adding img

        # print(img_paths[0])
        # img_path_split = img_paths[0].split('/')
        # if img_path_split[0] != '160906_pizza1':
        #     continue
        # if img_path_split[2] != '00_00':
        #     continue
        # if img_path_split[-1] != '00_00_00001433.jpg':
        #     continue

        with torch.no_grad():
            # outputs_2d, outputs_3d, outputs_rd = model(imgs, valid=True)  # (heatmaps + pafs) / relative-depth-maps / root-depth-map
            outputs_2d, outputs_3d, outputs_rd = model(imgs)            # for smap
            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()

            # ----------------------------------------------------------------

            if cfg.DO_FLIP:
                imgs_flip = torch.flip(imgs, [-1])  # imgs:(B,c,h,w) -1:垂直翻转
                outputs_2d_flip, outputs_3d_flip, outputs_rd_flip = model(imgs_flip)
                outputs_2d_flip = torch.flip(outputs_2d_flip, dims=[-1])  # flip back
                # outputs_3d_flip = torch.flip(outputs_3d_flip, dims=[-1])
                # outputs_rd_flip = torch.flip(outputs_rd_flip, dims=[-1])

                keypoint_pair = cfg.DATASET.KEYPOINT.FLIP_ORDER
                paf_pair = cfg.DATASET.PAF.FLIP_CHANNEL

                paf_abs_pair = [x+kpt_num for x in paf_pair]  # kpt_num:15
                pair = keypoint_pair + paf_abs_pair   # total list of the idx --> 15 + 28 = 43
                for i in range(len(pair)):
                    if i >= kpt_num and (i - kpt_num) % 2 == 0:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]] * -1
                    else:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]
                outputs_2d[:, kpt_num:] *= 0.5

                # -----------------------------------------------------------

            for i in range(len(imgs)):
                img_path = img_paths[i]
                img_path_split = img_path.split('/')

                if meta_data is not None:
                    # None : custom_dataset
                    # not-None: base_dataset --> generate_train & generate_result mode
                    # remove person who was blocked
                    new_gt_bodys = []
                    annotation = meta_data[i].numpy()
                    scale = scales[i]

                    # filter points
                    for j in range(len(annotation)):
                        if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                            new_gt_bodys.append(annotation[j])

                    # new_gt_bodys = filter_pose(np.asarray(new_gt_bodys))

                    gt_bodys = np.asarray(new_gt_bodys)    #　shape-->(pnum,15,11)  这里的pnum是真实的gt人数，将为0的全部过滤了
                    if len(gt_bodys) == 0:
                        # root_err.append(0)
                        # mpjpe.append(0)
                        continue
                    # groundtruth:[person..[keypoints..[x, y, Z, score(0:None, 1:invisible, 2:visible), X, Y, Z,
                    #                                   f_x, f_y, cx, cy]]]

                    if dataset_name != 'CMU':
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 7]
                        scale['cx'] = scale['img_width']/2
                        scale['cy'] = scale['img_height']/2
                    elif dataset_name == 'CMU':
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 8]
                        scale['cx'] = gt_bodys[0, 0, 9]
                        scale['cy'] = gt_bodys[0, 0, 10]
                else:
                    gt_bodys = None   #　just to run_inference , donnot use gt poses.
                    # use default values, but is to internet imgs
                    # while CMU has the real parameters
                    if dataset_name != 'CMU':
                        scale = {k: scales[k][i].numpy() for k in scales}
                        scale['f_x'] = scale['img_width']
                        scale['f_y'] = scale['img_width']
                        scale['cx'] = scale['img_width']/2
                        scale['cy'] = scale['img_height']/2
                    elif dataset_name == 'CMU':
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 8]
                        scale['cx'] = gt_bodys[0, 0, 9]
                        scale['cy'] = gt_bodys[0, 0, 10]

                K = np.asarray([[scale['f_x'], 0, scale['cx']], [0, scale['f_y'], scale['cy']], [0, 0, 1]])
                
                hmsIn = outputs_2d[i]  # (c, h, w)

                # for i in range(hmsIn.shape[0]):
                #     show_map(copy.deepcopy(hmsIn[i]), i)
                # embed()

                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255  #keypoints heatmaps
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127  #paf maps
                rDepth = outputs_rd[i][0]                #root depth maps (h, w)
                
                # show_map(rDepth, id=i)
                # break 
                # no batch implementation yet
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)  #　--> (128, 228)
                # here is to upsample the 2d-grouping size to input size
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape --> (256, 456)
                    pred_bodys_2d = pred_bodys_2d.numpy()

                """
                see the middle result
                for run_inference or generate_result or test .. before refine
                # save_img_results(ori_imgs, img_path, ori_resoulution_bodys, scale['scale'], idx)
                """
                # ori_resoulution_bodys = recover_origin_resolution(copy.deepcopy(pred_bodys_2d), scale['scale'])
                # for_test_gt = recover_origin_resolution(copy.deepcopy(np.array(new_gt_bodys)), scale['scale'])
                # draw_lines(ori_imgs, ori_resoulution_bodys, cfg.SHOW.BODY_EADGES, color=(255,0,0))
                # draw_cicles(for_test_gt, ori_imgs, is_gt=True)
                # cv2.imwrite('/home/xuchengjun/ZXin/smap/results/test_draw_lines.jpg', ori_imgs)
               
                pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
                root_d = outputs_rd[i][0].numpy()                   # --> (128, 208)

                #　upsample the outputs' shape to obtain more accurate results
                #　--> (256, 456)
                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
                
                # new_root_d_upsamp = cv2.resize(root_d, (0,0), fx=1/scale['scale'], fy=1/scale['scale'], interpolation=cv2.INTER_NEAREST) 
                # new_root_d_upsamp = np.maximum(root_d, 0)
                # new_root_d_upsamp /= np.max(new_root_d_upsamp)
                # new_root_d_upsamp = np.uint8(255 * new_root_d_upsamp)
                # new_root_depth_map = cv2.applyColorMap(new_root_d_upsamp, cv2.COLORMAP_JET)
                # new_root_depth_map = cv2.resize(new_root_depth_map, (1920,1080), fx=1/scale['scale'], fy=1/scale['scale'], interpolation=cv2.INTER_NEAREST)
                # draw_cicles(ori_resoulution_bodys, new_root_depth_map)
                # cv2.imwrite('/home/xuchengjun/ZXin/smap/results/depth_with_root.jpg', new_root_depth_map)
                
                # .....
                # ori_imgs = ori_imgs.astype(np.uint8)
                # add_img = cv2.addWeighted(ori_imgs, 1, new_root_depth_map, 0.3, 0)
                # cv2.imwrite('/home/xuchengjun/ZXin/smap/results/add_img.jpg',add_img)
                
                

                # generate 3d prediction bodys
                """
                if 'run_inference': just remove people whose root score is 0
                else: generate new_pred_bodys to get refine-net train-data-set
                """
                # --> (256, 456)
                pred_bodys_2d = register_pred(pred_bodys_2d, gt_bodys)  # 将pred和gt的值按照相同的顺序进行对准

                """
                filter poses 
                """
                if len(pred_bodys_2d) > 0:
                    match_pair_list = pred_bodys_2d[:, 2, 3] == 0
                    no_match_list =  [i for i in range(len(match_pair_list)) if match_pair_list[i] == True]
                    # print(no_match_list)
                    # remove no-matched-pair
                    if len(no_match_list) == 0:
                        pass
                    # 1. pred-bodys
                    pred_bodys_2d = pred_bodys_2d[pred_bodys_2d[:,2,3] != 0]
                    # 2. create new_gt_bodys ..
                    new_gt_bodys = np.zeros((gt_bodys.shape))
                    for i in range(len(gt_bodys)):
                        if (i in no_match_list):
                            continue
                        else:
                            new_gt_bodys[i] = gt_bodys[i]
                    new_gt_bodys = new_gt_bodys[new_gt_bodys[:,2,3] != 0] 
                else:
                    # no human --> no need continue doing the rest steps
                    print('no human')
                    # mpjpe.append(0)
                    # root_err.append(0)
                    continue

                # --> (256, 456)
                # pred_bodys_2d --> the root depth is 0, others are root-relative-depth .. 
                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale) #
                # --> (1080, 1920)
                pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale, pad_value)                  # actual 3D poses in camera-coordinate  shape-->(pnum,15,4)
                
                """
                refine
                """
                # new_pred_bodys_3d --> numpy()
                # embed()
                if refine_model is not None:
                    new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model, 
                                                                device=device, root_n=cfg.DATASET.ROOT_IDX)
                else:
                    new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4)

                """
                cal the metric
                """

                """
                root error
                """
                # metrics_root = cal_rootErr(new_pred_bodys_3d, new_gt_bodys)
                # # # print(metrics_root)
                # if len(metrics_root) == 0:
                #     metrics_root = [0]
                # root_err.append(np.mean(np.array(metrics_root)))

                """
                mpjpe
                if metrics_mpjpe > 1e-8:
                """
                # metrics_mpjpe, use_cal_list = cal_mpjpe(new_pred_bodys_3d, new_gt_bodys, metrics_root)
                # # metrics_mpjpe, use_cal_list = cal_mpjpe(new_pred_bodys_3d, gt_bodys, metrics_root)
                # # print(metrics_mpjpe)
                # mpjpe.append(metrics_mpjpe)
                # print(f'working .. {count}/{data_len}')
                # count += 1

                """
                f1_score
                """
                # if metrics_mpjpe <= 0:
                #     continue
                # else:
                # if len(use_cal_list) == 0: 
                #     continue
                # else:
                #     metrics_f1 = joint_det_metrics(new_pred_bodys_3d[use_cal_list], new_gt_bodys[use_cal_list], th=15.0)
                #     # metrics_f1 = joint_det_metrics(new_pred_bodys_3d, gt_bodys, th=25)
                #     pr.append(metrics_f1['pr'])
                #     re.append(metrics_f1['re'])
                #     f1.append(metrics_f1['f1'])                                   
                
                
                """
                after refine , project back to pixel
                """
                # after refine ..
                # 现在别用预训练的修正模型，本来就不是一个数据集
                # if cfg.TEST_MODE == 'generate_result':
                #     refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                #     # embed()
                #     draw_lines(ori_imgs, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                #     cv2.imwrite(f'/home/xuchengjun/ZXin/smap/results/refine.jpg',ori_imgs)
                # -------------
                """
                save for train refinenet
                """
                if cfg.TEST_MODE == "generate_train":
                    save_result_for_train_refine(pred_bodys_2d, new_pred_bodys_3d, new_gt_bodys, 
                                                 pred_rdepths, img_path_split, output_dir)
                else:
                    # run_inference / generate_result
                    save_result(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, img_path[i])

            print(f'generate .. {count}/{data_len}')
            count += 1

    # print(f'avg_pr:{np.mean(np.array(pr))}, avg_re:{np.mean(np.array(re))}, avg_f1:{np.mean(np.array(f1))} {len(f1)}')
    # print(f'avg_mpjpe:{np.mean(np.array(mpjpe))}, len:{len(mpjpe)}')
    # print(f'root_err:{np.mean(np.array(root_err))}, len:{len(root_err)}')


def main():
    """
    1.test_mode:
        (1):run_inference: just to val the model's efficiency, do not store the gt
                       using CustomDataset , batch_size set to one , meta_data is None
        (2):generate_train: do inference & store coresponding gt --> 3D pairs
                       using BaseDataset, batch_size set to .. , meta_data is not None,  to create refine-net train dataset
        (3):generate_result: val the model, but also store the gt
                       using BaseDataset, mata_data is not None
    2.data_mode:
        (1):test: use to obtain test dataset, link to BaseDataset(without meta data), such as CMU's 00_16/00_30 dirs
        (2):generation: use to generate refine-train-dataset, link to BaseDataset(with meta_data)

    above all:
        if test_mode --> generate_train && data_mode --> generation
            the pair mode to get the dataset to train refine model ...
    
    可以用test.py计算评测指标的。。但是要用的是测试集合

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", "-t", type=str, default="generate_train",
                        choices=['generate_train', 'generate_result', 'run_inference'],
                        help='Type of test. One of "generate_train": generate refineNet datasets, '
                             '"generate_result": save inference result and groundtruth, '
                             '"run_inference": save inference result for input images.')
    parser.add_argument("--data_mode", "-d", type=str, default="generation",
                        choices=['test', 'generation'],
                        help='Only used for "generate_train" test_mode, "generation" for refineNet train dataset,'
                             '"test" for refineNet test dataset.')
    # /home/xuchengjun/ZXin/smap/pretrained/main_model.pth
    # /media/xuchengjun/zx/human_pose/pth/main/12.16/train.pth
    # /media/xuchengjun/zx/human_pose/pth/main/1.4/train.pth
    parser.add_argument("--SMAP_path", "-p", type=str, default='/media/xuchengjun/disk/zx/human_pose/pth/main/20220520/train.pth',
                        help='Path to SMAP model')
    # /home/xuchengjun/ZXin/human_pose/pretrained/RefineNet.pth
    # /media/xuchengjun/zx/human_pose/pth/main/1.4/RefineNet_epoch_250.pth
    parser.add_argument("--RefineNet_path", "-rp", type=str, default='',
                        help='Path to RefineNet model, empty means without RefineNet')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch_size of test')
    parser.add_argument("--do_flip", type=float, default=0,
                        help='Set to 1 if do flip when test')
    parser.add_argument('--device', default='cuda:0')
    # /media/xuchengjun/datasets/CMU/170407_haggling_a1/hdImgs/00_16
    # /media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample
    # /media/xuchengjun/datasets/action/images/stand
    # /media/xuchengjun/datasets/coco_2017
    parser.add_argument("--dataset_path", type=str, default="/media/xuchengjun/disk/datasets/CMU/160906_pizza1/val",
                        help='Image dir path of "run_inference" test mode')
    parser.add_argument("--json_name", type=str, default="final_json",
                        help='Add a suffix to the result json.')
    args = parser.parse_args()

    # set params
    cfg.TEST_MODE = args.test_mode
    cfg.DATA_MODE = args.data_mode
    cfg.REFINE = len(args.RefineNet_path) > 0
    cfg.DO_FLIP = args.do_flip
    cfg.JSON_NAME = args.json_name
    cfg.TEST.IMG_PER_GPU = args.batch_size    # if just run_inference, set to 1

    # model = SMAP(cfg)
    # model = SMAP_(cfg, run_efficient=cfg.RUN_EFFICIENT)
    model = SMAP_new(cfg, run_efficient=cfg.RUN_EFFICIENT)
    # model = SMAP_tmp(cfg)
    device = torch.device(args.device)
    model.to(device)

    if args.test_mode == "run_inference":
        test_dataset = CustomDataset(cfg, args.dataset_path)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # generate_train or generate_result
        #　base_dataset
        data_loader = get_test_loader(cfg, num_gpu=1, local_rank=0, stage=args.data_mode)

    # load the refine net or not
    if cfg.REFINE:
        refine_model = RefineNet()
        refine_model.to(device)
        refine_model_path = args.RefineNet_path
    else:
        print('no using refine-net')
        refine_model = None
        refine_model_path = ""

    # check model path
    model_path = args.SMAP_path
    if os.path.exists(model_path):

        # smap
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)     

        if os.path.exists(refine_model_path):
            refine_model.load_state_dict(torch.load(refine_model_path))
        elif refine_model is not None:
            print(f'No such RefineNet checkpoint of {args.RefineNet_path}')
            return
        generate_3d_point_pairs(model, refine_model, data_loader, cfg, device, output_dir=cfg.TEST_PATH)
    else:
        print(f'No such checkpoint of SMAP {args.SMAP_path}')


if __name__ == '__main__':
    main()
