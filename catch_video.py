import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import torch 
import torchvision.transforms as transforms
from exps.stage3_root2.config import cfg
# from demo import process_video
from model.main_model.smap import SMAP
from model.refine_model.refinenet import RefineNet

import argparse
import dapalib
import numpy as np
from lib.utils.tools import *
from exps.stage3_root2.test_util import *
from path import Path

class CameraTopic(object):
    def __init__(self, topic_name, cfg, main_model, refine_model, device):
        # rospy.init_node('Image_sub', anonymous=True)
        self.main_model = main_model
        self.refine_model = refine_model
        self.cfg = cfg
        self.device = device
        self.image = None
        self.cv_bridge = CvBridge()
        self.topic = topic_name
        self.image_sub = rospy.Subscriber(self.topic, Image, self.callback)
        self.video_path = '/home/xuchengjun/Videos/output6.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 20.0, (1920, 1080))
        self.net_input_shape = (cfg.dataset.INPUT_SHAPE[1], cfg.dataset.INPUT_SHAPE[0])
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = transform
        cam_data = read_json('/home/xuchengjun/ZXin/smap/cam_data/cam.json')
        self.K = np.array(cam_data['K'])

        # 
        self.main_model.eval()
        if self.refine_model is not None:
            self.refine_model.eval()        

    def callback(self, msg):
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.image is not None:
            net_input_image, scales = self.aug_croppad(self.image)
            scales['K'] = self.K
            scales['f_x'] = self.K[0,0]
            scales['f_y'] = self.K[1,1]
            scales['cx'] = self.K[0,2]
            scales['cy'] = self.K[1,2]
            net_input_image = self.transform(net_input_image)
            net_input_image = net_input_image.unsqueeze(0)
            net_input_image = net_input_image.to(self.device)

            with torch.no_grad():
                outputs_2d, outputs_3d, outputs_rd = self.main_model(net_input_image)
                outputs_3d = outputs_3d.cpu()
                outputs_rd = outputs_rd.cpu()

                hmsIn = outputs_2d[0]
                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127 
                rDepth = outputs_rd[0][0]

                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
                if len(pred_bodys_2d) > 0:
                    print('working ..')
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape
                    pred_bodys_2d = pred_bodys_2d.numpy()

                    # ori_resoulution_bodys = recover_origin_resolution(pred_bodys_2d, scales['scale'])
                    # draw_lines(self.image, pred_bodys_2d, cfg.SHOW.BODY_EADGES, (255,0,0))

                    K = scales['K']
                    pafs_3d = outputs_3d[0].numpy().transpose(1, 2, 0)  #part relative depth map (c,h,w) --> (h,w,c) --> (128, 208)
                    root_d = outputs_rd[0][0].numpy()                   # --> (128, 208)
                    #　upsample the outputs' shape to obtain more accurate results
                    #　--> (256, 456)
                    paf_3d_upsamp = cv2.resize(
                        pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)  # (256,456,14)
                    root_d_upsamp = cv2.resize(
                        root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)   #  (256,456)
                    pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scales) #
                    pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scales, scales['pad_value'])

                    # """
                    # refine
                    # """
                    # # new_pred_bodys_3d --> numpy()
                    if self.refine_model is not None:
                        new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, self.refine_model, 
                                                                    device=self.device, root_n=cfg.DATASET.ROOT_IDX)
                    else:
                        new_pred_bodys_3d = pred_bodys_3d          #　shape-->(pnum,15,4)

                    print(new_pred_bodys_3d[:,2,2])

                    if refine_model is not None:
                        refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                        draw_lines(self.image, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                        draw_cicles(refine_pred_2d, self.image)
                    else:
                        refine_pred_2d = project_to_pixel(new_pred_bodys_3d, K)
                        draw_lines(self.image, refine_pred_2d, cfg.SHOW.BODY_EADGES, color=(0,0,255))
                        draw_cicles(refine_pred_2d, self.image)

                cv2.imwrite('./results/img.jpg', self.image)

                self.out.write(self.image)
            # cv2.imshow('img', self.image)
            # cv2.waitKey(33)
        else:
            raise StopIteration
    
    def aug_croppad(self, img):
        scale = dict()                    #创建字典
        crop_x = self.net_input_shape[0]  # width 自己设定的
        crop_y = self.net_input_shape[1]  # height 512
        scale['scale'] = min(crop_x / img.shape[1], crop_y / img.shape[0])  #返回的是最小值
        img_scale = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])
        
        scale['img_width'] = img.shape[1]
        scale['img_height'] = img.shape[0]
        scale['net_width'] = crop_x
        scale['net_height'] = crop_y
        pad_value = [0,0]  # left,up

        center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
        
        if img_scale.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img_scale.shape[1]) // 2
            margin_r = crop_x - img_scale.shape[1] - margin_l
            pad_l = np.ones((img_scale.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img_scale.shape[0], margin_r, 3), dtype=np.uint8) * 128
            pad_value[0] = margin_l
            img_scale = np.concatenate((pad_l, img_scale, pad_r), axis=1)        #在1维进行拼接　也就是w
        elif img_scale.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img_scale.shape[0]) // 2
            margin_d = crop_y - img_scale.shape[0] - margin_u
            pad_u = np.ones((margin_u, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img_scale.shape[1], 3), dtype=np.uint8) * 128
            pad_value[1] = margin_u
            img_scale = np.concatenate((pad_u, img_scale, pad_d), axis=0)       #在0维进行拼接　也就是h
            
        scale['pad_value'] = pad_value
        
        return img_scale, scale



if __name__ == '__main__':
    rospy.init_node('Image_sub', anonymous=True)
    parser = argparse.ArgumentParser()
    # /home/xuchengjun/ZXin/human_pose/pretrained/SMAP_model.pth
    # /media/xuchengjun/zx/human_pose/pth/main/12.16/train.pth
    parser.add_argument('--SMAP_path', type=str, 
                                       default='/media/xuchengjun/zx/human_pose/pth/main/12.16/train.pth')
    # /home/xuchengjun/ZXin/human_pose/pretrained/RefineNet.pth
    # /media/xuchengjun/zx/human_pose/pth/main/12.8/RefineNet_epoch_300.pth
    parser.add_argument('--RefineNet_path', type=str, 
                                       default='/media/xuchengjun/zx/human_pose/pth/main/12.8/RefineNet_epoch_300.pth')
    parser.add_argument('--device', default='cuda:0')  
    args = parser.parse_args()
    device = torch.device(args.device)

    # main model
    model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
    model.to(device)

    # refine model
    refine_model = RefineNet()
    refine_model.to(device)

    smap_model_path = args.SMAP_path
    refine_model_path = args.RefineNet_path

    # smap
    state_dict = torch.load(smap_model_path, map_location=torch.device('cpu'))
    state_dict = state_dict['model']
    model.load_state_dict(state_dict)   

    if Path(refine_model_path).exists():
        print('using refine net..')
        refine_state_dict = torch.load(refine_model_path)
        refine_model.load_state_dict(refine_state_dict)
    else:
        refine_model = None


    cam = CameraTopic('/kinect2_1/hd/image_color', cfg, model, refine_model, device)
    rospy.spin()


