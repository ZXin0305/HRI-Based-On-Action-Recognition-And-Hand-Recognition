import sys
sys.path.append("/home/xuchengjun/ZXin/smap")
import torch
from model.hand.rexnetv1 import ReXNetV1
import cv2
import numpy as np
from path import Path


class handpose_x_model(object):
    def __init__(self, model_path, img_size=256, num_classes=42, model_arch="rexnetv1", device='cpu'):
        print("handpose_x loading : ",model_path)
        self.img_size = img_size
        self.model_arch = model_arch
        self.device = device
        if model_arch == 'rexnetv1':
            model_ = ReXNetV1(num_classes=num_classes)
        else:
            print("model_arch=", model_arch)
            print("no support the model")

        model_.to(self.device)

        # if Path(model_path).exists():
        #     ck = torch.load(model_path, map_location=self.device)
        #     model_.load_state_dict(ck)
        #     print('handpose_x model loading : {}'.format(model_path))
        self.model_handpose = model_

    def predict(self, img, vis = False):
        with torch.no_grad():

            if not((img.shape[0] == self.img_size) and (img.shape[1] == self.img_size)):
                img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_CUBIC)

            img_ = img.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            img_ = img_.to(self.device)  # (bs, 3, h, w)
            pre_ = self.model_handpose(img_.float())
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            return output

        