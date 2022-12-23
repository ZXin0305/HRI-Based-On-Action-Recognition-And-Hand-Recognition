import sys
import torch
import torch.nn as nn
from model.main_model.conv import conv_bn_relu
from model.main_model.PRM import PRM
from IPython import embed

TOP_STRIDE = 4
MAXPOOL_STRIDE = 2

class HeadTop(nn.Module):
    def __init__(self, cnf, in_ch=3, out_ch=64):
        super(HeadTop, self).__init__()
        self.ori_shape = (cnf.INPUT_SHAPE[0] // TOP_STRIDE, cnf.INPUT_SHAPE[1] // TOP_STRIDE)
        self.top_conv = conv_bn_relu(in_ch, out_ch, kernel_size=7, stride=2, padding=3, has_bn=True, has_relu=True)  # 这里为什么都用的是（7,2,3）
        self.top_maxpool = nn.MaxPool2d(kernel_size=3, stride=MAXPOOL_STRIDE, padding=1)

        # first prm layer followed by a max-pooling layer
        self.pry_1 = PRM(out_ch, out_ch, self.ori_shape, cnf, 'preacat')
        # self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # second prm layer
        # self.pry_2 = PRM(out_ch, out_ch, self.ori_shape, cnf, 'preacat')
        
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.top_conv(x)   # the first layer to process feature
        out = self.top_maxpool(out)
        out = self.pry_1(out)       #这里with mask 开
        # out = self.maxpool_1(out)
        # out = self.pry_2(out)
        return out

class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        self.conv = conv_bn_relu(3, 64, kernel_size=7, stride=2, padding=3,
                has_bn=True, has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        return x
