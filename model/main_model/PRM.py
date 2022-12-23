"""
Residual Pyramid with Pooling
"""
import torch
import torch.nn as nn
from model.main_model.conv import conv_bn_relu
import torch.nn.functional as F
from IPython import embed

class PryBottleNet(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, scaled):
        super(PryBottleNet, self).__init__()
        self.ori_shape = ori_shape
        self.max_pool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(scaled, scaled))
        self.conv = conv_bn_relu(out_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=False)

    def forward(self, x):
        out = self.max_pool(x)
        out = self.conv(out)
        out = F.interpolate(out, size=self.ori_shape, mode='bilinear', align_corners=True)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, type, cardinality):
        super(ConvBlock, self).__init__()
        self.middle_ch = out_ch // 2
        self.C = cardinality  # the num of prm branch, default is 3
        self.ori_shape = ori_shape
        self.pyramid = list()
        self.scaled = 2 ** (1 / self.C)  # control the scaled ratio
        self.type = type

        self.main_branch = nn.Sequential()
        if self.type != 'no_preact':
            self.main_branch.add_module('activation_layer1', nn.BatchNorm2d(in_ch))
            self.main_branch.add_module('activation_layer2', nn.ReLU(inplace=True))

        self.main_branch.add_module('conv1', conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True))
        self.main_branch.add_module('conv2', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=3, stride=1, padding=1, has_relu=True, has_bn=True))
        self.main_branch.add_module('conv3', conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False))

        for i in range(self.C):
            tmp_scaled = 1 / (self.scaled ** (i+1))  # change the scaled to change the feature resolution
            self.pyramid.append(
                PryBottleNet(self.middle_ch, self.middle_ch, self.ori_shape, tmp_scaled)
            )

            setattr(self, 'pry{}'.format(i), self.pyramid[i])

        self.conv_top = conv_bn_relu(in_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  # 这个是在pra各个等级之前的，进行一次卷积
        self.bn = nn.BatchNorm2d(self.middle_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv_bot = conv_bn_relu(self.middle_ch, self.middle_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)  #这个是紧跟着pra输出相加之后的卷积
        self.conv_out = conv_bn_relu(self.middle_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)   #这个是再一次进行卷积

    def forward(self, x):
        # 1. main branch
        out_main = self.main_branch(x)   # (B, 32, 128, 208)

        # 2. pyramid branch
        # ---------------------------
        out_pry = None
        pyraTable = list()
        conv_top = self.conv_top(x)
        for i in range(self.C):
            out_pry = eval('self.pry' + str(i))(conv_top)   # 这里出来的都是和输入尺寸一样的
            pyraTable.append(out_pry)
            if i != 0:
                out_pry = pyraTable[i] + pyraTable[0]

        out_pry = self.bn(out_pry)        # 在前面使用bn和relu是为了减少方差
        out_pry = self.relu(out_pry)

        out_pry = self.conv_bot(out_pry)  # 金字塔分支进行相加后卷积

        # ------------------------------
        assert out_pry.shape == out_main.shape
        out = out_pry + out_main
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)

        return out


class SkipLayer(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(SkipLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv_bn_relu(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, has_bn=False, has_relu=False)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            out = x
        else:
            out = self.bn(x)
            out = self.relu(out)
            out = self.conv(out)

        return out


class PRM(nn.Module):
    def __init__(self, in_ch, out_ch, ori_shape, cnf, type):
        """
        :param in_ch: 64
        :param out_ch: 64
        :param cnf:
        """
        cardinality = 3   #这个数？？？？
        super(PRM, self).__init__()
        self.skip_layer = SkipLayer(in_ch, out_ch, stride=1)                      # 主要的原分支
        self.pry_layer = ConvBlock(in_ch, out_ch, ori_shape, type, cardinality)   #　多分辨率分支
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out_skip = self.skip_layer(x)    # (B, 64, 128, 208)
        out_pry = self.pry_layer(x)
        assert out_pry.shape == out_skip.shape
        out = out_pry + out_skip
        out = self.bn(out)
        out = self.relu(out)  
        return out 


if __name__ == '__main__':
    net = PRM(64, 64, (128, 208), None, 'no_preact')
    input = torch.ones(size=(16,64,128,208), dtype=torch.float32)
    out = net(input)
    embed()

