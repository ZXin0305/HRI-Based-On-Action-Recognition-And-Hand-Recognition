import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import torch
from model.main_model.conv import conv_bn_relu
from model.main_model.CA import CoordAtt

"""
this function is to create the new residual unit according to the paper
"""

# branch(B) main convolution block
class Branch_B(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: every downsample layer's channel
        :param out_ch:default upsample layer's channel is 256
        :return branch(B)'s output
        """
        super(Branch_B, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv_bn_relu(in_ch, in_ch // 4, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True)
        self.conv2 = conv_bn_relu(in_ch // 4, in_ch // 4, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=True)
        self.conv3 = conv_bn_relu(in_ch // 4, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out

# branch(A) skip layer branch
class Branch_A(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch:
        :param out_ch:
        :return before Branch_A, the channel has been changed  by conv
        """
        super(Branch_A, self).__init__()
        self.in_ch = in_ch
        self.upsample_ch = out_ch
        self.conv = conv_bn_relu(in_ch, out_ch, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=False)

    def forward(self, x):
        #here, the in_ch maybe is equal with the upsample_ch
        if self.in_ch != self.upsample_ch:
            out = self.conv(x)
        else:
            out = x

        return out

# branch(C) to expand the receive filed
class Branch_C(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch_C, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.ori_size = None
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bn=True, has_relu=False)

    def forward(self, x):
        # self.ori_size = x.shape[2:]
        out = self.maxpool(x)
        out = self.conv1(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out


# the new residual block
class ResidualPool(nn.Module):
    def __init__(self, in_ch, out_ch):
        assert in_ch % 2 == 0
        super(ResidualPool, self).__init__()
        self.branch_A = Branch_A(in_ch, out_ch)
        self.branch_B = Branch_B(in_ch, out_ch)
        self.branch_C = Branch_C(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.CA = CoordAtt(in_ch, out_ch)

    def forward(self, x):
        out_A = self.branch_A(x)  # the channnel down 2, while the (h, w) is same
        out_B = self.branch_B(x)  # the channnel down 2, while the (h, w) is same
        out_C = self.branch_C(x)  # the channnel down 2, while the (h, w) is same
        out = out_A + out_B + out_C
        out = self.CA(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

if __name__ == '__main__':
    input = torch.zeros(size=(1,2,5,5), dtype=torch.float32)
    input[:,:,2,1] = 1.
    input[:,:,2,2] = 1.
    input[:,:,2,3] = 1.
    input = input.to('cpu')
    net = ResidualPool(2, 1)
    net.to('cpu')
    net.train()
    out = net(input)
    print(out[0,0,:])



