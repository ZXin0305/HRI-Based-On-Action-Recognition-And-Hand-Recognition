import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class conv_bn_relu(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding,
                 has_bn=True,has_relu=True, efficient=False):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride,padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.efficient = efficient
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,x):
        def _func_factory(conv ,bn ,relu ,has_bn ,has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu )
        
        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x