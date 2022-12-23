"""
The code and network structure are based on https://github.com/megvii-detection/MSPN.
"""
import sys
sys.path.append('/home/xuchengjun/ZXin/smap')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from matplotlib import pyplot as plt
from lib.utils.loss_h import DepthLossWithMask, JointsL2Loss, DepthLoss

from model.main_model.conv import conv_bn_relu
from model.main_model.top import HeadTop as ResNet_top
from model.main_model.residual import ResidualPool
from IPython import embed
from matplotlib import pyplot as plt
import copy
from exps.stage3_root2.config import cfg 
from lib.utils.tools import *

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None,
            efficient=False):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(in_planes, planes, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu2 = conv_bn_relu(planes, planes, kernel_size=3,
                stride=stride, padding=1, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu3 = conv_bn_relu(planes, planes * self.expansion,
                kernel_size=1, stride=1, padding=0, has_bn=True,
                has_relu=False, efficient=efficient) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)
        # print('out shape',out.shape)
        # print('x shape',x.shape)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)

        return out


class ResNet_downsample_module(nn.Module):

    def __init__(self, block, layers, has_skip=False, efficient=False,
            zero_init_residual=False):
        super(ResNet_downsample_module, self).__init__()
        self.has_skip = has_skip 
        self.in_planes = 64
        self.local_expansion = [2,2,2,2]           # [2,2,2,1]     [1,2,2,2] 
        self._layer_planes = [64,128,256,512]      # [64,128,256,512]  [64,64,128,256]
        self.layer1 = self._make_layer(0, block, self._layer_planes[0], layers[0], efficient=efficient)
        self.layer2 = self._make_layer(1, block, self._layer_planes[1], layers[1], stride=2, efficient=efficient)
        self.layer3 = self._make_layer(2, block, self._layer_planes[2], layers[2], stride=2, efficient=efficient)
        self.layer4 = self._make_layer(3, block, self._layer_planes[3], layers[3], stride=2, efficient=efficient)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, up_layer_id, block, planes, blocks, stride=1, efficient=False):
        downsample = None
        block.expansion = self.local_expansion[up_layer_id]  

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, has_bn=True, has_relu=False, efficient=efficient)

        layers = list() 
        layers.append(block(self.in_planes, planes, stride, downsample, efficient=efficient))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, efficient=efficient))

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)

        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]

        return x4, x3, x2, x1


class Upsample_unit(nn.Module): 

    def __init__(self, ind, in_planes, up_size, output_chl_num, output_shape,
            chl_num=256, gen_skip=False, gen_cross_conv=False, efficient=False, using_stage=1):
        super(Upsample_unit, self).__init__()
        self.using_stage = using_stage
        self.output_shape = output_shape
        self.u_skip = conv_bn_relu(in_planes, chl_num, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
        self.residual_connection_new = ResidualPool(in_ch=chl_num, out_ch=chl_num)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=False,
                    efficient=efficient)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
            self.skip2 = conv_bn_relu(chl_num, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == 3 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(chl_num, 64, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        # keypoint heatmaps & paf maps --> 15 + 14*2
        self.res_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.res_conv2 = conv_bn_relu(chl_num, output_chl_num[0], kernel_size=3,
                stride=1, padding=1, has_bn=True, has_relu=False,
                efficient=efficient)

        # part relative depth maps --> 14
        self.res_d_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                                      stride=1, padding=0, has_bn=True, has_relu=True,
                                      efficient=efficient)
        self.res_d_conv2 = conv_bn_relu(chl_num, output_chl_num[1], kernel_size=3,
                                      stride=1, padding=1, has_bn=True, has_relu=False,
                                      efficient=efficient)

        # root depth map --> 1
        self.res_rd_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                                        stride=1, padding=0, has_bn=True, has_relu=True,
                                        efficient=efficient)
        self.res_rd_conv2 = conv_bn_relu(chl_num, 1, kernel_size=3,
                                        stride=1, padding=1, has_bn=True, has_relu=False,
                                        efficient=efficient)

    def forward(self, x, up_x):
        out = self.u_skip(x)

        # if self.ind == 3 and self.using_stage: 
        #     self.show_feature(copy.deepcopy(out))

        if self.ind > 0:
            out = self.residual_connection_new(out)
            # if self.ind == 3 and self.using_stage:
            #     self.show_feature(copy.deepcopy(out))
            up_x = F.interpolate(up_x, size=self.up_size, mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x 
        out = self.relu(out)

        res = None
        res_d = None
        res_rd = None
        res_rd_att = None  # 12.7

        if cfg.IS_TEST == 0 and self.ind != 3:
            # 在测试的时候,中间阶段不用进行输出了
            res = self.res_conv1(out)
            res = self.res_conv2(res)
            res = F.interpolate(res, size=self.output_shape, mode='bilinear', align_corners=True)

            res_d = self.res_d_conv1(out)
            res_d = self.res_d_conv2(res_d)
            res_d = F.interpolate(res_d, size=self.output_shape, mode='bilinear', align_corners=True)

            res_rd = self.res_rd_conv1(out)
            res_rd = self.res_rd_conv2(res_rd)
            res_rd = F.interpolate(res_rd, size=self.output_shape, mode='bilinear', align_corners=True)
            # 12.7
            res_rd_att = F.interpolate(res_rd, size=self.output_shape, mode='bilinear', align_corners=True)

        if cfg.IS_TEST == 1 and self.using_stage == 1:
            res = self.res_conv1(out)
            res = self.res_conv2(res)
            res = F.interpolate(res, size=self.output_shape, mode='bilinear', align_corners=True)

            res_d = self.res_d_conv1(out)
            res_d = self.res_d_conv2(res_d)
            res_d = F.interpolate(res_d, size=self.output_shape, mode='bilinear', align_corners=True)

            res_rd = self.res_rd_conv1(out)
            res_rd = self.res_rd_conv2(res_rd)
            res_rd = F.interpolate(res_rd, size=self.output_shape, mode='bilinear', align_corners=True)

        if self.ind == 3 and self.using_stage == 1:
            res = self.res_conv1(out)
            res = self.res_conv2(res)
            res = F.interpolate(res, size=self.output_shape, mode='bilinear', align_corners=True)

            res_d = self.res_d_conv1(out)
            res_d = self.res_d_conv2(res_d)
            res_d = F.interpolate(res_d, size=self.output_shape, mode='bilinear', align_corners=True)

            res_rd = self.res_rd_conv1(out)
            res_rd = self.res_rd_conv2(res_rd)
            res_rd = F.interpolate(res_rd, size=self.output_shape, mode='bilinear', align_corners=True)

            # 12.7
            if cfg.IS_TEST == 0:
                res_rd_att = F.interpolate(res_rd, size=self.output_shape, mode='bilinear', align_corners=True)
                # print('show map')
                # show_map(copy.deepcopy(res_rd_att[0][0]))
        
        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.ind == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        # return out, res, res_d, res_rd, skip1, skip2, cross_conv
        return out, res, res_d, res_rd, res_rd_att, skip1, skip2, cross_conv  #12.7

    def show_feature(self, feature):
        feature = feature.detach().cpu().numpy()
        ch = feature.shape[1]
        for i in range(ch):
            # feature[0, 1, :, :] += feature[0, i, :, :]
            plt.matshow(feature[0, i, :, :], cmap='viridis')
            plt.savefig(f'/home/xuchengjun/ZXin/smap/results/main/residual/res_{i}.jpg')


class Upsample_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, chl_num=256,
            gen_skip=False, gen_cross_conv=False, efficient=False, using_stage=1):
        super(Upsample_module, self).__init__()
        self.using_stage = using_stage
        self.in_planes = [1024,512,256,128] 
        h, w = output_shape
        self.up_sizes = [
                (h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up1 = Upsample_unit(0, self.in_planes[0], self.up_sizes[0],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient, using_stage=self.using_stage)
        self.up2 = Upsample_unit(1, self.in_planes[1], self.up_sizes[1],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient, using_stage=self.using_stage)
        self.up3 = Upsample_unit(2, self.in_planes[2], self.up_sizes[2],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient, using_stage=self.using_stage)
        self.up4 = Upsample_unit(3, self.in_planes[3], self.up_sizes[3],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient, using_stage=self.using_stage)

    def forward(self, x4, x3, x2, x1):
        # out1, res1, res_d1, res_rd1, skip1_1, skip2_1, _ = self.up1(x4, None)
        # out2, res2, res_d2, res_rd2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        # out3, res3, res_d3, res_rd3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        # out4, res4, res_d4, res_rd4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 12.7
        out1, res1, res_d1, res_rd1, res_rd_att1, skip1_1, skip2_1, _ = self.up1(x4, None)
        out2, res2, res_d2, res_rd2, res_rd_att2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, res3, res_d3, res_rd3, res_rd_att3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        out4, res4, res_d4, res_rd4, res_rd_att4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 'res' starts from small size
        res = [res1, res2, res3, res4]
        res_d = [res_d1, res_d2, res_d3, res_d4]
        res_rd = [res_rd1, res_rd2, res_rd3, res_rd4]
        res_rd_att = [res_rd_att1, res_rd_att2, res_rd_att3, res_rd_att4]   # 12.7
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_4, skip2_3, skip2_2, skip2_1]

        # return res, res_d, res_rd, skip1, skip2, cross_conv
        return res, res_d, res_rd, res_rd_att, skip1, skip2, cross_conv  # 12.7


class Single_stage_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, has_skip=False,
            gen_skip=False, gen_cross_conv=False, chl_num=256, efficient=False, 
            add_ori_sprivi=False, using_stage=1, zero_init_residual=False,):
        super(Single_stage_module, self).__init__()
        self.add_ori_sprivi = add_ori_sprivi
        self.using_stage = using_stage
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.chl_num = chl_num
        self.zero_init_residual = zero_init_residual 
        self.layers = [3, 4, 6, 3]  # resnet 50
        self.downsample = ResNet_downsample_module(Bottleneck, self.layers,
                self.has_skip, efficient, self.zero_init_residual)
        self.upsample = Upsample_module(output_chl_num, output_shape,
                self.chl_num, self.gen_skip, self.gen_cross_conv, efficient, self.using_stage)

    def forward(self, x, skip1, skip2, feature_x):
        if self.add_ori_sprivi:
            x = feature_x + x
        x4, x3, x2, x1 = self.downsample(x, skip1, skip2)
        # res, res_d, res_rd, skip1, skip2, cross_conv = self.upsample(x4, x3, x2, x1)
        res, res_d, res_rd, res_rd_att, skip1, skip2, cross_conv = self.upsample(x4, x3, x2, x1)  # 12.7
        
        # return res, res_d, res_rd, skip1, skip2, cross_conv
        return res, res_d, res_rd, res_rd_att, skip1, skip2, cross_conv  # 12.7


class SMAP_(nn.Module):
    
    def __init__(self, cfg, run_efficient=False, **kwargs):
        super(SMAP_, self).__init__()

        self.stage_num = cfg.MODEL.STAGE_NUM
        self.kpt_paf_num = cfg.DATASET.KEYPOINT.NUM + cfg.DATASET.PAF.NUM*2  # 15 + 14 * 2
        self.keypoint_num = cfg.DATASET.KEYPOINT.NUM   # 15 
        self.paf_num = cfg.DATASET.PAF.NUM             # 14
        self.output_shape = cfg.OUTPUT_SHAPE
        self.upsample_chl_num = cfg.MODEL.UPSAMPLE_CHANNEL_NUM

        self.ohkm = cfg.LOSS.OHKM
        self.topk = cfg.LOSS.TOPK
        self.ctf = cfg.LOSS.COARSE_TO_FINE
        self.using_current_stage = cfg.USING_CURRENT_STAGE
        
        self.top = ResNet_top(cnf=cfg, in_ch=3, out_ch=64)  #新的topNet
        self.modules_stages = list() 
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
                add_ori_sprivi = False
            else:
                has_skip = True
                add_ori_sprivi = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False 
                gen_cross_conv = False 

            self.modules_stages.append(
                    Single_stage_module(
                        [self.kpt_paf_num, self.paf_num], self.output_shape,
                        has_skip=has_skip, gen_skip=gen_skip,
                        gen_cross_conv=gen_cross_conv,
                        chl_num=self.upsample_chl_num,
                        efficient=run_efficient,
                        add_ori_sprivi = add_ori_sprivi,
                        using_stage = self.using_current_stage[i],
                        **kwargs
                        )
                    )
            setattr(self, 'stage%d' % i, self.modules_stages[i])

        self.cross_stage_conv = conv_bn_relu(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, has_bn=False, has_relu=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def _calculate_loss(self, outputs, valids, labels, rdepth_map, rdepth_mask):
        # outputs, valids, labels, rdepth_map, rdepth_mask  # 12.7
        # outputs: stg1 -> stg2 -> ... , res1: bottom -> up
        # valids: (B, C, 1), labels: (B, 5, C, H, W)
        loss2d_1 = JointsL2Loss()
        loss3d_1 = JointsL2Loss()
        if self.ohkm:
            loss2d_2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk, paf_num=self.paf_num)
            loss3d_2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk, paf_num=0)
        # loss_depth = DepthLoss()
        loss_depth = DepthLossWithMask()  #12.7
        loss_depth_att = nn.MSELoss(reduction='none')     #12.7
        loss, loss_2d, loss_bone, loss_root = 0., 0., 0., 0.
        loss_root_att = 0.
        for i in range(self.stage_num):
            for j in range(4):  # multi-scale
                ind = j
                if i == self.stage_num - 1 and self.ctf:  # coarse-to-fine
                    ind += 1 
                tmp_labels = labels[:, ind, :, :, :]
                keypoint_labels = tmp_labels[:, :self.keypoint_num, :, :]
                paf_labels = tmp_labels[:, self.keypoint_num:, :, :]
                paf_index = [idx for idx in range(3*self.paf_num) if idx % 3 != 2]
                tmp_labels_2d = torch.cat([keypoint_labels,
                                           paf_labels[:, paf_index, :, :]], 1)
                tmp_labels_3d = paf_labels[:, 2::3, :, :]

                if j == 3 and self.ohkm:
                    tmp_loss_2d = loss2d_2(outputs['heatmap_2d'][i][j],
                                        valids[:, :self.kpt_paf_num], tmp_labels_2d)
                    tmp_loss_3d = loss3d_2(outputs['det_d'][i][j],
                                        valids[:, self.kpt_paf_num:], tmp_labels_3d)
                else:
                    tmp_loss_2d = loss2d_1(outputs['heatmap_2d'][i][j],
                                        valids[:, :self.kpt_paf_num], tmp_labels_2d)
                    tmp_loss_3d = loss3d_1(outputs['det_d'][i][j],
                                        valids[:, self.kpt_paf_num:], tmp_labels_3d)
                # depth_loss = loss_depth(outputs['root_d'][i][j], rdepth)

                depth_loss = loss_depth(outputs['root_d'][i][j], [rdepth_map,rdepth_mask])
                depth_att_loss = loss_depth_att(outputs['root_d'][i][j], rdepth_mask).mean()

                if j == 3:
                    loss_2d += tmp_loss_2d
                    loss_bone += tmp_loss_3d
                    loss_root += depth_loss
                    loss_root_att += depth_att_loss

                # tmp_loss = 0.1 * tmp_loss_2d + 5 * tmp_loss_3d + 10 * depth_loss
                tmp_loss = 0.1 * tmp_loss_2d + 5 * tmp_loss_3d + 10 * depth_loss + depth_att_loss # 12.7
                if j < 3:
                    tmp_loss = tmp_loss / 4

                loss += tmp_loss

        return dict(total_loss=loss, loss_2d=loss_2d, loss_bone=loss_bone, loss_root=loss_root, loss_root_att=loss_root_att)
        
    def forward(self, imgs, valids=None, labels=None, rdepth=None, rdepth_map=None, rdepth_mask=None):
        x = self.top(imgs)
        feature_x = x
        # self.show_feature(feature_x)

        feature_x = self.bn(feature_x)
        feature_x = self.relu(feature_x)
        feature_x = self.cross_stage_conv(feature_x)

        skip1 = None
        skip2 = None
        outputs = dict()
        outputs['heatmap_2d'] = list()
        outputs['det_d'] = list()
        outputs['root_d'] = list()
        outputs['root_d_att_maps'] = list()
        for i in range(self.stage_num):
            # res, res_d, res_rd, skip1, skip2, x = eval('self.stage' + str(i))(x, skip1, skip2, feature_x)
            # 12.7
            res, res_d, res_rd, res_rd_att, skip1, skip2, x = eval('self.stage' + str(i))(x, skip1, skip2, feature_x)  # 12.7
            outputs['heatmap_2d'].append(res)
            outputs['det_d'].append(res_d)
            outputs['root_d'].append(res_rd)
            outputs['root_d_att_maps'].append(res_rd_att) # 12.7
            
        # print(outputs['heatmap_2d'])
        if valids is None and labels is None:
            outputs_2d = (outputs['heatmap_2d'][-1][-1] + outputs['heatmap_2d'][-1][-2] + outputs['heatmap_2d'][-1][-3])
            return outputs_2d, outputs['det_d'][-1][-1], outputs['root_d'][-1][-1]
        else:
            # return self._calculate_loss(outputs, valids, labels, rdepth)

            return self._calculate_loss(outputs, valids, labels, rdepth_map, rdepth_mask) # 12.7

    def show_feature(self, feature):
        feature = feature.detach().cpu().numpy()
        ch = feature.shape[1]
        for i in range(ch):
            plt.matshow(feature[0, i, :, :], cmap='viridis')
            plt.savefig(f'/home/xuchengjun/ZXin/smap/results/main/feature/feature_{i}.jpg')
        # plt.show()

if __name__ == '__main__':
    from exps.stage3_root2.config import cfg
    model = SMAP(cfg)

    input = torch.ones(size=(1,3,512,832), device='cpu')
    out_1, out_2, out_3 = model(input, valid=True)
    # outputs = model(input)
    embed()

        
