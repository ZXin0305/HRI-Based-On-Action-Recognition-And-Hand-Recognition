
import torch
import torch.nn as nn
import torch.autograd as autograd
from IPython import embed


#这里是为了确保后面在预测根深度的时候能够更加准确，只用对应的坐标点来计算loss
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, output, rdepth):
        batch_size = output.size(0)
        output_num = output.size(1)
        assert output_num == 1
        loss, count = 0., 0
        for i in range(batch_size):
            for j in range(len(rdepth[i])):
                if rdepth[i, j, 2] > 0:
                    loss += torch.abs(output[i, 0, int(rdepth[i][j][0]), int(rdepth[i][j][1])] - rdepth[i][j][2])
                    count += 1
        if count == 0:   # used for forward
            loss += torch.abs(output[0, 0, 1, 1] - rdepth[0][0][2])
            loss = loss * 0
            count = 1
        return loss / count

class DepthLossWithMask(nn.Module):
    def __init__(self):
        super(DepthLossWithMask, self).__init__()
        
    def forward(self, output, label):
        bs = output.shape[0]
        feature_ch = output.shape[1] 
        # pre = output.clone()
        # gt = label[0].clone()
        # mask = label[1].clone()

        assert output.size() == label[0].size()    # torch.Size([1, 1, 128, 208]) torch.Size([1, 1, 128, 208])

        loss = 0.
        nonzero_num = 0
        for i in range(bs):
            for j in range(feature_ch):
                nonzero_num += len(torch.nonzero(label[0][i, j]))
                loss += torch.sum(torch.abs(output[i, j] - label[0][i, j]) * label[1][i, j])
        if nonzero_num == 0:   # used for forward
            # loss += torch.abs(output[0, 0, 0, 0] - label[0][0，0，0，0])
            loss = loss * 0
            nonzero_num = 1
        return loss / nonzero_num


class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thres=0, paf_num=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.paf_num = paf_num
        self.thres = thres
        self.calculate = nn.MSELoss(reduction='none')

    def forward(self, output, valid, label):
        assert output.shape == label.shape
        
        tmp_loss = self.calculate(output, label)
        tmp_loss = tmp_loss.mean(dim=[2, 3])
        weight = torch.gt(valid.squeeze(), self.thres).float()
        tmp_loss *= weight

        if not self.has_ohkm:
            loss = tmp_loss.mean()
        else:
            if self.paf_num == 0:
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=1, sorted=False)
                loss = topk_val.mean()
            else:
                keypoint_num = output.shape[1] - self.paf_num * 2
                keypoint_loss = tmp_loss[:, :keypoint_num]
                paf_loss = tmp_loss[:, keypoint_num:]
                keypoint_topk_val, keypoint_topk_id = torch.topk(keypoint_loss, k=self.topk, dim=1, sorted=False)
                paf_topk_val, paf_topk_id = torch.topk(paf_loss, k=self.topk*2, dim=1, sorted=False)
                loss = keypoint_topk_val.mean() + paf_topk_val.mean()
        return loss

       