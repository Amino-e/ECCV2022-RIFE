import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *

# deconv: 转置卷积 (上采样) + PReLU激活块。stride=2 的转置卷积通常用于将特征图的尺寸放大一倍。
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )
# conv: 标准卷积 + PReLU激活块; dilation: 扩大感受野
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        # conv0: 两层卷积, 每层卷积后特征图尺寸减半, 通道数先减半再增倍至c
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        # convblock: 八层卷积, 保持特征图尺寸和通道数不变
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # lastconv: 转置卷积 + 上采样, 输出5通道(4通道光流+1通道mask)
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        # 如果传入的scale不为1(超分辨率), 则对输入特征图
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        # 对光流进行下采样
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x   # 残差连接
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)    # 上采样至原始尺寸的两倍
        flow = tmp[:, :4] * scale * 2   # 前四个通道为光流, 乘以scale*2以还原光流的实际幅度
        mask = tmp[:, 4:5]              # 最后一个通道为融合掩码mask
        return flow, mask
    
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        # 3个学生网络块
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)
        # 教师网络块
        self.block_tea = IFBlock(16+4, c=90)
        # 上下文网络和UNet细化模块
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4,2,1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                # 细化阶段：输入包含原图、上一级扭曲后的图、上一级的mask和flow
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d    # 残差连接
                mask = mask + mask_d    
            else:
                # 初始阶段：输入仅包含两帧图像
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            # 教师网络阶段：输入包含原图、扭曲后的图、mask、flow和gt
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                # loss_mask: 找出学生预测比老师差很多的区域
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                # 蒸馏损失: 仅在这些区域计算学生和老师的光流差异
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
