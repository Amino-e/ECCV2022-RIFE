import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 设置设备，优先使用GPU
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()   # 将模型加载到GPU
        # AdamW 是一种带有权重衰减 (weight_decay) 修正的 Adam 优化器
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()        # End-Point Error 评估指标, 计算光流估计的平均误差
        self.lap = LapLoss()    # 拉普拉斯损失函数, 用于图像重建任务中, 促进图像的平滑和细节保留
        self.sobel = SOBEL()    # Sobel 边缘检测算子, 用于提取图像的边缘信息
        if local_rank != -1:    # 若 local_rank 不为初始值 -1, 则使用分布式数据并行 (DDP) 来加速训练
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            # DDP 保存的模型参数前会加上 "module." 前缀, 这里去掉该前缀以匹配非 DDP 模型
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5):
        # scale_list是多尺度推理时的缩放比例, scale是输入图像相对于原始图像的缩放比例
        if scale_list is None:
            scale_list = [4, 2, 1]
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        # 调用IFNet的前向传播方法, 返回光流估计结果
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:       # TTA (Test-Time Augmentation) 测试时增强,将输入图像进行翻转,然后对翻转后的图像进行推理,最后将结果翻转回来并与原始结果平均
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        # 根据传入的学习率动态调整优化器的学习率
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()          # 学生网络的拉普拉斯损失
        loss_tea = (self.lap(merged_teacher, gt)).mean()    # 教师网络的拉普拉斯损失
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }
