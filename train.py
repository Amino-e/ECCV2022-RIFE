import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = 'train_log'

def get_learning_rate(step):       # 学习率策略：前2000步线性增加学习率至3e-4，后续步数余弦退火至3e-6
    if step < 2000:     # warm up：前2000步线性增加学习率至3e-4
        mul = step / 2000.
        return 3e-4 * mul
    else:               # 余弦退火：后续步数余弦退火至3e-6
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):          # 光流可视化
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    if local_rank == 0:             # 仅在主进程rank 0上创建TensorBoard writer，避免多进程同时写入文件冲突
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None

    step = 0              # 记录训练步数
    nr_eval = 0           # 记录验证次数
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)       # 分布式采样器，确保每个进程获取不同的数据
    # num_workers:使用8个子进程加载数据;pin_memory=True:将数据加载到页锁定内存中以加速GPU传输;drop_last=True:丢弃最后一个不完整的批次
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()

    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)

    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)        # 每个 epoch 开始前，需要调用 set_epoch 来打乱数据顺序
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.    # 数据送入GPU，并归一化至0-1
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]      # 前六个通道为输入的两帧图像,即torch.cat((img0, img1, gt), 0)中的img0和img1
            gt = data_gpu[:, 6:9]       # 后三个通道为gt图像,即torch.cat((img0, img1, gt), 0)中的gt
            learning_rate = get_learning_rate(step) * args.world_size / 4   # 学习率随步数变化，并根据分布式进程数调整

            pred, info = model.update(imgs, gt, learning_rate, training=True) # pass timestep if you are training RIFEm
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # 日志记录部分(主程序进行)
            if step % 200 == 1 and local_rank == 0:     # 每200步记录一次各种损失和学习率
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
            if step % 1000 == 1 and local_rank == 0:    # 每1000步记录一次图像和光流
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:     # 每步都在主进程打印一次日志
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:   # 每5个验证周期进行一次验证
            evaluate(model, val_data, step, local_rank, writer_val)
        model.save_model(log_path, local_rank)      # 每个epoch结束后保存一次模型
        dist.barrier()  # 同步各个进程，确保所有进程都完成当前epoch后再进行下一epoch

def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():       # 评估时不计算梯度，节省显存
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):    # 逐图计算PSNR
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank)
        
