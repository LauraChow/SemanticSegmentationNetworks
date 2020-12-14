

import cfg
from model import FCN
from dataset import CamvidDataset
from evaluation_indices import eval_semantic_segmentation

import numpy as np
import tqdm
import torchsummary
import torch as t
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchnet import meter

# 选择设备
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

def train(model, train_dataloader, val_dataloader, criterion, optimizer):
    print("开始训练！")
    # 设置模型为训练模式
    network = model.train()

    train_loss = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(cfg.DATASET[1])
    # 循环cfg.EPOCH_NUM次
    for epoch in range(cfg.EPOCH_NUM):
        train_loss.reset()
        train_cm.reset()
        with tqdm.tqdm(enumerate(train_dataloader), desc="epoch %3d/%-3d" % (epoch + 1, cfg.EPOCH_NUM),
                       total=len(train_dataloader), dynamic_ncols=True) as tdata:
            for step, sample in tdata:
                # 载入数据
                img = sample["img"].to(device)
                label = sample["label"].to(device)

                # 通过网络得到预测值
                pred = network(img)
                pred = F.log_softmax(pred, dim=1)

                # 计算loss与梯度
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()

                # 梯度下降
                optimizer.step()

                # 计算训练的loss与accuracy
                train_loss.add(loss.item())

                pre_label = pred.max(dim=1)[1].view(-1)
                true_label = label.view(-1)
                train_cm.add(pre_label, true_label)

                cm = train_cm.value()
                train_acc = np.trace(cm) / np.nansum(cm)

                # 输出训练loss与acc
                tdata.set_postfix_str("loss|acc(train): %.4f|%.4f, loss|acc(val): %.4f|%.4f, lr: %.4f"
                                  % (train_loss.value()[0], train_acc, 0, 0, cfg.LEARNING_RATE))





if __name__ == "__main__":
    # 初始化训练验证数据集
    train_ds = CamvidDataset(type="train", crop_size=cfg.CROP_SIZE)
    val_ds = CamvidDataset(type="val", crop_size=cfg.CROP_SIZE)

    # 加载训练验证数据集
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS_NUM)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS_NUM)

    # 加载模型并部署到设备上
    fcn = FCN.FCN(cfg.DATASET[1])
    # torchsummary.summary(model, (cfg.CHANNEL_NUM,) + cfg.CROP_SIZE, device="cpu")
    fcn.to(device)

    # 损失函数
    criterion = nn.NLLLoss().to(device)

    # 优化器
    optimizer = optim.Adam(fcn.parameters(), lr=1e-4)

    # 训练
    train(fcn, train_dl, val_dl, criterion, optimizer)
