import os
import time

import cfg
from model import FCN
from dataset import CamvidDataset
from evaluation_indices import eval_semantic_segmentation
from utils import PrettyFormatUtils

import numpy as np
import tqdm
import torchsummary
import torch as t
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchnet import meter
from torch.utils.tensorboard import SummaryWriter

# 选择设备
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

# 训练
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
          model_save_path, comment=""):
    PrettyFormatUtils.print_title("训练")

    epoch_start, best_loss, val_loss, val_acc = 0, float("inf"), float("inf"), 0,
    log_dir = os.path.join(model_save_path, comment, "runs", time.strftime('%m%d_%H%M%S'))

    # 如果有之前训练好的模型，则加载
    if os.path.exists(os.path.join(model_save_path, comment, "ckpt.pth")):
        epoch_start, model, optimizer, scheduler, best_loss, log_dir = \
            load_model(os.path.join(model_save_path, comment, "ckpt.pth"),
                       model, optimizer, scheduler)

    # 初始化训练指标：loss与混淆矩阵
    train_loss_meter = meter.AverageValueMeter()
    train_cm_meter = meter.ConfusionMeter(cfg.DATASET[1])
    writer = SummaryWriter(log_dir=log_dir)

    # 循环cfg.EPOCH_NUM次
    for epoch in range(epoch_start, cfg.EPOCH_NUM):
        train_loss_meter.reset()
        train_cm_meter.reset()

        # 设置模型为训练模式
        model = model.train()
        with tqdm.tqdm(enumerate(train_dataloader),
                       desc="epoch %3d/%-3d" % (epoch + 1, cfg.EPOCH_NUM),
                       postfix="loss|acc(train): %.4f|%.4f, loss|acc(val): %.4f|%.4f, lr: %.4f"
                               % (0, 0, 0, 0, optimizer.state_dict()["param_groups"][0]["lr"]),
                       total=len(train_dataloader), dynamic_ncols=True) as tdata:
            for step, sample in tdata:
                # 获取当前的迭代次数（以batch为单位）
                n_iter = epoch * len(train_dataloader) + step
                # 载入数据
                img = sample["img"].to(device)
                label = sample["label"].to(device)

                # 通过网络得到预测值与损失
                pred = model(img)
                pred = F.log_softmax(pred, dim=1)
                loss = criterion(pred, label)

                # 梯度下降
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算训练的loss与accuracy
                train_loss, train_acc = calculate_indices(pred, label, loss, train_loss_meter, train_cm_meter)
                PrettyFormatUtils.log_indices(writer, train_loss, train_acc, n_iter)

                if step == len(train_dataloader)-1:
                    val_loss, val_acc = val(model, val_dataloader, criterion)
                    PrettyFormatUtils.log_indices(writer, val_loss, val_acc, n_iter, "Val")

                # 输出训练loss与acc
                tdata.set_postfix_str("loss|acc(train): %.4f|%.4f, loss|acc(val): %.4f|%.4f, lr: %.4f"
                                      % (train_loss, train_acc, val_loss, val_acc,
                                         optimizer.state_dict()["param_groups"][0]["lr"]))

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(os.path.join(model_save_path, comment),
                       epoch, model, optimizer, scheduler, best_loss, log_dir)

    writer.close()

# 验证
@t.no_grad()
def val(model, val_dataloader, criterion):
    # 网络为验证模式
    network = model.eval()

    val_loss_meter = meter.AverageValueMeter()
    val_cm_meter = meter.ConfusionMeter(cfg.DATASET[1])
    for step, sample in enumerate(val_dataloader):
        # 载入数据
        img = sample["img"].to(device)
        label = sample["label"].to(device)

        # 通过网络得到预测值并计算当前的loss
        pred = network(img)
        pred = F.log_softmax(pred, dim=1)
        loss = criterion(pred, label)

        # 计算验证的loss与accuracy
        val_loss, val_acc = calculate_indices(pred, label, loss, val_loss_meter, val_cm_meter)

    return val_loss, val_acc


def calculate_indices(pred, label, loss, loss_meter, confusion_matrix_meter):
    loss_meter.add(loss.item())

    pre_label = pred.max(dim=1)[1].view(-1)
    true_label = label.view(-1)
    confusion_matrix_meter.add(pre_label, true_label)

    cm = confusion_matrix_meter.value()
    accuracy = np.trace(cm) / np.nansum(cm)

    return loss_meter.value()[0], accuracy


def save_model(save_path,
               epoch, model, optimizer, scheduler, best_loss, log_dir):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    t.save({"epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "logdir": log_dir
            }, os.path.join(save_path, "ckpt.pth"))


def load_model(save_path,
               model, optimizer, scheduler):
    ckpt = t.load(save_path, map_location=t.device('cpu'))

    epoch_start = ckpt["epoch"]+1
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_loss = ckpt["best_loss"]
    log_dir = ckpt["logdir"]

    return epoch_start, model, optimizer, scheduler, best_loss, log_dir


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
    optimizer = optim.Adam(fcn.parameters(), lr=cfg.LEARNING_RATE)

    # 学习率调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=2,
                                                     min_lr=0.00001, threshold=1)
    # 训练
    train(fcn, train_dl, val_dl, criterion, optimizer, scheduler, cfg.MODEL_SAVE_PATH)
