import os
import time

import cfg
from model import FCN
from dataset import CamvidDataset
from evaluation_indices import cal_semantic_segmentation_indices
from utils import PrettyFormatUtils, ExaminationUtils
from utils.ImgUtils.LabelProcessor import LabelProcessor


import tqdm
import torchsummary
import torch as t
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchnet import meter
from torch.utils.tensorboard import SummaryWriter


# 训练
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device,
          model_save_path, comment=""):
    # 提示为训练模式
    PrettyFormatUtils.print_title("训练")

    # 初始化训练参数
    epoch_start, best_loss, val_loss, val_acc = 0, float("inf"), float("inf"), 0
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
        model.train()
        with tqdm.tqdm(enumerate(train_dataloader),
                       desc="epoch %3d/%-3d" % (epoch + 1, cfg.EPOCH_NUM),
                       postfix="loss|acc(train): %.4f|%.4f, loss|acc(val): %.4f|%.4f, lr: %.4f"
                               % (float("inf"), 0, float("inf"), 0, optimizer.state_dict()["param_groups"][0]["lr"]),
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

                # 计算训练的指标
                train_loss_meter.add(loss.item())
                train_loss = train_loss_meter.value()[0]
                train_dict = cal_semantic_segmentation_indices(pred, label, train_cm_meter)
                PrettyFormatUtils.log_indices(writer, train_loss, train_dict["OverallAccuracy"], n_iter)

                # 每个epoch最后计算验证的指标
                if step == len(train_dataloader)-1:
                    val_loss, val_acc = val(model, val_dataloader, criterion, device)
                    PrettyFormatUtils.log_indices(writer, val_loss, val_acc, n_iter, "Val")

                # 在进度条上更新训练/验证的loss与acc
                tdata.set_postfix_str("loss|acc(train): %.4f|%.4f, loss|acc(val): %.4f|%.4f, lr: %.4f"
                                      % (train_loss, train_dict["OverallAccuracy"], val_loss, val_acc,
                                         optimizer.state_dict()["param_groups"][0]["lr"]))

        # 根据验证集的表现调整学习率
        scheduler.step(val_loss)

        # 保存表现最好的模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(os.path.join(model_save_path, comment),
                       epoch, model, optimizer, scheduler, best_loss, log_dir)

    writer.close()

# 验证
@t.no_grad()
def val(model, val_dataloader, criterion, device):
    # 网络为验证模式
    model.eval()

    # 初始化验证指标：loss与混淆局长
    val_loss_meter = meter.AverageValueMeter()
    val_cm_meter = meter.ConfusionMeter(cfg.DATASET[1])

    # 开始验证
    for step, sample in enumerate(val_dataloader):
        # 载入数据
        img = sample["img"].to(device)
        label = sample["label"].to(device)

        # 通过网络得到预测值并计算当前的loss
        pred = model(img)
        pred = F.log_softmax(pred, dim=1)
        loss = criterion(pred, label)

        # 计算验证的指标
        val_loss_meter.add(loss.item())
        val_loss = val_loss_meter.value()[0]
        val_dict = cal_semantic_segmentation_indices(pred, label, val_cm_meter)

    return val_loss, val_dict["OverallAccuracy"]

# 测试
@t.no_grad()
def test(model, test_dataloader, device,
         model_save_path, comment=""):
    # 提示为训练模式
    PrettyFormatUtils.print_title("测试")

    model_path = os.path.join(model_save_path, comment, "ckpt.pth")
    ExaminationUtils.is_path_exist(False, model_path)

    # 加载模型
    ckpt = t.load(model_path, map_location=t.device('cpu'))
    model.load_state_dict(ckpt["model"])

    # 网络为验证模式
    model.eval()

    # 初始化测试指标：loss与混淆局长
    test_loss_meter = meter.AverageValueMeter()
    test_cm_meter = meter.ConfusionMeter(cfg.DATASET[1])

    # 开始测试
    with tqdm.tqdm(enumerate(test_dataloader),
                   postfix="loss|acc: %.4f|%.4f" % (float("inf"), 0),
                   desc="Test", total=len(test_dataloader), dynamic_ncols=True) as tdata:
        for step, sample in tdata:
            # 载入数据
            img = sample["img"].to(device)
            label = sample["label"].to(device)

            # 通过网络得到预测值并计算当前的loss
            pred = model(img)
            pred = F.log_softmax(pred, dim=1)
            loss = criterion(pred, label)

            # 计算验证的指标
            test_loss_meter.add(loss.item())
            test_loss = test_loss_meter.value()[0]
            test_dict = cal_semantic_segmentation_indices(pred, label, test_cm_meter)

            # 在进度条上更新训练/验证的loss与acc
            tdata.set_postfix_str("loss|acc: %.4f|%.4f" % (test_loss, test_dict["OverallAccuracy"]))

    print("Test loss:", test_loss)

# 预测
@t.no_grad()
def predict(model, predict_dataloader, prediction_save_path, class_dict_path, device,
            model_save_path, comment=""):
    # 提示为训练模式
    PrettyFormatUtils.print_title("预测")

    model_path = os.path.join(model_save_path, comment, "ckpt.pth")
    ExaminationUtils.is_path_exist(False, model_path, class_dict_path)
    ExaminationUtils.is_path_exist(True, prediction_save_path)

    label_processor = LabelProcessor(class_dict_path)

    # 加载模型
    ckpt = t.load(model_path, map_location=t.device('cpu'))
    model.load_state_dict(ckpt["model"])

    # 开始预测
    with tqdm.tqdm(enumerate(predict_dataloader),
                   desc="Prediction", total=len(predict_dataloader), dynamic_ncols=True) as tdata:
        for step, sample in tdata:
            # 载入数据
            img = sample["img"].to(device)

            # 通过网络得到预测值并计算当前的loss
            pred = model(img)
            pred = F.log_softmax(pred, dim=1)

            # 得到预测图并保存
            pred_img = label_processor.decode_label_img(pred)
            pred_img.save(os.path.join(prediction_save_path, str(step)+".png"))


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
    # 选择设备
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

    # 初始化训练验证数据集
    train_ds = CamvidDataset(type="train", crop_size=cfg.CROP_SIZE)
    val_ds = CamvidDataset(type="val", crop_size=cfg.CROP_SIZE)

    test_ds = CamvidDataset(type="test", crop_size=cfg.CROP_SIZE)

    # 加载训练验证数据集
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS_NUM)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS_NUM)

    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.WORKERS_NUM)
    predict_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.WORKERS_NUM)

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
    # train(fcn, train_dl, val_dl, criterion, optimizer, scheduler, device, cfg.MODEL_SAVE_PATH)

    # 测试
    # test(fcn, test_dl, device, cfg.MODEL_SAVE_PATH)

    # 预测
    predict(fcn, predict_dl, cfg.PREDICT_SAVE_PATH, cfg.CLASS_DICT_PATH, device, cfg.MODEL_SAVE_PATH)
