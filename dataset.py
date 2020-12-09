import os

import cfg
from utils import OSUtils, ExaminationUtils, ImgUtils

from PIL import Image
from torch.utils.data import Dataset


# 处理Camvid数据集的类
class CamvidDataset(Dataset):
    label_processor = ImgUtils.LabelProcessor(cfg.CLASS_DICT_PATH)

    # 相当于CamvidDataset的构造函数
    def __init__(self, dataset_path="./dataset/CamVid",
                 class_dict_path="./dataset/CamVid/class_dict.csv",
                 type="train",
                 crop_size=None):
        print("初始化", type, "set ... ...")

        # 检查数据集路径
        ExaminationUtils.is_path_exist(dataset_path)

        # 获取影像路径与标签路径并检查
        self.img_path = os.path.join(dataset_path, type)
        self.label_path = os.path.join(dataset_path, type+"_labels")
        ExaminationUtils.is_path_exist(self.img_path, self.label_path)

        # 获取影像路径下的所有影像名
        self.imgs = OSUtils.list_file_name(self.img_path)
        self.labels = OSUtils.list_file_name(self.label_path)

        # 检查标签路径是否与影像路径的影像名匹配
        if len(self.imgs) != len(self.labels):
            raise ValueError("影像（{}）与标签（{}）数目不匹配！".format(len(self.imgs), len(self.labels)))
        print(type, "set", "大小为：", len(self.imgs))

        # 初始化参数
        self.crop_size = crop_size

    def __getitem__(self, idx):
        # 打开影像与标签
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('RGB')

        # 预处理：影像、标签中心裁剪，影像归一化，标签由三通道颜色值转为相应类别的索引
        img, label = ImgUtils.ImgProcessor.center_crop(self.crop_size, img, label)
        img, label = ImgUtils.ImgProcessor.img_transform(img, label, self.label_processor)

        return {"img": img, "label": label}

    def __len__(self):
        return len(self.imgs)


# 测试代码
if __name__ == "__main__":
    train = CamvidDataset(type="train", crop_size=cfg.CROP_SIZE)
    val = CamvidDataset(type="val", crop_size=cfg.CROP_SIZE)
    test = CamvidDataset(type="test", crop_size=cfg.CROP_SIZE)

    print(train[0], val[0], test[1])
