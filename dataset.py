import os

from utils import osUtils, examinationUtils, ImgUtils

from PIL import Image
from torch.utils.data import Dataset


# 处理Camvid数据集的类
class CamvidDataset(Dataset):
    # 相当于CamvidDataset的构造函数
    def __init__(self, dataset_path="./dataset/CamVid",
                 class_dict_path="./dataset/CamVid/class_dict.csv",
                 type="train",
                 crop_size=None):
        # 检查数据集路径
        examinationUtils.is_path_exist(dataset_path)

        # 获取影像路径与标签路径并检查
        self.img_path = os.path.join(dataset_path, type)
        self.label_path = os.path.join(dataset_path, type+"_labels")
        examinationUtils.is_path_exist(self.img_path, self.label_path)

        # 获取影像路径下的所有影像名
        self.img_names = osUtils.list_file_name(self.img_path)

        # 检查标签路径是否与影像路径的影像名匹配
        if not examinationUtils.is_file_in_path(self.img_names, self.label_path):
            raise ValueError("影像与标签不匹配！")

        # 初始化参数
        self.crop_size = crop_size
        self.label_processor = ImgUtils.LabelProcessor(class_dict_path)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # 打开影像与标签
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        label = Image.open(os.path.join(self.label_path, img_name)).convert('RGB')

        # 预处理：影像、标签中心裁剪，影像归一化，标签由三通道颜色值转为相应类别的索引
        img, label = ImgUtils.center_crop(self.crop_size, img, label)
        img, label = ImgUtils.img_transform(img, label, self.label_processor)

        return {"img": img, "label": label}

    def __len__(self):
        return len(self.img_names)
