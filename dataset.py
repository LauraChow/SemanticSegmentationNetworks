import os

from utils import osUtils, examinationUtils, imgUtils

from PIL import Image
from torch.utils.data import Dataset


# 处理Camvid数据集的类
class CamvidDataset(Dataset):
    # 相当于CamvidDataset的构造函数
    def __init__(self, dataset_path="./dataset/CamVid", type="train", crop_size=None):
        examinationUtils.is_path_exist(dataset_path)

        # 获取影像路径与标签路径
        self.img_path = os.path.join(dataset_path, type)
        self.label_path = os.path.join(dataset_path, type+"_labels")
        examinationUtils.is_path_exist(self.img_path, self.label_path)

        # 获取影像路径下的所有影像名
        self.img_names = osUtils.list_file_name(self.img_path)

        # 检查标签路径是否与影像路径的影像名匹配
        if not examinationUtils.is_file_in_path(self.img_names, self.label_path):
            raise ValueError("影像与标签不匹配！")

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # 打开影像与标签
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        label = Image.open(os.path.join(self.label_path, img_name)).convert('RGB')

        # 中心裁剪
        img, label = imgUtils.center_crop(self.crop_size, img, label)


        return {"img": img, "label": label}

    def __len__(self):
        return len(self.img_names)
