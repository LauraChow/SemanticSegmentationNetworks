import pandas as pd
import torch as t
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as ff
from PIL import Image


class ImgProcessor:
    @staticmethod
    def center_crop(crop_size, *imgs):
        cropped = []
        for img in imgs:
            cropped.append(ff.center_crop(img, crop_size))
        return tuple(cropped)

    @staticmethod
    def img_transform(img, label, label_processor):
        # 影像处理
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)


        # 标签处理
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


class LabelProcessor:
    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return np.array(colormap).astype('uint8')

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    # 三通道影像转label标签
    def encode_label_img(self, img):
        data = np.array(img, dtype='uint8')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='long')

    # label标签转三通道影像
    def decode_label_img(self, img):
        # 压缩batch通道
        img = img.max(dim=1)[1].squeeze()
        # 转成numpy
        img = img.cpu().data.numpy()
        # 转成3通道numpy
        img = self.colormap[img]
        # 转成影像
        img = Image.fromarray(img)
        return img
