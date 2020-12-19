import torch as t

import torchvision.transforms as transforms
import torchvision.transforms.functional as ff

import numpy as np
from PIL import Image


def center_crop(crop_size, *imgs):
    cropped = []
    for img in imgs:
        cropped.append(ff.center_crop(img, crop_size))
    return tuple(cropped)


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
