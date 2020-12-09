'''
    配置文件，包含以下信息：
    · 数据集路径
        ·  所有数据集均存放在dataset路径下，位于不同文件夹
        ·  所有数据集的目录结构均为：
            train
            train_labels
            val
            val_labels
            test
            test_labels
            class_dict.csv
    · 影像处理桉树
    · 训练参数
'''

# 数据集路径
DATASET_PATH = "./dataset/CamVid"
CLASS_DICT_PATH = "./dataset/CamVid/class_dict.csv"

# 影像处理参数
CROP_SIZE = (352, 480)
ORIGIN_SIZE = (360, 480)

# 训练参数
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 0.01