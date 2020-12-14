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

# 数据集参数
DATASET = ['CamVid', 12]
CHANNEL_NUM = 3
CROP_SIZE = (352, 480)
ORIGIN_SIZE = (360, 480)

# 训练参数
EPOCH_NUM = 200
BATCH_SIZE = 4
WORKERS_NUM = 4
LEARNING_RATE = .01
