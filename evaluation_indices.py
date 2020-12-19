import numpy as np


def cal_semantic_segmentation_indices(pred, label, confusion_matrix_meter):
    pre_label = pred.max(dim=1)[1].view(-1)
    true_label = label.view(-1)
    confusion_matrix_meter.add(pre_label, true_label)

    cm = confusion_matrix_meter.value()

    res_dict = {
        "OverallAccuracy": cal_overall_accuracy(cm),
        "Kappa": cal_kappa(cm),
        "ClassAccuracy": cal_class_accuracy(cm),
        "ClassPrecision": cal_class_precision(cm),
        "mIoU": cal_miou(cal_iou(cm))
    }

    return res_dict

def cal_overall_accuracy(confusion_matrix):
    return np.trace(confusion_matrix) / (confusion_matrix.sum() + 1e-10)

def cal_class_accuracy(confusion_matrix):
    return np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1e-10)

def cal_class_precision(confusion_matrix):
    return np.diag(confusion_matrix) / (confusion_matrix.sum(axis=0) + 1e-10)

def cal_iou(confusion_matrix):
    denominator = (confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1) - np.diag(confusion_matrix) + 1e-10)
    return np.diag(confusion_matrix) / denominator


def cal_miou(iou):
    return np.nanmean(iou)


def cal_kappa(confusion_matrix):
    pe = (confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1)).sum() / (confusion_matrix.sum() ** 2)
    po = cal_overall_accuracy(confusion_matrix)
    return (po - pe) / (1 - pe)

