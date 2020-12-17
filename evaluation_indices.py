import numpy as np

def cal_semantic_segmentation_indices(pred, label, loss, loss_meter, confusion_matrix_meter):
    loss_meter.add(loss.item())

    pre_label = pred.max(dim=1)[1].view(-1)
    true_label = label.view(-1)
    confusion_matrix_meter.add(pre_label, true_label)

    cm = confusion_matrix_meter.value()
    accuracy = np.trace(cm) / np.nansum(cm)

    return loss_meter.value()[0], accuracy