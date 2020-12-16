import os
from torch.utils.tensorboard import SummaryWriter

def print_title(title):
    print(title.center(60, "—"))


def log_indices(writer, loss, acc, global_step, type="Train"):
    writer.add_scalar(type+"/Loss", loss, global_step)
    writer.add_scalar(type+"/Accuracy", acc, global_step)
    writer.flush()
