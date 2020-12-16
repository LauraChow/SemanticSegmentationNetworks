from torch.utils.tensorboard import SummaryWriter

def print_title(title):
    print(title.center(60, "—"))


def log_(writer, loss, acc, global_step):
    writer.add_scalar('train/Loss', loss, global_step)
    writer.add_scalar('train/Accuracy', acc, global_step)
    writer.flush()
