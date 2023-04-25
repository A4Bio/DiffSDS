
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_optim_scheduler(lr, epoch, model, steps_per_epoch):
    optimizer = optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()), lr=lr, weight_decay=0.0)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epoch * steps_per_epoch)
    warmup_steps = int(epoch * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, epoch)
    return optimizer, scheduler