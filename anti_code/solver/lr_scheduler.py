# encoding: utf-8
from bisect import bisect_right
import torch
import math
import torch.optim as optim

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
def exp_warmup(base_value, max_warmup_iter, cur_step):
    """exponential warmup proposed in mean teacher

    calcurate
    base_value * exp(-5(1 - t)^2), t = cur_step / max_warmup_iter

    Parameters
    -----
    base_value: float
        maximum value
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    if max_warmup_iter <= cur_step:
        return base_value
    return base_value * math.exp(-5 * (1 - cur_step/max_warmup_iter)**2)


def linear_warmup(base_value, max_warmup_iter, cur_step):
    """linear warmup

    calcurate
    base_value * (cur_step / max_warmup_iter)
    
    Parameters
    -----
    base_value: float
        maximum value
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    if max_warmup_iter <= cur_step:
        return base_value
    return base_value * cur_step / max_warmup_iter


def cosine_decay(base_lr, max_iteration, cur_step):
    """cosine learning rate decay
    
    cosine learning rate decay with parameters proposed FixMatch
    base_lr * cos( (7\pi cur_step) / (16 max_warmup_iter) )

    Parameters
    -----
    base_lr: float
        maximum learning rate
    max_warmup_iter: int
        maximum warmup iteration
    cur_step: int
        current iteration
    """
    return base_lr * (math.cos( (7*math.pi*cur_step) / (16*max_iteration) ))

'''
def CosineAnnealingLR(optimizer, max_iteration):
    """
    generate cosine annealing learning rate scheduler as LambdaLR
    """
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda cur_step : math.cos((7*math.pi*cur_step) / (16*max_iteration)))
'''

def CosineAnnealingLR(optimizer, max_iteration,num_warmup_steps=0,num_cycles=7./16.,last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, max_iteration - num_warmup_steps))
        no_progress = min(1,max(no_progress, 0))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)