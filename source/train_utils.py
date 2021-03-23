import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def follow_shape(model):
    def hook_shape(self, input, output):
        print('the layer is: ', self)
        print('input shape is: ', len(input), input[0].shape)
        print('output shape is: ', output.shape)

    for i in model.features.modules():
        if isinstance(i, nn.Conv2d) or isinstance(i, nn.Conv1d):
            i.register_forward_hook(hook_shape)

def even_intervals(nepochs, ninterval=1):
    """Divide a number of epochs into regular intervals."""
    out = list(np.linspace(0, nepochs, num=ninterval, endpoint=False, dtype=int))[1:]
    return out
