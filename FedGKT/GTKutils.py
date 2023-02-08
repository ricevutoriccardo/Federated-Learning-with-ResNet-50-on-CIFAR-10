import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0
        self.list = []

    def update(self, val):
        self.total += val
        self.steps += 1
        self.list.append(val)

    def value(self):
        return self.total / float(self.steps)

    def get_list(self):
        return (np.array(self.list) / self.steps).tolist()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
       
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
      
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss

