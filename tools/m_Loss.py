import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import itertools

EPS = 1e-2
esp = 1e-8


class Fidelity_Loss(torch.nn.Module):
    # 计算一对数之间的 loss
    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) +
                    torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)


class Fidelity_Loss_distortion(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss_distortion, self).__init__()

    def forward(self, p, g):
        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = torch.sqrt(p_i * g_i + esp)
            loss = loss + loss_i
        loss = 1 - loss
        loss = loss / p.size(1)
        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Multi_Fidelity_Loss, self).__init__()

    def forward(self, p, g):

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + esp) +
                          torch.sqrt((1 - p_i) * (1 - g_i) + esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)


eps = 1e-12


def loss_m(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #

    y_pred = y_pred.unsqueeze(1)
    y = y.unsqueeze(1)
    preds = y_pred-y_pred.t()
    gts = y - y.t()

    # signed = torch.sign(gts)
    # 返回上三角矩阵的行列坐标，一共两行，第一行位行坐标，第二行为列坐标
    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    g = 0.5 * (torch.sign(gts) + 1)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean(
        (1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss
