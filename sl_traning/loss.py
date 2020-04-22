import torch
import torch.nn as nn


def loss_func(pred: torch.tensor, valid_moves: torch.tensor, y: torch.tensor) -> torch.float:
    """
    pred: N x C
    valid_moves: N x C
    y: N
    """
    pred = pred * (valid_moves == 1)
    loss_func = nn.CrossEntropyLoss()
    return loss_func(pred, y)
