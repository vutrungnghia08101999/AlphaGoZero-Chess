import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FICSDataset(Dataset):
    def __init__(self, dataset: list):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        state, valid_moves, expect_move = self.dataset[index]
        state = torch.tensor(state, dtype=torch.float)
        valid_moves = torch.tensor(valid_moves.flatten(), dtype=torch.float)
        expect_move = expect_move.flatten()
        s = np.argwhere(expect_move == 1)
        assert s.shape[0] == 1 and s.shape[1] == 1
        expect_move = torch.tensor(s[0][0])
        return state, valid_moves, expect_move
