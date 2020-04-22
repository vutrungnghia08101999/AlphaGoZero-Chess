import os
from tqdm import tqdm
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

from sl_traning.model import ChessModel
from sl_traning.dataset import FICSDataset
from sl_traning.utils import read_yaml
from sl_traning.loss import loss_func

configs = read_yaml('sl_traning/configs.yml')

def read_data(sets: dict):
    s = []
    for k, v in sets.items():
        with open(os.path.join(configs['dataroot'], v), 'rb') as f:
            a = pickle.load(f)
        s = s + a
    return s

def evaluate(valid_dataloader, model: ChessModel) -> torch.float:
    model.eval()
    n = len(valid_dataloader.dataset)
    print(f'Evaluate on {n} samples')
    corrent_predictions = 0
    for state, valid_actions, y in tqdm(valid_dataloader):
        with torch.no_grad():
            pred = model(state)
        pred = pred * (valid_actions == 1)
        pred = torch.argmax(pred, dim=1)
        corrent_predictions += sum(pred == y)
    
    print(f'Acc: {corrent_predictions}/{n} = {corrent_predictions/len(valid_dataloader.dataset)}')


train_dataset = FICSDataset(read_data(configs['train']))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataset = FICSDataset(read_data(configs['valid']))
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

model = ChessModel()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)

for idx, (state, valid_actions, y) in enumerate(train_dataloader):
    model.train()
    pred = model(state)
    loss = loss_func(pred, valid_actions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx % 10 == 0:
        print(f'Iter {idx}/{len(train_dataloader)} - loss: {loss.item()}')

    if idx % 100 == 0:
        evaluate(valid_dataloader, model)
