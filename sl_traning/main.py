import os
from tqdm import tqdm
import datetime
import pickle
import logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import torch
from torch.utils.data import DataLoader

from model import ChessModel
from dataset import FICSDataset
from utils import read_yaml

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

configs = read_yaml('configs.yml')


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
    logging.info(f'Evaluate on {n} samples')
    corrent_predictions = 0
    for state, valid_actions, y in tqdm(valid_dataloader):
        state, valid_actions, y = state.cuda(), valid_actions.cuda(), y.cuda()
        with torch.no_grad():
            pred = F.softmax(model(state), dim=1)
        pred = pred * (valid_actions == 1)
        pred = torch.argmax(pred, dim=1)
        corrent_predictions += sum(pred == y)
    
    logging.info(f'Acc: {corrent_predictions}/{n} = {corrent_predictions * 1.0/n}')
    
def loss_func(pred: torch.tensor, valid_moves: torch.tensor, y: torch.tensor) -> torch.float:
    """
    pred: N x C
    valid_moves: N x C
    y: N
    """
#     pred = pred * (valid_moves == 1)
    loss_func = nn.CrossEntropyLoss()
    return loss_func(pred, y)

train_dataset = FICSDataset(read_data(configs['train']))
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataset = FICSDataset(read_data(configs['valid']))
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

model = ChessModel()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)

for epoch in range(10):
    logging.info('===================')
    logging.info(f'EPOCH: {epoch}')
    for idx, (state, valid_actions, y) in enumerate(train_dataloader):
        model.train()
        state = state.cuda()
        valid_actions = valid_actions.cuda()
        y = y.cuda()
        pred = model(state)
        loss = loss_func(pred, valid_actions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            logging.info(f'Iter {idx}/{len(train_dataloader)} - loss: {loss.item()}')

        if idx % 100 == 0:
            evaluate(valid_dataloader, model)
            x = datetime.datetime.now()
            time = x.strftime("%y-%m-%d_%H:%M:%S")
            model_checkpoint = os.path.join('models', f'checkpoint_{time}_{epoch}_{idx}.pth')
            with torch.no_grad():
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_checkpoint)
    evaluate(valid_dataloader, model)
