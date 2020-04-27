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
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

configs = read_yaml('configs.yml')
logging.info('\n\n========== ALPHAZERO ========\n\n')
logging.info(configs)

def read_data(sets: list):
    s = []
    for batch in sets:
        path = os.path.join(configs['root'], batch)
        logging.info(f'Load {path}')
        with open(path, 'rb') as f:
            a = pickle.load(f)
        for k, v in a.items():
            s = s + v
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
    # pred = pred + (valid_moves == 0) * (-20)
    loss_func = nn.CrossEntropyLoss()
    return loss_func(pred, y)

train_dataset = FICSDataset(read_data(configs['train']))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_test_dataset = FICSDataset(read_data(configs['train_test']))
train_test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = FICSDataset(read_data(configs['valid']))
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

model = ChessModel()
model = model.cuda()
if configs['resume']:
    checkpoint = torch.load(configs['resume'])
    model.load_state_dict(checkpoint['state_dict'])

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)

evaluate(valid_dataloader, model)
for epoch in range(50):
    logging.info('===================')
    logging.info(f'EPOCH: {epoch}')
    for (state, valid_actions, y) in tqdm(train_dataloader):
        model.train()
        state = state.cuda()
        valid_actions = valid_actions.cuda()
        y = y.cuda()
        pred = model(state)
        loss = loss_func(pred, valid_actions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x = datetime.datetime.now()
    time = x.strftime("%y-%m-%d_%H:%M:%S")
    model_checkpoint = os.path.join('/content/gdrive/My Drive/MOUNT/models', f'checkpoint_{time}_{epoch}.pth')
    logging.info(model_checkpoint)
    with torch.no_grad():
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_checkpoint)
    evaluate(train_test_dataloader, model)
    evaluate(valid_dataloader, model)
