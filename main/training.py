import argparse
import logging
import os
import pickle
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from main.model import ChessModel

logging.basicConfig(filename='main/logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str)
parser.add_argument('--modelszoo', type=str)
parser.add_argument('--last_iter', type=int)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=200)
args = parser.parse_args()


logging.info('\n\n********* TRAINING *********\n\n')
logging.info(args._get_kwargs())

class ALPHAZERODataset(Dataset):
    def __init__(self, games: list):
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        state, pi, value = self.games[index]
        state = torch.tensor(state, dtype=torch.float)
        pi = torch.tensor(pi.flatten(), dtype=torch.float)
        value = torch.tensor(value, dtype=torch.float)
        return state, pi, value

def compute_loss(p, v, pi, value, parameters) -> torch.tensor:
    """
    p: batch_size x 4992
    v: batch_size x 1
    pi: batch_size x 4992
    value: batch_size
    """
    v = v.squeeze()
    l1 = (v - value) * (v - value)

    l2 = -1 * pi * torch.log(p)
    l2 = l2.sum(dim=1)

    # reg = torch.tensor(0.0)
    # for param in parameters:
    #     reg += (param ** 2).sum()

    return l1.mean(), l2.mean()  # + 0.0001 * reg

LASTEST_ITER_PATH = os.path.join(args.dataroot, str(args.last_iter + 1))
LASTEST_MODEL_PATH = os.path.join(args.modelszoo, str(args.last_iter) + '.pth')
games = []
for filename in os.listdir(LASTEST_ITER_PATH):
    filepath = os.path.join(LASTEST_ITER_PATH, filename)
    logging.info(f'Load: {filepath}')
    with open(filepath, 'rb') as f:
        game = pickle.load(f)
    games = games + game

dataset = ALPHAZERODataset(games)
logging.info(f'Train on {len(dataset)} positions')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = ChessModel()
if args.last_iter != 0:
    checkpoint = torch.load(LASTEST_MODEL_PATH)
    logging.info(f'Load model: {LASTEST_MODEL_PATH}')
    model.load_state_dict(checkpoint['state_dict'])

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0.0002,
                             amsgrad=False)

for epoch in range(args.epochs):
    logging.info(f'EPOCH: {epoch}')
    for batch_id, (state, pi, value) in enumerate(dataloader):
        p, v = model(state)
        l1, l2 = compute_loss(p, v, pi, value, model.parameters())
        loss = l1 + l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            logging.info(f'value loss: {l1.item()} - prob loss: {l2.item()}')

OUT_MODEL_PATH = os.path.join(args.modelszoo, str(args.last_iter + 1) + '.pth')
torch.save({'state_dict': model.state_dict()}, OUT_MODEL_PATH)
logging.info(f'Save model at: {OUT_MODEL_PATH}')
logging.info(f'Completed training')
