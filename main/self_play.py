import argparse
import logging
import os
import numpy as np
import pickle
import time
from tqdm import tqdm
import copy

import torch

from chess_rules.ChessObjects import Board
from chess_rules.TensorBoard import TensorBoard
from main.MCTS import MCTSNode
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
parser.add_argument('--game_id', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--n_moves', type=int, default=512)
parser.add_argument('--n_simulation', type=int, default=400)
args = parser.parse_args()

np.random.seed(args.seed)

logging.info('\n\n********* SELF PLAY *********\n\n')
logging.info(args._get_kwargs())

def save_file(game: list, filepath: str):
    logging.info(f'Save game at {filepath}')
    with open(filepath, 'wb') as f:
        pickle.dump(game, f)

def self_play(latest_model: ChessModel, game_id: int, iter_path: str, n_moves=512, n_simulation=400) -> None:
    game = []
    filepath = os.path.join(iter_path, str(game_id) + '.pkl')
    # latest_model.eval()
    tensor_board = TensorBoard(Board(), Board(), 1)
    # tensor_board.boards[-1].display()
    latest_model.eval()
    logging.info('1-white-move first')
    for move in tqdm(range(n_moves)):
        # logging.info(f'Turn: {tensor_board.turn}')
        root = MCTSNode(
            tensor_board=tensor_board,
            model=latest_model,
            index=-1,
            perspective=tensor_board.turn,
            parent=None)
        root.expand_and_backpropagate()

        # logging.info(f'Run {n_simulation} simulations in MCTS')
        for idx in range(n_simulation):  # tqdm(range(n_simulation)):
            best_child = root.traverse()
            best_child.expand_and_backpropagate()
        pi_policy, mv = root.get_pi_policy_and_most_visited_move()
        board, _, _ = tensor_board.encode_to_tensor()
        game.append([
            board,
            pi_policy
        ])
        # logging.info(f'Most visit move: {mv}')
        # moves = tensor_board.get_valid_moves()
        # print(len(moves))
        # for move in moves:
        #     print(move)
        tensor_board = tensor_board.get_next_state(mv)
        # tensor_board.boards[-1].display()
        if tensor_board.is_checkmate():
            logging.info(f'{abs(1 - tensor_board.turn)} won')
            value = (tensor_board.turn == 1) * (-1) + (tensor_board.turn == 0) * 1
            logging.info(f'{value} for tem 1 and {-1*value} for team 0')
            for i in range(len(game)):
                if i % 2 == 0:
                    game[i].append(value)
                else:
                    game[i].append(-1 * value)
            save_file(game, filepath=filepath)
            return
        elif tensor_board.is_draw():
            logging.info('Draw')
            logging.info('0 for both team')
            for tup in game:
                tup.append(0)
            save_file(game, filepath=filepath)
            return
    for tup in game:
        tup.append(0)
    logging.info(f'Finish after {n_moves} moves')
    logging.info('0 for both team')
    save_file(game, filepath=filepath)
    return

model = ChessModel()
if args.last_iter != 0:
    path = os.path.join(args.modelszoo, str(args.last_iter) + '.pth')
    checkpoint = torch.load(path)
    logging.info(f'Load model: {path}')
    model.load_state_dict(checkpoint['state_dict'])
model.eval()

iter_path = os.path.join(args.dataroot, str(args.last_iter + 1))
os.makedirs(iter_path, exist_ok=True)

self_play(model, args.game_id, iter_path, args.n_moves, args.n_simulation)
logging.info(f'Completed self play - game_id: {args.game_id}\n\n')
