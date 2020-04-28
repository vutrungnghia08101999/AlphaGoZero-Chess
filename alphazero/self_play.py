import logging
import os
import pickle
import time
from tqdm import tqdm

import torch

from chess_rules.ChessObjects import Board
from chess_rules.TensorBoard import TensorBoard
from alphazero.model import ChessModel
from alphazero.MCTS import MCTSNode
from alphazero.utils import read_yaml

logging.basicConfig(level=logging.INFO)

configs = read_yaml('alphazero/configs.yml')


def self_play(latest_model: ChessModel, n_moves=512, n_simulation=100):
    game = []
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
            return game
        elif tensor_board.is_draw():
            logging.info('Draw')
            logging.info('0 for both team')
            for tup in game:
                tup.append(0)
            return game
    for tup in game:
        tup.append(0)
    logging.info(f'Finish after {n_moves} moves')
    logging.info('0 for both team')
    return game


model = ChessModel()
checkpoint = torch.load(os.path.join(configs['modelsroot'], str(configs['last_iter']) + '.pth'))
model.load_state_dict(checkpoint['state_dict'])

games = {}
for i in range(configs['self_play']['n_games']):
    logging.info(f'Self playing game: {i}')
    game = self_play(latest_model=model,
                     n_moves=configs['self_play']['n_moves'],
                     n_simulation=configs['self_play']['n_simulation'])
    games[i] = game

outfile = os.path.join(configs['dataroot'], str(configs['last_iter'] + 1) + '.pkl')
logging.info(f'Save data at: {outfile}')
assert not os.path.isfile(outfile)
with open(outfile, 'wb') as f:
    pickle.dump(games, f)

logging.info('Completed self play\n\n')

# import datetime
# # games = {}
# model = ChessModel()
# model.state_dict()
# x = datetime.datetime.now()
# time = x.strftime("%y_%m_%d_%H_%M_%S")
# model_checkpoint = f'/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL/{time}_0.pth'
# torch.save({'epoch': 0, 'state_dict': model.state_dict()}, model_checkpoint)
# latest_model = model
# latest_model.eval()
# game = self_play(latest_model, 512, 200)
# games[0] = game
# with open('tmp.pkl', 'wb') as f:
#     pickle.dump(games, f)

# tensor_board = TensorBoard(Board(), Board(), 1)
# root = MCTSNode(
#     tensor_board=tensor_board,
#     model=latest_model,
#     index=-1,
#     parent=None)
# root.expand_and_backpropagate()

# best_child = root.traverse()
# best_child.expand_and_backpropagate()
