import logging
import pickle
import time
from tqdm import tqdm

import torch

from chess_rules.ChessObjects import Board
from chess_rules.TensorBoard import TensorBoard
from alphazero.model import ChessModel
from alphazero.MCTS import MCTSNode

# torch.random.seed(0)
logging.basicConfig(level=logging.INFO)

def self_play(latest_model: ChessModel, n_moves=512, n_simulation=100):
    dataset = []
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
        dataset.append([
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
            for i in range(len(dataset)):
                if i % 2 == 0:
                    dataset[i].append(value)
                else:
                    dataset[i].append(-1 * value)
            return dataset
        elif tensor_board.is_draw():
            logging.info('Draw')
            logging.info('0 for both team')
            for tup in dataset:
                tup.append(0)
            return dataset
    for tup in dataset:
        tup.append(0)
    logging.info(f'Finish after {n_moves} moves')
    logging.info('0 for both team')
    return dataset


model = ChessModel()
latest_model = model
latest_model.eval()
dataset = self_play(latest_model, 100, 10)
with open('tmp.pkl', 'wb') as f:
    pickle.dump(dataset, f)
# tensor_board = TensorBoard(Board(), Board(), 1)
# root = MCTSNode(
#     tensor_board=tensor_board,
#     model=latest_model,
#     index=-1,
#     parent=None)
# root.expand_and_backpropagate()

# best_child = root.traverse()
# best_child.expand_and_backpropagate()
