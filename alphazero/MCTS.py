import copy
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from chess_rules.TensorBoard import TensorBoard, Board, Move
from mcts.ModeInference import predict
from sl_traning.model import ChessModel

np.random.seed(0)

class Node(object):
    def __init__(self,
                 perspective: int,
                 tensor_board: TensorBoard,
                 model: ChessModel,
                 index: int,
                 is_draw: bool,
                 is_checkmate: bool,
                 parent):
        self.model = model
        self.index = index
        self.perspective = perspective
        self.tensor_board = tensor_board
        self.parent = parent
        self.is_draw = is_draw
        self.is_checkmate = is_checkmate
        self.children = {}
        self.is_expand = False

    def expand(self) -> None:
        assert self.is_expand is False
        self.is_expand = True
        predictions = predict(self.model, self.tensor_board)
        for i in range(len(predictions)):
            move, prob, tensor_board = predictions[i]
            self.children[i] = {
                'node': Node(
                    perspective=self.perspective,
                    tensor_board=tensor_board,
                    model=self.model,
                    index=i,
                    is_draw=tensor_board.is_draw(),
                    is_checkmate=tensor_board.is_checkmate(),
                    parent=self),
                'W': 0,
                'N': 0,
                'Q': 0,
                'P': prob
            }

    def _compute_PUCT(self) -> None:
        N = 0
        for k, dic in self.children.items():
            N += dic['N']
        for k, dic in self.children.items():
            dic['PUCT'] = dic['Q'] + dic['P'] * np.sqrt(N + 1e-8) / (1 + dic['N'])

    def get_best_child(self):
        assert not self.is_checkmate and not self.is_draw and self.is_expand
        assert len(self.children) > 0

        self._compute_PUCT()
        child = None
        PUCT = -1000000.0
        for k, dic in self.children.items():
            if dic['PUCT'] > PUCT:
                PUCT = dic['PUCT']
                child = dic['node']
        assert child is not None
        return child

    def traverse(self):
        current = self
        print('root', end='')
        while not current.is_checkmate and not current.is_draw and current.is_expand:
            current = current.get_best_child()
            print(f' => {current.index}', end='')
        return current

    def backpropagate(self):
        assert self.is_expand
        # future_reward = self.rollout()
        perspective = self.perspective
        if tensor_board.is_checkmate():
            if perspective == tensor_board.turn:
                future_rewards = -1
            else:
                future_rewards = 1
        else:
            future_rewards = 0

        print('Future reward:', future_rewards)

        current = self.parent
        index = self.index
        while current is not None:
            current.children[index]['N'] += 1
            current.children[index]['W'] += future_rewards
            current.children[index]['Q'] = current.children[index]['W'] / current.children[index]['N']
            index = current.index
            current = current.parent

    # def rollout(self, n=1024):
    #     tensor_board = copy.deepcopy(self.tensor_board)
    #     perspective = self.perspective

    #     for simulation in range(n):
    #         if tensor_board.is_draw():
    #             return 0
    #         elif tensor_board.is_checkmate():
    #             if perspective == tensor_board.turn:
    #                 return -1
    #             else:
    #                 return 1
    #         moves = tensor_board.get_valid_moves()
    #         assert len(moves) > 0
    #         r = np.random.randint(0, len(moves))
    #         move = moves[r]
    #         tensor_board = tensor_board.get_next_state(move=move)

    #     return 0


# def sl_and_mcts(tensor_board: TensorBoard, model: ChessModel, n=300):
#     assert len(tensor_board.get_valid_moves()) > 0
#     print(f'PERSPECTIVE: {tensor_board.turn}')
#     print('==============================')
#     model.eval()
#     root = Node(
#         perspective=tensor_board.turn,
#         tensor_board=tensor_board,
#         model=model,
#         index=-1,
#         is_draw=tensor_board.is_draw(),
#         is_checkmate=tensor_board.is_checkmate(),
#         parent=None)
#     root.expand()
#     for idx in range(n):
#         print(f'{idx + 1}/{n}')
#         print('Trajectory: ', end='')
#         best_child = root.traverse()
#         best_child.expand()
#         print(f'\nNo.Children: {len(best_child.children)}')
#         print('Rollout and backpropagate.....')
#         best_child.backpropagate()
#         print('***********************')

#     for k, dic in root.children.items():
#         N = dic['N']
#         W = dic['W']
#         P = dic['P']
#         Q = dic['Q']
#         mv = dic['node'].tensor_board.boards[-1].last_mv
#         print(f'N: {N} - W: {W} - P: {P} - Q: {Q} - Move: {mv}')

#     move = None
#     n_visits = -1
#     for k, dic in root.children.items():
#         if dic['N'] > n_visits:
#             n_visits = dic['N']
#             move = dic['node'].tensor_board.boards[-1].last_mv
#     print(f'Best move: {move}')
#     return move

model = ChessModel()
checkpoint = torch.load(
    '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/SL/best_model_1.pth',
    map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
tensor_board = TensorBoard(Board(), Board(), 1)

root = Node(
    perspective=tensor_board.turn,
    tensor_board=tensor_board,
    model=model,
    index=-1,
    is_draw=tensor_board.is_draw(),
    is_checkmate=tensor_board.is_checkmate(),
    parent=None)
root.expand()
n = 5
for idx in range(n):
    # print(f'{idx + 1}/{n}')
    # print('Trajectory: ', end='')
    best_child = root.traverse()
    best_child.expand()
    # print(f'\nNo.Children: {len(best_child.children)}')
    # print('Rollout and backpropagate.....')
    best_child.backpropagate()
    print('***********************')
