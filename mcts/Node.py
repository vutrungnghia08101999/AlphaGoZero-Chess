import copy
import math
import numpy as np

import torch

from chess_rules.TensorBoard import TensorBoard, Board
from mcts.ModeInference import predict
from sl_traning.model import ChessModel


class Node(object):
    def __init__(self,
                 perspective: int,
                 tensor_board: TensorBoard,
                 model: ChessModel,
                 index: int,
                 is_draw: bool,
                 is_checkmate: bool,
                 parent=None):
        self.model = model
        self.index_in_parent_list = index
        self.perspective = perspective
        self.tensor_board = tensor_board
        self.parent = parent
        self.children = []
        self.is_expand = False
        self.is_draw = is_draw
        self.is_checkmate = is_checkmate

    def expand(self) -> None:
        self.is_expand = True
        predictions = predict(self.model, self.tensor_board)
        for i in range(len(predictions)):
            move, prob, tensor_board = predictions[i]
            self.children.append({
                'node': Node(perspective=self.perspective,
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
            })

    def _compute_PUCT(self):
        N = 0
        for dic in self.children:
            N += dic['N']
        for dic in self.children:
            dic['PUCT'] = dic['Q'] + dic['P'] * np.sqrt(N + 1e-8) / (1 + dic['N'])

    def search_next_move(self) -> 'Node':
        assert self.is_expand is True
        assert len(self.children) > 0
        self._compute_PUCT()
        next_node = None
        PUCT = -10000.0
        for dic in self.children:
            if dic['PUCT'] > PUCT:
                next_node = dic['node']
                PUCT = dic['PUCT']
        return next_node

    def get_best_child(self):
        current = self
        while current.is_expand:
            next = 

# model = ChessModel()
# checkpoint = torch.load(
#     '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/SL/checkpoint_20-04-26_14_06_38_39.pth',
#     map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['state_dict'])
# tensor_board = TensorBoard(Board(), Board(), 1)

# root = Node(1, tensor_board, model, parent=None)

# root.expand()
# len(root.children)
# dic = root.search_next_move()
