import logging
import chess
import time
import copy
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from TensorBoard import TensorChessBoard
from model import ChessModel

def decode(index: int, legal_moves: list) -> str:
    assert index >= 0 and index < 8 * 8 * 64
    move = TensorChessBoard.decode_moves_from_index(index, legal_moves)
    assert move in legal_moves
    return move

def predict_p_and_v(model: ChessModel, tensor_board: TensorChessBoard) -> list:
    # s1 = time.time() * 1000
    encoded_board = tensor_board.encode()
    valid_moves = tensor_board.encode_moves()
    legal_moves = [x.uci() for x in tensor_board.board.legal_moves]
    n_promoted_moves = len([x for x in legal_moves if len(x) == 5])
    n_mvs = len(legal_moves) - n_promoted_moves + n_promoted_moves // 4

    inputs = ToTensor()(encoded_board)
    inputs = inputs.unsqueeze(0)
    # s2 = time.time() * 1000
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        inputs = inputs.type(torch.cuda.FloatTensor)
    else:
        inputs = inputs.type(torch.FloatTensor)

    with torch.no_grad():
        p, v = model(inputs)
    p = p.cpu()
    v = v.cpu()

    # s3 = time.time() * 1000
    p = p.squeeze()
    valid_moves = torch.from_numpy(valid_moves).flatten()
    p = (p + (1e-6)) * (valid_moves == 1)
    probs, indices = torch.topk(p, k=n_mvs, dim=0, largest=True)
    # s4 = time.time() * 1000
    predictions = []
    for i in range(n_mvs):
        # s1 = time.time() * 1000
        move = decode(indices[i].item(), legal_moves)
        # s2 = time.time() * 1000
        prob = probs[i]
        predictions.append((move, prob))
        # s3 = time.time() * 1000
        # print(s2 - s1, s3 - s2)
    # s4 = time.time() * 1000
    # print(s2 - s1, s3 - s2, s4 - s3)
    return predictions, float(v.squeeze())

class MCTSNode(object):
    def __init__(self,
                 model,
                 index: int,
                 parent,
                 perspective: bool,
                 is_game_over: bool,
                 is_checkmate: bool,
                 is_draw: bool,
                 tensor_board=None,  # default is None except root node
                 is_expand=False,
                 temperature=1):
        self.model = model
        self.index = index
        self.parent = parent
        self.tensor_board = tensor_board

        self.children = {}
        self.is_game_over = is_game_over
        self.is_checkmate = is_checkmate
        self.is_draw = is_draw
        self.is_expand = is_expand
        self.perspective = perspective
        if self.parent is None:
            self.temperature = temperature

    def _compute_PUCT(self) -> None:
        N = 0
        for k, dic in self.children.items():
            N += dic['N']
        for k, dic in self.children.items():
            dic['PUCT'] = dic['Q'] + dic['P'] * np.sqrt(N + 1e-8) / (1 + dic['N'])

    def _get_best_child(self):
        assert not self.is_game_over and self.is_expand
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
        # print('=================')
        # print('root', end='')
        while not current.is_game_over and current.is_expand:
            current = current._get_best_child()
            # print(f' => {current.index}', end='')
        # print()
        return current

    def expand_and_backpropagate(self) -> None:
        assert (not self.is_expand and not self.is_game_over) or (not self.is_expand and self.is_game_over)
        v = None
        if not self.is_expand and not self.is_game_over:
            self.is_expand = True
            # s1 = time.time() * 1000
            predictions, v = predict_p_and_v(self.model, self.tensor_board)  # 23.0048828125
            # s2 = time.time() * 1000
            noise = np.random.dirichlet(np.zeros([len(predictions)], dtype=np.float32) + 192)
            for i in range(len(predictions)):
                move, prob = predictions[i]
                if self.parent is None:  # add dirichle noise to the root node
                    prob = 0.7 * prob + 0.3 * noise[i]
                TB = self.tensor_board.get_next_state(move)
                self.children[i] = {
                    'node': MCTSNode(
                        model=self.model,
                        index=i,
                        parent=self,
                        tensor_board=TB,
                        is_game_over=TB.is_game_over,
                        is_checkmate=TB.is_checkmate,
                        is_draw=TB.is_draw,
                        perspective=self.perspective),
                    'move': move,
                    'W': 0,
                    'N': 0,
                    'Q': 0,
                    'P': prob
                }
            # s3 = time.time() * 1000
            # print(s2 - s1, s3 - s2)
        else:
            assert self.is_game_over
            assert (self.is_checkmate and self.is_draw) is False
            if self.is_draw:
                v = 0
                logging.info('case 2.1')
            else:
                if self.perspective == self.tensor_board.turn:
                    v = -1
                    logging.info('case 2.2')
                else:
                    v = 1
                    logging.info('case 2.3')
        assert v is not None
        # print(v)
        current = self.parent
        index = self.index
        while current is not None:
            current.children[index]['N'] += 1
            current.children[index]['W'] += v
            current.children[index]['Q'] = current.children[index]['W'] / current.children[index]['N']
            index = current.index
            current = current.parent

    def get_pi_policy_and_most_visited_move(self) -> np.array:
        assert self.parent is None
        denominator = 0
        for k, dic in self.children.items():
            denominator += dic['N'] ** (1 / self.temperature)
        visit_count = []
        for k, dic in self.children.items():
            numerator = dic['N'] ** (1 / self.temperature)
            visit_count.append((dic['move'], numerator * 1.0 / denominator))

        # get best move
        mv = None
        metric = -1
        for move, f in visit_count:
            if f > metric:
                metric = f
                mv = move

        # for k, dic in self.children.items():
        #     print(dic['move'], dic['N'])
        # print(f'Best move: {mv}')

        # encode pi policy to numpy array size 8 x 8 x 78
        pi_policy = np.zeros((8, 8, 64)) * 0.0
        for move, p in visit_count:
            s = TensorChessBoard.encode_move(move)
            pi_policy = pi_policy + s * (p + 1e-6)
        assert sum(sum(sum(pi_policy != 0))) == sum(sum(sum(self.tensor_board.encode_moves())))
        return pi_policy, mv

# model = ChessModel()
# model.eval()
# tensor_board = TensorChessBoard()
# root = MCTSNode(
#     model=model,
#     index=-1,
#     parent=None,
#     perspective=tensor_board.turn,
#     is_game_over=tensor_board.is_game_over,
#     is_checkmate=tensor_board.is_checkmate,
#     is_draw=tensor_board.is_draw,
#     tensor_board=tensor_board
# )

# for i in tqdm(range(1000)):
#     node = root.traverse()
#     node.expand_and_backpropagate()

# pi, mv = root.get_pi_policy_and_most_visited_move()
