import logging
import time
import copy
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from chess_rules.TensorBoard import TensorBoard, Board, Move
from alphazero.model import ChessModel

np.random.seed(0)

def decode(index: int) -> Move:
    assert index >= 0 and index < 8 * 8 * 78
    s = np.zeros((4992))
    s[index] = 1
    s = s.reshape(8, 8, 78)
    move = TensorBoard.decode_tensor_to_moves(s)
    assert len(move) == 1
    return move[0]

def predict_p_and_v(model: ChessModel, tensor_board: TensorBoard) -> list:
    encoded_board, valid_moves, n_mvs = tensor_board.encode_to_tensor()
    encoded_board = torch.tensor(encoded_board, dtype=torch.float)
    encoded_board = encoded_board.view(-1, 8, 8, 104)

    with torch.no_grad():  # 22.450927734375
        p, v = model(encoded_board)

    p = p.squeeze()
    valid_moves = torch.tensor(valid_moves).flatten()
    p = p * (valid_moves == 1) + (1e-6) * (valid_moves == 1)
    probs, indices = torch.topk(p, k=n_mvs, dim=0, largest=True)
    predictions = []
    for i in range(n_mvs):   # 29.221435546875
        move = decode(indices[i])
        s = tensor_board.get_next_state(move)  # 1.255859375
        prob = probs[i]

        predictions.append((move, prob, s))
    return predictions, float(v.squeeze())

def predict_v(model: ChessModel, tensor_board: TensorBoard) -> float:
    encoded_board, _, _ = tensor_board.encode_to_tensor()
    encoded_board = torch.tensor(encoded_board, dtype=torch.float)
    encoded_board = encoded_board.view(-1, 8, 8, 104)
    with torch.no_grad():
        p, v = model(encoded_board)
    return float(v.squeeze())

class MCTSNode(object):
    def __init__(self,
                 tensor_board: TensorBoard,
                 model,
                 index: int,
                 parent,
                 temperature=1):
        self.model = model
        self.index = index
        self.tensor_board = tensor_board
        self.parent = parent
        self.children = {}
        self.is_terminate = False
        self.is_expand = False
        if self.parent is None:
            self.temperature = temperature

    def _compute_PUCT(self) -> None:
        N = 0
        for k, dic in self.children.items():
            N += dic['N']
        for k, dic in self.children.items():
            dic['PUCT'] = dic['Q'] + dic['P'] * np.sqrt(N + 1e-8) / (1 + dic['N'])

    def _get_best_child(self):
        assert not self.is_terminate and self.is_expand
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
        # print('root', end='')
        while not current.is_terminate and current.is_expand:
            current = current._get_best_child()
            # print(f' => {current.index}', end='')
        # print()
        return current

    def expand_and_backpropagate(self) -> None:
        assert (not self.is_expand and not self.is_terminate) or (self.is_expand and self.is_terminate)
        v = None
        # s1 = time.time() * 1000
        if not self.is_expand and not self.is_terminate:
            self.is_expand = True
            self.is_terminate = self.tensor_board.is_terminate()
            if self.is_terminate:
                v = predict_v(self.model, self.tensor_board)
                logging.info('case 1.1')
            else:
                # print('case 1.2')
                # s1 = time.time() * 1000
                predictions, v = predict_p_and_v(self.model, self.tensor_board)
                noise = np.random.dirichlet(np.zeros([len(predictions)], dtype=np.float32) + 192)
                # print(noise)
                for i in range(len(predictions)):
                    move, prob, tensor_board = predictions[i]
                    if self.parent is None:  # add dirichle noise to the root node
                        prob = 0.7 * prob + 0.3 * noise[i]
                        # print(noise[i])
                    self.children[i] = {
                        'node': MCTSNode(
                            tensor_board=tensor_board,
                            model=self.model,
                            index=i,
                            parent=self),
                        'move': move,
                        'W': 0,
                        'N': 0,
                        'Q': 0,
                        'P': prob
                    }
                # s3 = time.time() * 1000
                # print(s2 - s1, s3 - s2)
        else:
            logging.info('case 2')
            v = predict_v(self.model, self.tensor_board)

        # s2 = time.time() * 1000
        # print(v)
        assert v is not None
        current = self.parent
        index = self.index
        while current is not None:
            current.children[index]['N'] += 1
            current.children[index]['W'] += v
            current.children[index]['Q'] = current.children[index]['W'] / current.children[index]['N']
            index = current.index
            current = current.parent
        # s3 = time.time() * 1000
        # print(s2 - s1)
        # print(s3 - s2)

    def get_pi_policy_and_most_visited_move(self) -> np.array:
        assert self.parent is None
        denominator = 0
        for k, dic in self.children.items():
            denominator += dic['N'] ** (1 / self.temperature)
        visit_count = []
        for k, dic in self.children.items():
            numerator = dic['N'] ** (1 / self.temperature)
            visit_count.append((dic['move'], numerator / denominator))

        # for move, f in visit_count:
        #     print('%.4f' % f, move)
        # print(len(visit_count))

        # get best move
        mv = None
        metric = -1
        for move, f in visit_count:
            if f > metric:
                metric = f
                mv = move

        # encode pi policy to numpy array size 8 x 8 x 78
        pi_policy = np.zeros((8, 8, 78)) * 0.0
        for move, p in visit_count:
            s = TensorBoard.encode_action_to_tensor(move)
            pi_policy = pi_policy + s * (p + 1e-6)
        assert sum(sum(sum(pi_policy != 0))) == sum(sum(sum(self.tensor_board.encode_actions_to_tensor())))
        return pi_policy, mv
