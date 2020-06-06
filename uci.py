import argparse
import pkg_resources.py2_warn
import chess
import logging
from tqdm import tqdm
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch

torch.set_num_threads(1)
PATH = '/home/vutrungnghia/chess-engine/models/7.pth'

mapping_position = {
    'k': 0, 'q': 1, 'r': 2, 'b': 3, 'n': 4, 'p': 5,
    'K': 6, 'Q': 7, 'R': 8, 'B': 9, 'N': 10, 'P': 11
}

mapping_uci_index = {
    '8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7,
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7
}

mapping_index_uci_row = {
    0: '8', 1: '7', 2: '6', 3: '5', 4: '4', 5: '3', 6: '2', 7: '1'
}

mapping_index_uci_col = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'
}

class TensorChessBoard(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.is_checkmate = self.board.is_checkmate()
        self.is_game_over = self.board.is_game_over()
        self.is_draw = self.is_game_over and (not self.is_checkmate)
        self.turn = self.board.turn

    def encode_board(self):
        board_str = str(self.board)
        tensor = np.zeros((8, 8, 12))
        s = board_str.split('\n')
        s = [x.split(' ') for x in s]
        s = np.array(s)
        for row in range(8):
            for col in range(8):
                if s[row][col] == '.':
                    continue
                c = mapping_position[s[row][col]]
                tensor[row][col][c] = 1
        return tensor

    def encode_moves(self):
        tensor = np.zeros((8, 8, 64))
        moves = [x.uci() for x in self.board.legal_moves]
        for move in moves:
            start = move[0:2]
            end = move[2:4]
            start_row = mapping_uci_index[start[1]]
            start_col = mapping_uci_index[start[0]]

            end_row = mapping_uci_index[end[1]]
            end_col = mapping_uci_index[end[0]]

            c = end_row * 8 + end_col
            tensor[start_row][start_col][c] = 1
        return tensor

    def encode(self):
        s1 = self.encode_board()
        s2 = self.encode_moves()
        turn = np.ones((8, 8, 1)) * self.board.turn
        return np.concatenate((s1, s2, turn), axis=2)

    def get_next_state(self, move):
        board = copy.deepcopy(self.board)
        board.push_uci(move)
        return TensorChessBoard(board=board)

    @staticmethod
    def decode_moves(tensor: np.array, board: chess.Board):
        """[summary]

        Arguments:
            tensor {np.array} -- 8 x 8 x 64
        """
        s = np.argwhere(tensor == 1)
        moves_uci = []
        legal_moves = [x.uci() for x in board.legal_moves]
        promoted_moves = [x for x in legal_moves if len(x) == 5]
        for start_row, start_col, c in s:
            end_row = c // 8
            end_col = c % 8
            start = mapping_index_uci_col[start_col] + mapping_index_uci_row[start_row]
            end = mapping_index_uci_col[end_col] + mapping_index_uci_row[end_row]
            move = start + end
            for mv in promoted_moves:
                if move in mv:
                    move = move + 'q'
                    break
            moves_uci.append(move)
        return moves_uci

    def display(self):
        s = str(self.board)
        s = s.split('\n')
        for u in s:
            print(u)
        print()

    @staticmethod
    def decode_moves_from_index(index: int, legal_moves: list):
        """[summary]
        index = 512 * row + 64 * col + channel
        index in {0, 1, 2, ... 4095}
        row, col in {0, 1, 2, ... 7}
        channel in {0, 1, 2, ... 63}
        Arguments:
            tensor {np.array} -- 8 x 8 x 64
        """
        s1 = time.time() * 1000
        start_row = index // 512
        start_col = (index % 512) // 64
        channel = (index % 512) % 64
        end_row = channel // 8
        end_col = channel % 8
        s2 = time.time() * 1000
        promoted_moves = [x for x in legal_moves if len(x) == 5]
        start = mapping_index_uci_col[start_col] + mapping_index_uci_row[start_row]
        end = mapping_index_uci_col[end_col] + mapping_index_uci_row[end_row]
        move = start + end
        for mv in promoted_moves:
            if move in mv:
                move = move + 'q'
                break
        return move

    @staticmethod
    def encode_move(move: str):
        tensor = np.zeros((8, 8, 64))
        start = move[0:2]
        end = move[2:4]
        start_row = mapping_uci_index[start[1]]
        start_col = mapping_uci_index[start[0]]

        end_row = mapping_uci_index[end[1]]
        end_col = mapping_uci_index[end[0]]

        c = end_row * 8 + end_col
        tensor[start_row][start_col][c] = 1
        return tensor

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=77, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.batchNorm = nn.BatchNorm2d(num_features=256)

    def forward(self, s):
        s = F.relu(self.batchNorm(self.conv(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.BatchNorm2d1(out))
        out = self.conv2(out)
        out = self.BatchNorm2d2(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # policy block
        self.conv1 = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=1)
        self.batchNorm2d1 = nn.BatchNorm2d(2)
        # self.dropOut = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * 8 * 2, 8 * 8 * 64)

        # value block
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1, stride=1)  # value head
        self.batchNorm2d2 = nn.BatchNorm2d(1)
        self.fc2 = nn.Linear(8 * 8 * 1, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s):
        # policy block
        p = F.relu(self.batchNorm2d1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 2)
        p = self.fc1(p)
        p = F.softmax(p, dim=1)

        # value block
        v = F.relu(self.batchNorm2d2(self.conv2(s)))
        v = v.view(-1, 8 * 8 * 1)  # batch_size X channel X height X width
        v = F.relu(self.fc2(v))
        v = torch.tanh(self.fc3(v))
        return p, v

class ChessModel(nn.Module):
    def __init__(self, n_blocks=5):
        super(ChessModel, self).__init__()
        self.n_blocks = n_blocks
        self.conv = ConvBlock()
        for block in range(self.n_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(self.n_blocks):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s)
        return p, v

def decode(index: int, legal_moves: list) -> str:
    assert index >= 0 and index < 8 * 8 * 64
    move = TensorChessBoard.decode_moves_from_index(index, legal_moves)
    assert move in legal_moves
    return move

def predict_p_and_v(model: ChessModel, tensor_board: TensorChessBoard) -> list:
    # s1 = time.time() * 1000
    # print("CCCCCCCCCCCCCCCCCCCCCCCCCC")
    encoded_board = tensor_board.encode()
    valid_moves = tensor_board.encode_moves()
    legal_moves = [x.uci() for x in tensor_board.board.legal_moves]
    n_promoted_moves = len([x for x in legal_moves if len(x) == 5])
    n_mvs = len(legal_moves) - n_promoted_moves + n_promoted_moves // 4

    # inputs = ToTensor()(encoded_board)  # using transpose
    inputs = encoded_board.transpose(2, 0, 1)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.unsqueeze(0)
    # s2 = time.time() * 1000
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        inputs = inputs.type(torch.cuda.FloatTensor)
    else:
        inputs = inputs.type(torch.FloatTensor)

    # print("DDDDDDDDDDDDDDDDDDDDDDDDDDD")
    with torch.no_grad():
        p, v = model(inputs)
    # print("EEEEEEEEEEEEEEEEEEEEEEEEEEE")
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
    # print('FFFFFFFFFFFFFFFFFFFFFFFFFF')
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
            # print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
            predictions, v = predict_p_and_v(self.model, self.tensor_board)  # 23.0048828125
            # print("BBBBBBBBBBBBBBBBBBBBBBBBBB")
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

        for k, dic in self.children.items():
            print(dic['move'], dic['N'])
        print(f'Best move: {mv}')

        # encode pi policy to numpy array size 8 x 8 x 78
        pi_policy = np.zeros((8, 8, 64)) * 0.0
        for move, p in visit_count:
            s = TensorChessBoard.encode_move(move)
            pi_policy = pi_policy + s * (p + 1e-6)
        assert sum(sum(sum(pi_policy != 0))) == sum(sum(sum(self.tensor_board.encode_moves())))
        return pi_policy, mv

model = ChessModel()
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
print(f"Load movel at: {PATH}")
model.eval()

def search_move(fen_representation: str, n_simulations=256):
    board = chess.Board(fen_representation)
    tensor_board = TensorChessBoard(board=board)
    root = MCTSNode(
        model=model,
        index=-1,
        parent=None,
        perspective=tensor_board.turn,
        is_game_over=tensor_board.is_game_over,
        is_checkmate=tensor_board.is_checkmate,
        is_draw=tensor_board.is_draw,
        tensor_board=tensor_board)
    print(f'Run {n_simulations} simulations in MCTS')
    for idx in tqdm(range(n_simulations)):
        best_child = root.traverse()
        best_child.expand_and_backpropagate()
    _, mv = root.get_pi_policy_and_most_visited_move()
    tensor_board = tensor_board.get_next_state(mv)
    return tensor_board.board.fen(), mv

# ********************** UCI ********************
board = None

def input_UCI():
    print("id name AlphaZero")
    print("id author Lusheeta")
    print("uciok")

def is_ready():
    print("readyok")

def new_game():
    pass

def go():
    global board
    _, move = search_move(board.fen())
    print(f"bestmove {move}")

def quit():
    print("Good game")

def new_position(input_string: str):
    global board
    board = chess.Board()

    if "moves" in input_string:
        moves = input_string.split(' ')[3:]
        for move in moves:
            board.push_uci(move)

# input_string = "position startpos moves e2e4 g7g6 f1c4 f8g7 g1f3 e7e6 d2d4 g7f6 d1d2 b7b6"

while True:
    input_string = input()
    if input_string == "uci":
        input_UCI()
    elif input_string == "isready":
        is_ready()
    elif input_string == "ucinewgame":
        new_game()
    elif input_string.startswith("position"):
        new_position(input_string)
    elif input_string.startswith("go"):
        go()
    elif input_string == "quit":
        quit()
        break
print('Finished')
