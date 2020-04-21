import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from TensorBoard import TensorBoard
from chess_rules.Rules import Rules
from chess_rules.ChessObjects import (
    Board,
    Move,
    Spot,
    Pawn,
    King,
    Queen,
    Rook,
    Bishop,
    Knight
)

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

from supervised_learning.utils import read_yaml

configs = read_yaml('supervised_learning/configs.yml')

processed_data = read_yaml(configs['preprocess_data']['output'])

# n = 0
# for k, v in dataset.items():
#     n += len(v)

# a, b, c, d = [], [], [], []
# e = []
# for key, match in dataset.items():
#     for move in match:
#         if move == 'O-O-O' or move == 'O-O':
#             e.append(move)
#         else:
#             if len(move) == 2:
#                 c.append(move[-2]), d.append(move[-1])
#             elif len(move) == 3:
#                 b.append(move[-3]), c.append(move[-2]), d.append(move[-1])
#             elif len(move) == 4:
#                 a.append(move[-4]), b.append(move[-3]), c.append(move[-2]), d.append(move[-1])
#             else:
#                 raise RuntimeError('Fuck')

# assert len(d) + len(e) == n
# a, b, c, d, e = pd.Series(a).unique(), pd.Series(b).unique(), pd.Series(c).unique(), pd.Series(d).unique(), pd.Series(e).unique()

def notation_to_move(turn: int, board: Board, notation: str) -> Move:
    ranks = {'8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7}
    files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    pieces = ['B', 'N', 'Q', 'R', 'K']

    def is_valid_move(mv: Move, moves: list) -> bool:
        for move in moves:
            if mv == move:
                return True
        return False

    def get_class(c: str):
        if c == 'B':
            return Bishop
        if c == 'N':
            return Knight
        if c == 'R':
            return Rook
        if c == 'K':
            return King
        if c == 'Q':
            return Queen
        raise RuntimeError(f'{c} is null or pawn')

    valid_moves = Rules.get_all_valid_moves(turn, board)
    if notation == 'O-O-O' or notation == 'O-O':
        s_row = abs(1 - turn) * 0 + abs(0 - turn) * 7
        start = Spot(s_row, 4)
        if notation == 'O-O-O':
            end = Spot(s_row, 2)
        else:
            end = Spot(s_row, 6)
        move = Move(start, end, is_castling=True)
        if is_valid_move(move, valid_moves):
            return move
        raise RuntimeError(f'invalid move: {move}')
    elif len(notation) == 2:
        end_row, end_col = ranks[notation[-1]], files[notation[-2]]
        for move in valid_moves:
            start = move.start
            end = move.end
            if end.row == end_row and end.col == end_col and isinstance(board.board[start.row][start.col], Pawn):
                return move
        raise RuntimeError(f'move not found: {notation}')
    elif len(notation) == 3:
        end_row, end_col, c = ranks[notation[-1]], files[notation[-2]], notation[-3]
        if c in pieces:
            c = get_class(c)
            for move in valid_moves:
                start, end = move.start, move.end
                if end.row == end_row and end.col == end_col and isinstance(board.board[start.row][start.col], c):
                    return move
            raise RuntimeError(f'move not found: {notation}')
        elif c in ranks.keys():
            raise RuntimeError(f'Impossible move: {notation}')
        elif c in files.keys():
            for move in valid_moves:
                start, end = move.start, move.end
                if end.row == end_row and end.col == end_col:
                    if isinstance(board.board[start.row][start.col], Pawn) and start.col == files[c]:
                        return move
            raise RuntimeError(f'Impossible move: {notation}')
        else:
            raise RuntimeError(f'move not found: {notation}')
    elif len(notation) == 4:
        end_row, end_col, c, piece = ranks[notation[-1]], files[notation[-2]], notation[-3], notation[-4]
        if c in ranks.keys():
            for move in valid_moves:
                start, end = move.start, move.end
                if end.row == end_row and end.col == end_col:
                    if isinstance(board.board[start.row][start.col], get_class(piece)) and start.row == ranks[c]:
                        return move
            raise RuntimeError(f'move not found: {notation}')
        elif c in files.keys():
            for move in valid_moves:
                start, end = move.start, move.end
                if end.row == end_row and end.col == end_col:
                    if isinstance(board.board[start.row][start.col], get_class(piece)) and start.col == files[c]:
                        return move
            raise RuntimeError(f'move not found: {notation}')
        else:
            raise RuntimeError(f'move not found: {notation}')
    else:
        raise RuntimeError(f'move not found: {notation}')

# import time
# s = dataset[1000]
# turn = 1
# board = Board()
# move = notation_to_move(turn, board, s[0])
# for i in range(len(s)):
#     turn = abs(1 - (i % 2))
#     move = notation_to_move(turn, board, s[i])
#     board = Rules.get_next_state(move, board)
#     board.display()
#     print(move)
#     time.sleep(5)

dataset = []
# for key, match in processed_data.items():
tensor_board = TensorBoard(Board(), Board(), 1)
for notation in tqdm(processed_data[0]):
    move = notation_to_move(tensor_board.turn, tensor_board.boards[-1], notation)
    expect_action = TensorBoard.encode_action_to_tensor(move)
    state = tensor_board.encode_board_to_tensor()
    valid_actions = tensor_board.encode_actions_to_tensor()
    dataset.append((state, valid_actions, expect_action))
    tensor_board = tensor_board.get_next_state(move)


dataset = dataset[0:10]
for i in range(len(dataset)):
    x = TensorBoard.decode_board_tensor_to_board(dataset[i][0])
    x['board'].display()
    print('n_mvs: ' + str(x['board'].n_mvs))
    print('turn: ' + str(x['turn']))
    a = TensorBoard.decode_actions_tensor_to_moves(dataset[i][1])
    for u in a:
        print(u)
    print('****')
    b = TensorBoard.decode_actions_tensor_to_moves(dataset[i][2])
    for u in b:
        print(u)
    print('========================')
