import argparse
import pandas as pd
import logging
import pickle
import os
import numpy as np
from tqdm import tqdm

from chess_rules.TensorBoard import TensorBoard
from chess_rules.Rules import Rules
from sl_data_processing.utils import read_yaml
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

logging.basicConfig(filename='sl_data_processing/logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str)
args = parser.parse_args()
configs = read_yaml('sl_data_processing/configs.yml')

assert 'black' in args.time or 'white' in args.time
with open(os.path.join(configs['encode_data']['input'], args.time + '.pkl'), 'rb') as f:
    processed_data = pickle.load(f)
OUTPUT = os.path.join(configs['encode_data']['output'], args.time)
os.makedirs(OUTPUT, exist_ok=True)

logging.info(configs['encode_data']['input'])
logging.info(f'TIME: {args.time}')
logging.info(f'OUTPUT: {OUTPUT}')

n = 0
for game in processed_data:
    n += len(game)

a, b, c, d = [], [], [], []
e = []
for game in processed_data:
    for move in game:
        if move == 'O-O-O' or move == 'O-O':
            e.append(move)
        else:
            if len(move) == 2:
                c.append(move[-2]), d.append(move[-1])
            elif len(move) == 3:
                b.append(move[-3]), c.append(move[-2]), d.append(move[-1])
            elif len(move) == 4:
                a.append(move[-4]), b.append(move[-3]), c.append(move[-2]), d.append(move[-1])
            else:
                raise RuntimeError(f'{move}')

assert len(d) + len(e) == n
a, b, c, d, e = pd.Series(a).unique(), pd.Series(b).unique(), pd.Series(c).unique(), pd.Series(d).unique(), pd.Series(e).unique()
logging.info(a)
logging.info(b)
logging.info(c)
logging.info(d)
logging.info(e)


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


dataset = {}
for i in tqdm(range(len(processed_data))):
    if i <= 2000:
        continue
    game = processed_data[i]
    tensor_board = TensorBoard(Board(), Board(), 1)
    data = []
    for index in range(len(game)):
        notation = game[index]
        move = notation_to_move(tensor_board.turn, tensor_board.boards[-1], notation)

        expect_action = TensorBoard.encode_action_to_tensor(move)
        state = tensor_board.encode_board_to_tensor()

        valid_actions = tensor_board.encode_actions_to_tensor()
        flag = ('black' in args.time and index % 2 == 1) or ('white' in args.time and index % 2 == 0)
        if flag:
            data.append((state.astype(np.int16), valid_actions.astype(np.uint8), expect_action.astype(np.uint8)))
        tensor_board = tensor_board.get_next_state(move)
    dataset[i] = data

    if i % 500 == 0 and i > 0:
        outfile = os.path.join(OUTPUT, f'{i - 500}_{i}.pkl')
        logging.info(f'Save 500 games at: {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(dataset, f)
        dataset = {}

n = len(processed_data) % 500
outfile = os.path.join(OUTPUT, f'{len(processed_data) - n}_{len(processed_data)}.pkl')
logging.info(f'Save {n} games at: {outfile}')
with open(outfile, 'wb') as f:
    pickle.dump(dataset, f)
logging.info('Completed')
