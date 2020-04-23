import copy
import numbers

import numpy as np

from chess_rules.Rules import Rules
from chess_rules.ChessObjects import (
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
    Board,
    Move,
    Spot
)

CASTLING = 'castling'
EN_PASSANT = 'en_passant'
QUEEN = 'queen'
PROMOTED = 'promoted'
KNIGHT = 'knight'
PAWN_DOUBLE_MOVE = 'pawn_double_move'

actions_ed = {
    'encoder': {
        CASTLING: {0: {-2: 10, 2: 11}},
        EN_PASSANT: {-1: {-1: 6, 1: 7}, 1: {-1: 8, 1: 9}},
        KNIGHT: {
            -2: {-1: 12, 1: 13},
            -1: {-2: 14, 2: 15},
            1: {-2: 16, 2: 17},
            2: {-1: 18, 1: 19}},
        PROMOTED: {-1: {-1: 0, 0: 1, 1: 2}, 1: {-1: 3, 0: 4, 1: 5}},
        QUEEN: {
            -7: {-7: 20, 0: 21, 7: 22},
            -6: {-6: 23, 0: 24, 6: 25},
            -5: {-5: 26, 0: 27, 5: 28},
            -4: {-4: 29, 0: 30, 4: 31},
            -3: {-3: 32, 0: 33, 3: 34},
            -2: {-2: 35, 0: 36, 2: 37},
            -1: {-1: 38, 0: 39, 1: 40},
            0: {-7: 41,
                -6: 42,
                -5: 43,
                -4: 44,
                -3: 45,
                -2: 46,
                -1: 47,
                1: 48,
                2: 49,
                3: 50,
                4: 51,
                5: 52,
                6: 53,
                7: 54},
            1: {-1: 55, 0: 56, 1: 57},
            2: {-2: 58, 0: 59, 2: 60},
            3: {-3: 61, 0: 62, 3: 63},
            4: {-4: 64, 0: 65, 4: 66},
            5: {-5: 67, 0: 68, 5: 69},
            6: {-6: 70, 0: 71, 6: 72},
            7: {-7: 73, 0: 74, 7: 75}},
        PAWN_DOUBLE_MOVE: {-2: {0: 76}, 2: {0: 77}}},
    'decoder': {
        10: (0, -2, CASTLING),
        11: (0, 2, CASTLING),
        6: (-1, -1, EN_PASSANT),
        7: (-1, 1, EN_PASSANT),
        8: (1, -1, EN_PASSANT),
        9: (1, 1, EN_PASSANT),
        12: (-2, -1, KNIGHT),
        13: (-2, 1, KNIGHT),
        14: (-1, -2, KNIGHT),
        15: (-1, 2, KNIGHT),
        16: (1, -2, KNIGHT),
        17: (1, 2, KNIGHT),
        18: (2, -1, KNIGHT),
        19: (2, 1, KNIGHT),
        0: (-1, -1, PROMOTED),
        1: (-1, 0, PROMOTED),
        2: (-1, 1, PROMOTED),
        3: (1, -1, PROMOTED),
        4: (1, 0, PROMOTED),
        5: (1, 1, PROMOTED),
        20: (-7, -7, QUEEN),
        21: (-7, 0, QUEEN),
        22: (-7, 7, QUEEN),
        23: (-6, -6, QUEEN),
        24: (-6, 0, QUEEN),
        25: (-6, 6, QUEEN),
        26: (-5, -5, QUEEN),
        27: (-5, 0, QUEEN),
        28: (-5, 5, QUEEN),
        29: (-4, -4, QUEEN),
        30: (-4, 0, QUEEN),
        31: (-4, 4, QUEEN),
        32: (-3, -3, QUEEN),
        33: (-3, 0, QUEEN),
        34: (-3, 3, QUEEN),
        35: (-2, -2, QUEEN),
        36: (-2, 0, QUEEN),
        37: (-2, 2, QUEEN),
        38: (-1, -1, QUEEN),
        39: (-1, 0, QUEEN),
        40: (-1, 1, QUEEN),
        41: (0, -7, QUEEN),
        42: (0, -6, QUEEN),
        43: (0, -5, QUEEN),
        44: (0, -4, QUEEN),
        45: (0, -3, QUEEN),
        46: (0, -2, QUEEN),
        47: (0, -1, QUEEN),
        48: (0, 1, QUEEN),
        49: (0, 2, QUEEN),
        50: (0, 3, QUEEN),
        51: (0, 4, QUEEN),
        52: (0, 5, QUEEN),
        53: (0, 6, QUEEN),
        54: (0, 7, QUEEN),
        55: (1, -1, QUEEN),
        56: (1, 0, QUEEN),
        57: (1, 1, QUEEN),
        58: (2, -2, QUEEN),
        59: (2, 0, QUEEN),
        60: (2, 2, QUEEN),
        61: (3, -3, QUEEN),
        62: (3, 0, QUEEN),
        63: (3, 3, QUEEN),
        64: (4, -4, QUEEN),
        65: (4, 0, QUEEN),
        66: (4, 4, QUEEN),
        67: (5, -5, QUEEN),
        68: (5, 0, QUEEN),
        69: (5, 5, QUEEN),
        70: (6, -6, QUEEN),
        71: (6, 0, QUEEN),
        72: (6, 6, QUEEN),
        73: (7, -7, QUEEN),
        74: (7, 0, QUEEN),
        75: (7, 7, QUEEN),
        76: (-2, 0, PAWN_DOUBLE_MOVE),
        77: (2, 0, PAWN_DOUBLE_MOVE)}}


class TensorBoard(object):
    def __init__(self, last_board=None, board=None, turn=None):
        super(TensorBoard, self).__init__()
        if last_board is not None and board is not None and turn is not None:
            self.boards = [copy.deepcopy(last_board), copy.deepcopy(board)]
            self.turn = turn
        else:
            self.boards = [Board(), Board()]
            self.turn = 1

    def encode_board_to_tensor(self) -> np.array:
        s = np.zeros((8, 8, 26))
        s1 = self._board_to_tensor(self.boards[-1])
        s2 = self._board_to_tensor(self.boards[-2])
        turn = np.ones((8, 8)) * self.turn
        n_mvs = np.ones((8, 8)) * self.boards[1].n_mvs

        s[:, :, 0:12] = s1
        s[:, :, 12: 24] = s2
        s[:, :, 24] = turn
        s[:, :, 25] = n_mvs
        return s

    def _board_to_tensor(self, board: Board) -> np.array:
        def _get_layer(piece):
            assert not isinstance(piece, numbers.Number)
            team = piece.team
            if isinstance(piece, King):
                return abs(1 - team) * 0 + abs(0 - team) * 6
            if isinstance(piece, Queen):
                return abs(1 - team) * 1 + abs(0 - team) * 7
            if isinstance(piece, Rook):
                return abs(1 - team) * 2 + abs(0 - team) * 8
            if isinstance(piece, Bishop):
                return abs(1 - team) * 3 + abs(0 - team) * 9
            if isinstance(piece, Knight):
                return abs(1 - team) * 4 + abs(0 - team) * 10
            return abs(1 - team) * 5 + abs(0 - team) * 11
        s = np.zeros((8, 8, 12))
        for row in range(8):
            for col in range(8):
                if isinstance(board.board[row][col], numbers.Number):
                    continue
                channel = _get_layer(board.board[row][col])
                s[row][col][channel] = 1
        return s

    def encode_actions_to_tensor(self) -> np.array:
        valid_moves = Rules.get_all_valid_moves(self.turn, self.boards[-1])
        s = np.zeros((8, 8, 78))
        for move in valid_moves:
            row_diff = move.end.row - move.start.row
            col_diff = move.end.col - move.start.col
            if move.is_castling:
                mv_type = CASTLING
            elif move.is_en_passant:
                mv_type = EN_PASSANT
            elif move.is_promoted:
                mv_type = PROMOTED
            elif move.is_pdb_move:
                mv_type = PAWN_DOUBLE_MOVE
            elif row_diff * row_diff + col_diff * col_diff == 5:
                mv_type = KNIGHT
            else:
                mv_type = QUEEN
            index = actions_ed['encoder'][mv_type][row_diff][col_diff]
            s[move.start.row][move.start.col][index] = 1
        return s

    @staticmethod
    def encode_action_to_tensor(move: Move) -> np.array:
        s = np.zeros((8, 8, 78))
        row_diff = move.end.row - move.start.row
        col_diff = move.end.col - move.start.col
        if move.is_castling:
            mv_type = CASTLING
        elif move.is_en_passant:
            mv_type = EN_PASSANT
        elif move.is_promoted:
            mv_type = PROMOTED
        elif move.is_pdb_move:
            mv_type = PAWN_DOUBLE_MOVE
        elif row_diff * row_diff + col_diff * col_diff == 5:
            mv_type = KNIGHT
        else:
            mv_type = QUEEN
        index = actions_ed['encoder'][mv_type][row_diff][col_diff]
        s[move.start.row][move.start.col][index] = 1
        return s

    @staticmethod
    def decode_tensor_to_moves(actions: np.array) -> list:
        assert actions.shape[0] == 8 and actions.shape[1] == 8
        assert actions.shape[2] == 78
        moves = []
        s = np.argwhere(actions == 1)
        for i in range(s.shape[0]):
            s_row, s_col, index = s[i]
            row_diff, col_diff, mv_type = actions_ed['decoder'][index]
            e_row = s_row + row_diff
            e_col = s_col + col_diff
            start, end = Spot(s_row, s_col), Spot(e_row, e_col)
            move = Move(start, end,
                        is_castling=(mv_type == CASTLING),
                        is_en_passant=(mv_type == EN_PASSANT),
                        is_promoted=(mv_type == PROMOTED),
                        is_pdb_move=(mv_type == PAWN_DOUBLE_MOVE))
            moves.append(move)
        return moves

    @staticmethod
    def decode_tensor_to_board(tensor_board: np.array) -> tuple:
        mapping = [
            (0, King, 0), (1, Queen, 0), (2, Rook, 0),
            (3, Bishop, 0), (4, Knight, 0), (5, Pawn, 0),
            (6, King, 1), (7, Queen, 1), (8, Rook, 1),
            (9, Bishop, 1), (10, Knight, 1), (11, Pawn, 1),
        ]

        board = Board()
        for row in range(8):
            for col in range(8):
                board.board[row][col] = 0

        for (index, c, team) in mapping:
            s = np.argwhere(tensor_board[:, :, index] == 1)
            for s_row, s_col in s:
                board.board[s_row][s_col] = c(team)

        team = tensor_board[:, :, 24][0][0]
        board.n_mvs = tensor_board[:, :, 25][0][0]
        return {'board': board, 'turn': team}

    def get_next_state(self, move):
        assert self.boards[-1].board[move.start.row][move.start.col].team == self.turn
        last_board = copy.deepcopy(self.boards[-1])
        board = Rules.get_next_state(move, self.boards[-1])
        turn = abs(1 - self.turn)
        return TensorBoard(last_board, board, turn)

    def get_valid_moves(self):
        return Rules.get_all_valid_moves(self.turn, self.boards[-1])


# ************* TEST ****************
# from tqdm import tqdm
# N_TEST = 100
# for n in tqdm(range(N_TEST)):
#     board = Board()
#     (a, b, c, d) = (
#         np.random.randint(1, 7), np.random.randint(1, 7),
#         np.random.randint(1, 7), np.random.randint(1, 7))
#     flag_1 = np.random.randint(0, 2)
#     flag_2 = np.random.randint(0, 2)
#     board.board[a][b] = Queen(flag_1)
#     board.board[c][d] = Queen(flag_2)
#     tensor_board = TensorBoard(Board(), board, np.random.randint(0, 2))
#     s = tensor_board.get_valid_moves()
#     a = tensor_board.encode_actions_to_tensor()
#     g = TensorBoard.decode_tensor_to_moves(a)
#     assert len(s) == len(g)
#     for u in s:
#         assert u in g
# ***********************************

# export actions encoder and decoder
# import yaml
# with open('actions_ed.yml', 'w') as outfile:
#     yaml.dump(actions_ed, outfile)
