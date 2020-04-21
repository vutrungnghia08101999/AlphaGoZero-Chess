import logging
import copy
import numbers

import numpy as np

logging.basicConfig(level=logging.INFO)

class King(object):
    def __init__(self, team: int, is_moved=False):
        super(King, self).__init__()
        self.team = team
        self.is_moved = is_moved

    def __str__(self):
        if self.team == 0:
            return 'KING'
        return 'king'


class Queen(object):
    def __init__(self, team: int):
        super(Queen, self).__init__()
        self.team = team

    def __str__(self):
        if self.team == 0:
            return 'QUEEN'
        return 'queen'


class Rook(object):
    def __init__(self, team: int, is_moved=False):
        super(Rook, self).__init__()
        self.team = team
        self.is_moved = is_moved

    def __str__(self):
        if self.team == 0:
            return 'ROOK'
        return 'rook'


class Bishop(object):
    def __init__(self, team: int):
        super(Bishop, self).__init__()
        self.team = team

    def __str__(self):
        if self.team == 0:
            return 'BISHOP'
        return 'bishop'


class Knight(object):
    def __init__(self, team: int):
        super(Knight, self).__init__()
        self.team = team

    def __str__(self):
        if self.team == 0:
            return 'KNIGHT'
        return 'knight'


class Pawn(object):
    def __init__(self, team: int):
        super(Pawn, self).__init__()
        self.team = team

    def __str__(self):
        if self.team == 0:
            return 'PAWN'
        return 'pawn'


class Spot(object):
    def __init__(self, row: int, col: int, is_castling=False, is_promoted=False, is_en_passant=False, is_pdb_move=False):
        super(Spot, self).__init__()
        self.row = row
        self.col = col
        self.is_castling = is_castling
        self.is_promoted = is_promoted
        self.is_en_passant = is_en_passant
        self.is_pdb_move = is_pdb_move

    def __eq__(self, spot):
        flag_1 = (self.row == spot.row) and (self.col == spot.col)
        flag_2 = (self.is_castling == spot.is_castling) and (self.is_promoted == spot.is_promoted) and \
            (self.is_en_passant == spot.is_en_passant) and (self.is_pdb_move == spot.is_pdb_move)
        return flag_1 and flag_2

    def __str__(self):
        s = f'({self.row}, {self.col})'
        if self.is_castling:
            return s + ' - castling'
        if self.is_en_passant:
            return s + ' - en_passant'
        if self.is_promoted:
            return s + ' - promoted'
        if self.is_pdb_move:
            return s + ' - pdb_move'
        return s


class Move(object):
    def __init__(self, start: Spot, end: Spot, is_castling=False, is_promoted=False, is_en_passant=False, is_pdb_move=False):
        self.start = Spot(start.row, start.col)
        self.end = Spot(end.row, end.col)
        self.is_castling = is_castling
        self.is_promoted = is_promoted
        self.is_en_passant = is_en_passant
        self.is_pdb_move = is_pdb_move

    def __eq__(self, move):
        flag_1 = (self.start == move.start) and (self.end == move.end)
        flag_2 = (self.is_castling == move.is_castling) and (self.is_promoted == move.is_promoted) and \
            (self.is_en_passant == move.is_en_passant) and (self.is_pdb_move == move.is_pdb_move)
        return flag_1 and flag_2

    def __str__(self):
        s = f'({self.start.row}, {self.start.col}) => ({self.end.row}, {self.end.col})'
        if self.is_castling:
            return s + ' - castling'
        if self.is_en_passant:
            return s + ' - en_passant'
        if self.is_promoted:
            return s + ' - promoted'
        if self.is_pdb_move:
            return s + ' - pdb_move'
        return s

class Board(object):
    def __init__(self, last_mv=None, n_mvs=0, board=None):
        if last_mv is not None and board is not None:
            self.last_mv = copy.deepcopy(last_mv)
            self.n_mvs = n_mvs
            self.board = copy.deepcopy(board)
        else:
            self.last_mv = Move(Spot(-1, -1), Spot(-1, -1))
            self.n_mvs = 0
            self.board = self.init_board()

    def init_board(self) -> np.array:
        board = np.zeros((8, 8))
        board = np.array(board).astype(object)
        board[0][0] = Rook(0)
        board[0][1] = Knight(0)
        board[0][2] = Bishop(0)
        board[0][3] = Queen(0)
        board[0][4] = King(0)
        board[0][5] = Bishop(0)
        board[0][6] = Knight(0)
        board[0][7] = Rook(0)
        for col in range(8):
            board[1][col] = Pawn(0)

        board[7][0] = Rook(1)
        board[7][1] = Knight(1)
        board[7][2] = Bishop(1)
        board[7][3] = Queen(1)
        board[7][4] = King(1)
        board[7][5] = Bishop(1)
        board[7][6] = Knight(1)
        board[7][7] = Rook(1)
        for col in range(8):
            board[6][col] = Pawn(1)
        return board

    def display(self):
        print("\n\t\t\t  ____0________1________2________3________4________5________6________7____")
        for row in range(8):
            print("\t\t\t  |        |        |        |        |        |        |        |        |")
            s = f"\t\t\t{row} |"
            for col in range(8):
                if isinstance(self.board[row][col], numbers.Number):
                    s = s + "        |"
                else:
                    s = s + " %-7s|" % (str(self.board[row][col]))
            print(s)
            print("\t\t\t  |________|________|________|________|________|________|________|________|")
        print('\n')

# s = Board()
# s.board[3][0] = Rook(1)
# s.display()
