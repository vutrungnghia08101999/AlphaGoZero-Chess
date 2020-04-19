import copy
import numbers

from BasicRules import BasicRules
from Utils import Utils
from ChessObjects import (
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
    Move,
    Spot,
    Board
)


def is_checked(team: int, board: Board):
    king_position = Utils.get_king_spot(team, board)
    assert king_position.row != -1 and king_position.col != -1

    for row in range(8):
        for col in range(8):
            if isinstance(board.board[row][col], numbers.Number):
                continue
            if board.board[row][col].team == team:
                continue

            influence_spots = BasicRules.get_influence_spots(Spot(row, col), board)
            for spot in influence_spots:
                if spot.row == king_position.row and spot.col == king_position.col:
                    return True

    return False


def get_next_state(move: Move, board: Board):
    """
    make sure move is valid previously
    """
    start = move.start
    end = move.end
    team = board.board[start.row][start.col].team
    board = copy.deepcopy(board)

    if move.is_castling:
        board.board[end.row][end.col] = board.board[start.row][start.col]
        board.board[start.row][start.col] = 0
        board.board[end.row][end.col].is_moved = True
        if end.col == 1:
            board.board[end.row][2] = board.board[start.row][0]
            board.board[start.row][0] = 0
            board.board[end.row][2].is_moved = True
        elif end.col == 5:
            board.board[end.row][4] = board.board[start.row][7]
            board.board[start.row][7] = 0
            board.board[end.row][4].is_moved = True

    elif move.is_promoted:
        board.board[end.row][end.col] = Queen(team)
        board.board[start.row][start.col] = 0

    elif move.is_en_passant:
        board.board[end.row][end.col] = board.board[start.row][start.col]
        board.board[start.row][start.col] = 0
        board.board[start.row][end.col] = 0

    else:
        board.board[end.row][end.col] = board.board[start.row][start.col]
        board.board[start.row][start.col] = 0

    return Board(last_mv=copy.deepcopy(move), n_mvs=board.n_mvs + 1, board=board)


def get_next_state_unverify(move: Move, board: Board):
    start = move.start
    end = move.end
    team = board.board[start.row][start.col].team
    board = copy.deepcopy(board)
    if move.is_en_passant:
        board.board[end.row][end.col] = board.board[start.row][start.col]
        board.board[start.row][start.col] = 0
        board.board[start.row][end.col] = 0
        return board
    board.board[end.row][end.col] = board.board[start.row][start.col]
    board.board[start.row][start.col] = 0
    return board


def get_valid_moves(spot: Spot, board: Board):
    row = spot.row
    col = spot.col
    assert not isinstance(board.board[row][col], numbers.Number)
    team = board.board[row][col].team
    piece = board.board[row][col]

    reached_spots = BasicRules.get_reached_spots(spot, board)
    valid_moves = []
    for target in reached_spots:
        if target.is_en_passant:
            move = Move(Spot(row, col), Spot(target.row, target.col), is_en_passant=True)
            next_state = get_next_state_unverify(move, board)
            if not is_checked(team, next_state):
                valid_moves.append(move)
        elif target.is_castling:
            if is_checked(team, board):
                continue
            middle_mv = Move(Spot(row, col), Spot(row, int((spot.col + target.col) / 2)))
            next_state = get_next_state_unverify(middle_mv, board)
            if is_checked(team, next_state):
                continue
            move = Move(Spot(row, col), Spot(target.row, target.col), is_castling=True)
            next_state = get_next_state_unverify(move, board)
            if not is_checked(team, next_state):
                valid_moves.append(move)
        else:
            move = Move(Spot(row, col),
                        Spot(target.row, target.col),
                        is_pdb_move=target.is_pdb_move,
                        is_promoted=target.is_promoted)
            next_state = get_next_state_unverify(move, board)
            if is_checked(team, next_state):
                continue
            valid_moves.append(move)

    return valid_moves


def get_all_valid_moves(team: int, board: Board):
    valid_moves = []
    for row in range(8):
        for col in range(8):
            piece = board.board[row][col]
            if isinstance(piece, numbers.Number):
                continue
            if piece.team != team:
                continue
            valid_moves = valid_moves + get_valid_moves(Spot(row, col), board)
    return valid_moves

board = Board()
board.board[0][1] = 0
board.board[0][2] = 0
board.board[3][1] = board.board[6][1]
board.board[6][1] = 0
board.board[3][2] = board.board[1][2]
board.board[1][2] = 0
board.last_mv = Move(Spot(1, 2), Spot(3, 2), is_pdb_move=True)
# board.display()

# s = get_valid_moves(Spot(0, 3), board)
# spot = Spot(3, 1)
# for i in s:
#     print(i)

# for u in reached_spots:
#     print(u)

# s = get_all_valid_moves(0, board)
# for u in s:
#     print(u)
