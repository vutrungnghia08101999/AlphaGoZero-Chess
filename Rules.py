import copy
import numbers

from BasicRules import BasicRules
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


def get_king_spot(team: int, board: Board):
    for row in range(8):
        for col in range(8):
            if isinstance(board.board[row][col], King):
                if board.board[row][col].team == team:
                    return Spot(row, col)
    return Spot(-1, -1)


class Rules(object):
    def __init__(self):
        super(Rules, self).__init__()

    def _is_checked(team: int, board: Board) -> bool:
        king_position = get_king_spot(team, board)
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

    def get_next_state(move: Move, board: Board) -> Board:
        """
        make sure move is valid previously
        """
        start = move.start
        end = move.end
        team = board.board[start.row][start.col].team
        board = copy.deepcopy(board)

        if move.is_castling:
            assert start.col == 4
            assert end.col == 2 or end.col == 6
            board.board[end.row][end.col] = board.board[start.row][start.col]
            board.board[start.row][start.col] = 0
            board.board[end.row][end.col].is_moved = True
            rook_des_col = int((start.col + end.col) / 2)
            if end.col < 4:
                rook_start_col = 0
            else:
                rook_start_col = 7

            board.board[end.row][rook_des_col] = board.board[start.row][rook_start_col]
            board.board[start.row][rook_start_col] = 0
            board.board[end.row][rook_des_col].is_moved = True

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

        board.last_mv = copy.deepcopy(move)
        board.n_mvs += 1
        return board

    def _get_next_state_unverify(move: Move, board: Board) -> Board:
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

    def _get_valid_moves(spot: Spot, board: Board) -> list:
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
                next_state = Rules._get_next_state_unverify(move, board)
                if not Rules._is_checked(team, next_state):
                    valid_moves.append(move)
            elif target.is_castling:
                if Rules._is_checked(team, board):
                    continue
                middle_mv = Move(Spot(row, col), Spot(row, int((spot.col + target.col) / 2)))
                next_state = Rules._get_next_state_unverify(middle_mv, board)
                if Rules._is_checked(team, next_state):
                    continue
                move = Move(Spot(row, col), Spot(target.row, target.col), is_castling=True)
                next_state = Rules._get_next_state_unverify(move, board)
                if not Rules._is_checked(team, next_state):
                    valid_moves.append(move)
            else:
                move = Move(Spot(row, col),
                            Spot(target.row, target.col),
                            is_pdb_move=target.is_pdb_move,
                            is_promoted=target.is_promoted)
                next_state = Rules._get_next_state_unverify(move, board)
                if Rules._is_checked(team, next_state):
                    continue
                valid_moves.append(move)

        return valid_moves

    def get_all_valid_moves(team: int, board: Board) -> list:
        valid_moves = []
        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if isinstance(piece, numbers.Number):
                    continue
                if piece.team != team:
                    continue
                valid_moves = valid_moves + Rules._get_valid_moves(Spot(row, col), board)
        return valid_moves

    def is_valid_move(team: int, move: Move, board: Board) -> bool:
        """
        This function modify {move} - a player move
        """
        valid_moves = Rules.get_all_valid_moves(team, board)
        for mv in valid_moves:
            if mv.start == move.start and mv.end == move.end:
                move.is_castling = mv.is_castling
                move.is_en_passant = mv.is_en_passant
                move.is_pdb_move = mv.is_pdb_move
                move.is_promoted = mv.is_promoted
                return True
        return False

    def is_checkmate(team: int, board: Board) -> bool:
        valid_moves = Rules.get_all_valid_moves(team, board)
        if len(valid_moves) == 0 and Rules._is_checked(team, board):
            return True
        return False

    def is_draw(team: int, board: Board) -> bool:
        valid_moves = Rules.get_all_valid_moves(team, board)
        if len(valid_moves) == 0 and not Rules._is_checked(team, board):
            return True
        return False

# board = Board()
# board.board[0][1] = 0
# board.board[0][2] = 0
# board.board[3][1] = board.board[6][1]
# board.board[6][1] = 0
# board.board[3][2] = board.board[1][2]
# board.board[1][2] = 0
# board.last_mv = Move(Spot(1, 2), Spot(3, 2), is_pdb_move=True)
# board.display()

# t = Rules.get_next_state(Move(Spot(0, 3), Spot(0, 1), is_castling=True), board)
# board.board[2][2] = Rook(1)
# r = Rules.is_valid_move(0, Move(Spot(0, 3), Spot(0, 1), is_castling=True), board)
# # s = get_valid_moves(Spot(0, 3), board)
# # spot = Spot(3, 1)
# # for i in s:
# #     print(i)

# # for u in reached_spots:
# #     print(u)

# s = get_all_valid_moves(0, board)
# for u in s:
#     print(u)
# board.board[7][1] = 0
# board.board[7][2] = 0
# board.display()
# s = Rules._get_valid_moves(Spot(7, 3), board)
# for u in s:
#     print(u)

# t = BasicRules.get_reached_spots(Spot(7, 3), board)
# for u in t:
#     print(u)
# from BasicRules import KingRules
# r = KingRules.get_reached_spots(Spot(7, 3), board)
# for u in r:
#     print(u)
