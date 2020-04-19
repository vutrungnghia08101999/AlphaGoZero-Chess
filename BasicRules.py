import numbers
import copy

from ChessObjects import (
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
    Spot,
    Board
)


def influence_to_reached_spots(team: int, influence_spots: list, board: Board):
    reached_spots = []
    for spot in influence_spots:
        r = spot.row
        c = spot.col
        if(not isinstance(board.board[r][c], numbers.Number)):
            if(board.board[r][c].team == team):
                continue
        reached_spots.append(Spot(spot.row, spot.col))
    return reached_spots


class KingRules(object):
    def __init__(self):
        super(KingRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], King)
        influence_spots = []
        for i in [row - 1, row, row + 1]:
            for j in [col - 1, col, col + 1]:
                if i >= 0 and i <= 7 and j >= 0 and j <= 7 and (i != row or j != col):
                    influence_spots.append(Spot(i, j))
        return influence_spots

    @staticmethod
    def get_reached_spots(spot: Spot, board: Board):
        assert isinstance(board.board[spot.row][spot.col], King)
        team = board.board[spot.row][spot.col].team
        influence_spots = KingRules.get_influence_spots(spot, board)
        reached_spots = influence_to_reached_spots(team, influence_spots, board)
        castling_spots = KingRules._get_castling_spots(spot, board)

        return reached_spots + castling_spots

    @staticmethod
    def _get_castling_spots(spot: Spot, board: Board):
        """
        have not include: ischeck
        """
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], King)
        king = board.board[row][col]
        if king.is_moved:
            return []

        castling_spots = []
        left, right = board.board[row][0], board.board[row][7]
        (p1, p2, p3, p4, p5) = (
            board.board[row][1], board.board[row][2], board.board[row][4], board.board[row][5], board.board[row][6])
        if isinstance(left, Rook):
            if not left.is_moved and isinstance(p1, numbers.Number) and isinstance(p2, numbers.Number):
                castling_spots.append(Spot(row, 1, is_castling=True))

        if isinstance(right, Rook):
            if not right.is_moved:
                if isinstance(p3, numbers.Number) and isinstance(p4, numbers.Number) and isinstance(p5, numbers.Number):
                    castling_spots.append(Spot(row, 5, is_castling=True))
        return castling_spots


class QueenRules(object):
    def __init__(self):
        super(QueenRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Queen)
        influence_spots = []
        for j in range(col + 1, 8):
            influence_spots.append(Spot(row, j))
            if(not isinstance(board.board[row][j], numbers.Number)):
                break
        for j in range(col - 1, -1, -1):
            influence_spots.append(Spot(row, j))
            if(not isinstance(board.board[row][j], numbers.Number)):
                break
        for i in range(row + 1, 8):
            influence_spots.append(Spot(i, col))
            if(not isinstance(board.board[i][col], numbers.Number)):
                break
        for i in range(row - 1, -1, -1):
            influence_spots.append(Spot(i, col))
            if(not isinstance(board.board[i][col], numbers.Number)):
                break

        for i in range(1, 8):
            if row + i > 7 or col + i > 7:
                break
            influence_spots.append(Spot(row + i, col + i))
            if(not isinstance(board.board[row + i][col + i], numbers.Number)):
                break
        for i in range(1, 8):
            if row - i < 0 or col - i < 0:
                break
            influence_spots.append(Spot(row - i, col - i))
            if(not isinstance(board.board[row - i][col - i], numbers.Number)):
                break
        for i in range(1, 8):
            if row - i < 0 or col + i > 7:
                break
            influence_spots.append(Spot(row - i, col + i))
            if(not isinstance(board.board[row - i][col + i], numbers.Number)):
                break
        for i in range(1, 8):
            if row + i > 7 or col - i < 0:
                break
            influence_spots.append(Spot(row + i, col - i))
            if(not isinstance(board.board[row + i][col - i], numbers.Number)):
                break
        return influence_spots

    def get_reached_spots(spot: Spot, board: Board):
        assert isinstance(board.board[spot.row][spot.col], Queen)
        team = board.board[spot.row][spot.col].team
        influence_spots = QueenRules.get_influence_spots(spot, board)
        reached_spots = influence_to_reached_spots(team, influence_spots, board)
        return reached_spots


class RookRules(object):
    def __init__(self):
        super(RookRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Rook)
        influence_spots = []
        for j in range(col + 1, 8):
            influence_spots.append(Spot(row, j))
            if(not isinstance(board.board[row][j], numbers.Number)):
                break
        for j in range(col - 1, -1, -1):
            influence_spots.append(Spot(row, j))
            if(not isinstance(board.board[row][j], numbers.Number)):
                break
        for i in range(row + 1, 8):
            influence_spots.append(Spot(i, col))
            if(not isinstance(board.board[i][col], numbers.Number)):
                break
        for i in range(row - 1, -1, -1):
            influence_spots.append(Spot(i, col))
            if(not isinstance(board.board[i][col], numbers.Number)):
                break
        return influence_spots

    def get_reached_spots(spot: Spot, board: Board):
        assert isinstance(board.board[spot.row][spot.col], Rook)
        team = board.board[spot.row][spot.col].team
        influence_spots = RookRules.get_influence_spots(spot, board)
        reached_spots = influence_to_reached_spots(team, influence_spots, board)
        return reached_spots


class BishopRules(object):
    def __init__(self):
        super(BishopRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Bishop)
        influence_spots = []
        for i in range(1, 8):
            if row + i > 7 or col + i > 7:
                break
            influence_spots.append(Spot(row + i, col + i))
            if(not isinstance(board.board[row + i][col + i], numbers.Number)):
                break
        for i in range(1, 8):
            if row - i < 0 or col - i < 0:
                break
            influence_spots.append(Spot(row - i, col - i))
            if(not isinstance(board.board[row - i][col - i], numbers.Number)):
                break
        for i in range(1, 8):
            if row - i < 0 or col + i > 7:
                break
            influence_spots.append(Spot(row - i, col + i))
            if(not isinstance(board.board[row - i][col + i], numbers.Number)):
                break
        for i in range(1, 8):
            if row + i > 7 or col - i < 0:
                break
            influence_spots.append(Spot(row + i, col - i))
            if(not isinstance(board.board[row + i][col - i], numbers.Number)):
                break
        return influence_spots

    def get_reached_spots(spot: Spot, board: Board):
        assert isinstance(board.board[spot.row][spot.col], Bishop)
        team = board.board[spot.row][spot.col].team
        influence_spots = BishopRules.get_influence_spots(spot, board)
        reached_spots = influence_to_reached_spots(team, influence_spots, board)
        return reached_spots


class KnightRules(object):
    def __init__(self):
        super(KnightRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Knight)
        influence_spots = []
        if row - 2 >= 0 and col + 1 <= 7:
            influence_spots.append(Spot(row - 2, col + 1))
        if row - 1 >= 0 and col + 2 <= 7:
            influence_spots.append(Spot(row - 1, col + 2))

        if row + 1 <= 7 and col + 2 <= 7:
            influence_spots.append(Spot(row + 1, col + 2))
        if row + 2 <= 7 and col + 1 <= 7:
            influence_spots.append(Spot(row + 2, col + 1))

        if row + 2 <= 7 and col - 1 >= 0:
            influence_spots.append(Spot(row + 2, col - 1))
        if row + 1 <= 7 and col - 2 >= 0:
            influence_spots.append(Spot(row + 1, col - 2))

        if row - 1 >= 0 and col - 2 >= 0:
            influence_spots.append(Spot(row - 1, col - 2))
        if row - 2 >= 0 and col - 1 >= 0:
            influence_spots.append(Spot(row - 2, col - 1))

        return influence_spots

    def get_reached_spots(spot: Spot, board: Board):
        assert isinstance(board.board[spot.row][spot.col], Knight)
        team = board.board[spot.row][spot.col].team
        influence_spots = KnightRules.get_influence_spots(spot, board)
        reached_spots = influence_to_reached_spots(team, influence_spots, board)
        return reached_spots


class PawnRules(object):
    def __init__(self):
        super(PawnRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Pawn)
        influence_spots = []
        pawn = board.board[row][col]
        if pawn.team == 0:
            if row + 1 <= 7 and col - 1 >= 0:
                influence_spots.append(Spot(row + 1, col - 1))
            if row + 1 <= 7 and col + 1 <= 7:
                influence_spots.append(Spot(row + 1, col + 1))

        elif pawn.team == 1:
            if row - 1 >= 0 and col - 1 >= 0:
                influence_spots.append(Spot(row - 1, col - 1))
            if row - 1 >= 0 and col + 1 <= 7:
                influence_spots.append(Spot(row - 1, col + 1))

        return influence_spots

    @staticmethod
    def get_reached_spots(spot: Spot, board: Board):
        row = spot.row
        col = spot.col
        assert isinstance(board.board[row][col], Pawn)
        reached_spots = []

        pawn = board.board[row][col]

        if pawn.team == 0:
            if row + 1 <= 7:
                if isinstance(board.board[row + 1][col], numbers.Number):
                    reached_spots.append(Spot(row + 1, col, is_promoted=(row + 1 == 7)))

            if row == 1:
                if isinstance(board.board[row + 1][col], numbers.Number) and isinstance(board.board[row + 2][col],
                                                                                        numbers.Number):
                    reached_spots.append(Spot(row + 2, col, is_pdb_move=True))

        elif pawn.team == 1:
            if row - 1 >= 0:
                if isinstance(board.board[row - 1][col], numbers.Number):
                    reached_spots.append(Spot(row - 1, col, is_promoted=(row - 1 == 0)))

            if row == 6:
                if isinstance(board.board[row - 1][col], numbers.Number) and isinstance(board.board[row - 2][col],
                                                                                        numbers.Number):
                    reached_spots.append(Spot(row - 2, col, is_pdb_move=True))

        influence_spots = PawnRules.get_influence_spots(spot, board)
        for spt in influence_spots:
            r = spt.row
            c = spt.col
            if isinstance(board.board[r][c], numbers.Number):
                continue
            if board.board[r][c].team == pawn.team:
                continue
            if pawn.team == 0:
                reached_spots.append(Spot(r, c, is_promoted=(r == 7)))
            elif pawn.team == 1:
                reached_spots.append(Spot(r, c, is_promoted=(c == 0)))

        last_move = copy.deepcopy(board.last_mv)
        start = last_move.start
        end = last_move.end
        if last_move.is_pdb_move:
            if abs(end.col - col) == 1:
                if pawn.team == 0 and row == 4:
                    reached_spots.append(Spot(row + 1, end.col, is_en_passant=True))
                elif pawn.team == 1 and row == 3:
                    reached_spots.append(Spot(row - 1, end.col, is_en_passant=True))

        return reached_spots


def choose_rules(piece):
    c = [King, Queen, Rook, Bishop, Knight, Pawn]
    rules = [KingRules, QueenRules, RookRules, BishopRules, KnightRules, PawnRules]
    for i in range(len(c)):
        if isinstance(piece, c[i]):
            return rules[i]
    raise RuntimeError(f'{piece} in not a piece')


class BasicRules(object):
    def __init__(self):
        super(BasicRules, self).__init__()

    @staticmethod
    def get_influence_spots(spot: Spot, board: Board):
        rule = choose_rules(board.board[spot.row][spot.col])
        return rule.get_influence_spots(spot, board)

    @staticmethod
    def get_reached_spots(spot: Spot, board: Board):
        rule = choose_rules(board.board[spot.row][spot.col])
        return rule.get_reached_spots(spot, board)
