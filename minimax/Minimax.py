import copy
import numbers
from random import shuffle

from chess_rules.Rules import Rules
from chess_rules.ChessObjects import (
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

class Minimax(object):
    def __init__(self):
        super(Minimax, self).__init__()
        self.DFSCalls = 0
        self.next_move = None

    def search_next_move(self, team: int, board: Board, root_tree_depth=6):
        self.DFSCalls = 0
        maximum = self.dfs_alpha_beta(board, team, root_tree_depth, team, root_tree_depth, -1000000000, 1000000000)
        print(f"DFSCalls: {self.DFSCalls}")
        print(f"Minimize-maximize algorithm metrics: {maximum}")
        return self.next_move

    def dfs_alpha_beta(self, board: Board, flag: int, depth: int, team: int, root_tree_depth: int, alpha: float, beta: float):
        self.DFSCalls += 1
        if depth == 1:
            return self.evaluate(team, board)

        moves = Rules.get_all_valid_moves(flag, board)
        shuffle(moves)

        if len(moves) == 0:
            return self.evaluate(team, board)

        next_team = abs(1 - flag)
        if flag == team:
            maximum = -1000000000
            for move in moves:
                B = Rules.get_next_state(move, board)
                metric = self.dfs_alpha_beta(B, next_team, depth - 1, team, root_tree_depth, alpha, beta)
                if metric > maximum:
                    maximum = metric
                    if depth == root_tree_depth:
                        self.next_move = copy.deepcopy(move)
                alpha = max(metric, alpha)
                if beta <= alpha:
                    break
            return maximum
        else:
            minimum = 1000000000
            for move in moves:
                B = Rules.get_next_state(move, board)
                metric = self.dfs_alpha_beta(B, next_team, depth - 1, team, root_tree_depth, alpha, beta)
                minimum = min(minimum, metric)
                beta = min(beta, metric)
                if beta <= alpha:
                    break
            return minimum

    def evaluate(self, team: int, board: Board):
        current_team = 0
        opponent = 0
        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if isinstance(piece, numbers.Number):
                    continue
                score = 0
                if isinstance(piece, Pawn):
                    if piece.team == 0:
                        if row >= 5:
                            score = 5
                        else:
                            score = 1
                    elif piece.team == 1:
                        if row <= 2:
                            score = 5
                        else:
                            score = 1
                else:
                    score = self.get_score(piece)
                if piece.team == team:
                    current_team += score
                else:
                    opponent += score
        return (current_team - opponent) / 43

    def get_score(self, piece):
        if isinstance(piece, King):
            return 4
        if isinstance(piece, Queen):
            return 9
        if isinstance(piece, Rook):
            return 5
        if isinstance(piece, Bishop):
            return 3
        if isinstance(piece, Knight):
            return 3
        raise RuntimeError(f'{piece} is null or pawn')


# minimax = Minimax()
# board = Board()
# board.display()
# mv = minimax.search_next_move(0, board, 6)
# minimax.evaluate(0, board)
