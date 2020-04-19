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


class Utils(object):
    def __init__(self):
        super(Utils, self).__init__()

    def get_king_spot(team: int, board: Board):
        for row in range(8):
            for col in range(8):
                if isinstance(board.board[row][col], King):
                    if board.board[row][col].team == team:
                        return Spot(row, col)
        return Spot(-1, -1)
