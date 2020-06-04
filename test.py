import chess

board = chess.Board()
board.push_uci('b1c3')

s = board.fen()

t = chess.Board(s)
