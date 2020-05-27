import chess
import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("stockfish")

board = chess.Board()
board.push_uci('g1f3')
board