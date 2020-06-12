import argparse
import chess
from tqdm import tqdm

import torch

from model import ChessModel
from TensorBoard import TensorChessBoard
from MCTS import MCTSNode

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL/10.pth')
parser.add_argument('--n_simulation', type=int, default=1000)
args = parser.parse_args()

model = ChessModel()
checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def search_move(fen_representation: str, n_simulations=1000):
    board = chess.Board(fen_representation)
    tensor_board = TensorChessBoard(board=board)
    root = MCTSNode(
        model=model,
        index=-1,
        parent=None,
        perspective=tensor_board.turn,
        is_game_over=tensor_board.is_game_over,
        is_checkmate=tensor_board.is_checkmate,
        is_draw=tensor_board.is_draw,
        tensor_board=tensor_board)
    print(f'Run {n_simulations} simulations in MCTS')
    for idx in tqdm(range(n_simulations)):
        best_child = root.traverse()
        best_child.expand_and_backpropagate()
    _, mv = root.get_pi_policy_and_most_visited_move()
    tensor_board = tensor_board.get_next_state(mv)
    return tensor_board.board.fen(), mv

# *************** PLAY ******************
board = chess.Board()
print(board)
while True:
    print(f'TURN: {board.turn}')
    legal_moves = [x.uci() for x in board.legal_moves]
    move = None
    while move not in legal_moves:
        move = input('UCI move: ')
    board.push_uci(move)
    print(board)

    if board.is_checkmate():
        print(f'{True} won')
        break
    elif board.is_game_over():
        print('Draw')
        break
    
    print(f'TURN: {board.turn}')
    next_fen, move = search_move(board.fen())
    board = chess.Board(next_fen)
    print(board)
    if board.is_checkmate():
        print(f'{False} won')
        break
    elif board.is_game_over():
        print('Draw')
        break

print('Finished')
