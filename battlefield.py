import argparse
from tqdm import tqdm

import torch

from model import ChessModel
from TensorBoard import TensorChessBoard
from MCTS import MCTSNode

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL/1.pth')
parser.add_argument('--n_simulation', type=int, default=1000)
args = parser.parse_args()

model = ChessModel()
checkpoint = torch.load(args.weights)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

tensor_board = TensorChessBoard()
tensor_board.display()
while True:
    print(f'TURN: {tensor_board.board.turn}')
    legal_moves = [x.uci() for x in tensor_board.board.legal_moves]
    move = None
    while move not in legal_moves:
        move = input('UCI move: ')

    tensor_board = tensor_board.get_next_state(move)
    if tensor_board.is_game_over:
        print(f'{True} won')
        break
    print(f'TURN: {tensor_board.board.turn}')
    root = MCTSNode(
        model=model,
        index=-1,
        parent=None,
        perspective=tensor_board.turn,
        is_game_over=tensor_board.is_game_over,
        is_checkmate=tensor_board.is_checkmate,
        is_draw=tensor_board.is_draw,
        tensor_board=tensor_board)
    print(f'Run {args.n_simulation} simulations in MCTS')
    for idx in tqdm(range(args.n_simulation)):
        best_child = root.traverse()
        best_child.expand_and_backpropagate()
    _, mv = root.get_pi_policy_and_most_visited_move()
    tensor_board = tensor_board.get_next_state(mv)
    tensor_board.display()
    if tensor_board.is_game_over:
        print(f'{False} won')
        break

print('Finished')
