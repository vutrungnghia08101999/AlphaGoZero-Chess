import time
import numpy as np

import torch
import torch.nn.functional as F

from chess_rules.TensorBoard import TensorBoard
from sl_traning.model import ChessModel
from minimax.Minimax import Minimax
from chess_rules.ChessObjects import Board, Spot, Move
from chess_rules.Rules import Rules


# ************************ MINIMAX VS MINIMAX ***********************
# DEPTH = 5
# minimax = Minimax()
# board = Board()
# board.display()
# turn = 0
# while True:
#     move = minimax.search_next_move(turn, board, DEPTH)
#     board = Rules.get_next_state(move, board)
#     board.display()
#     print(f'last move: {board.last_mv}')
#     print(f'n_moves: {board.n_mvs}')
#     time.sleep(2)
#     turn = abs(1 - turn)
#     if Rules.is_checkmate(turn, board):
#         print(f'team {abs(1 - turn)} win')
#         break
#     elif Rules.is_draw(turn, board):
#         print(f'draw')
#         break

# ************************ MINIMAX VS PLAYER ***********************
# DEPTH = 4
# minimax = Minimax()
# board = Board()
# turn = 0
# while True:
#     print(f'Turn {turn}')
#     move = minimax.search_next_move(turn, board, DEPTH)
#     board = Rules.get_next_state(move, board)
#     board.display()
#     print(f'last move: {board.last_mv}')
#     print(f'n_moves: {board.n_mvs}')
#     print('===================================')
#     turn = abs(1 - turn)
#     if Rules.is_checkmate(turn, board):
#         print(f'team {abs(1 - turn)} win')
#         break
#     elif Rules.is_draw(turn, board):
#         print(f'draw')
#         break
#     print(f'Turn {turn}')
#     valid_moves = Rules.get_all_valid_moves(turn, board)
#     print(f'All valids moves: {len(valid_moves)}')
#     for mv in valid_moves:
#         print(mv)
#     print('*****')
#     while True:
#         try:
#             s = input(f'your move: ')
#             s = [int(x) for x in list(s)]
#             move = Move(Spot(s[0], s[1]), Spot(s[2], s[3]))
#             if not Rules.is_valid_move(turn, move, board):
#                 continue
#             board = Rules.get_next_state(move, board)
#             board.display()
#             print(f'last move: {board.last_mv}')
#             print(f'n_moves: {board.n_mvs}')
#             print('===================================')
#             break
#         except Exception as e:
#             pass

#     turn = abs(1 - turn)
#     if Rules.is_checkmate(turn, board):
#         print(f'team {abs(1 - turn)} win')
#         break
#     elif Rules.is_draw(turn, board):
#         print(f'draw')
#         break

# ************************ SUPERVISED LEARNING VS PLAYER || MINIMAX ***********************

MODEL_PATH = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/SL/best_model_1.pth'
DEPTH = 4
model = ChessModel()
checkpoint = torch.load(
    MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

history = []
board = Board()
history.append(board)
history[-1].display()
turn = 1
while True:
    # ************** PLAYER **************
    while True:
        try:
            s = input(f'your move: ')
            s = [int(x) for x in list(s)]
            move = Move(Spot(s[0], s[1]), Spot(s[2], s[3]))
            if not Rules.is_valid_move(turn, move, board):
                continue
            break
        except Exception as e:
            pass
    # ************** MINIMAX **************
    # minimax = Minimax()
    # move = minimax.search_next_move(turn, board, DEPTH)
    # **************************************
    # if Rules.is_checkmate(turn, board):
    #    print(f'Team {abs(1-turn)} win')
    #    break
    # elif Rules.is_draw(turn, board):
    #    print('draw')
    #    break
    # *************************************
    board = Rules.get_next_state(move, board)
    turn = abs(1 - turn)
    history.append(board)
    history[-1].display()
    print(f'last move: {history[-1].last_mv}')
    print(f'n moves: {history[-1].n_mvs}')
    print('===================================')

    tensor_board = TensorBoard(history[-2], history[-1], turn)
    model.eval()
    encoded_board = tensor_board.encode_board_to_tensor()

    encoded_board = torch.tensor(encoded_board, dtype=torch.float)
    encoded_board = encoded_board.view(-1, 8, 8, 26)
    with torch.no_grad():
        pred = model(encoded_board)
    pred = pred.squeeze()
    pred = F.softmax(pred, dim=0)
    valid_moves = tensor_board.encode_actions_to_tensor()
    valid_moves = valid_moves.flatten()

    pred = pred * (valid_moves == 1)
    idx = torch.argmax(pred)
    encoded_move = (np.array(pred) * 0).astype(np.int16)
    encoded_move[idx] = 1

    encoded_move = encoded_move.reshape(8, 8, 78)
    move = TensorBoard.decode_tensor_to_moves(encoded_move)
    assert len(move) == 1
    board = Rules.get_next_state(move[0], board)

    turn = abs(1 - turn)
    history.append(board)
    history[-1].display()
    print(f'last move: {history[-1].last_mv}')
    print(f'n moves: {history[-1].n_mvs}')
    print('===================================')

    if Rules.is_checkmate(turn, board):
        print(f'Team {abs(1-turn)} win')
        break
    elif Rules.is_draw(turn, board):
        print('draw')
        break

# ************************ SUPERVISED LEARNING VS MINIMAX ***********************
