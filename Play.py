import time

from Minimax import Minimax
from ChessObjects import Board, Spot, Move
from Rules import Rules

# minimax = Minimax()
# board = Board()
# board.display()
# turn = 0
# while True:
#     move = minimax.search_next_move(turn, board, 5)
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

minimax = Minimax()
board = Board()
turn = 0
while True:
    print(f'Turn {turn}')
    move = minimax.search_next_move(turn, board, 4)
    board = Rules.get_next_state(move, board)
    board.display()
    print(f'last move: {board.last_mv}')
    print(f'n_moves: {board.n_mvs}')
    print('===================================')
    turn = abs(1 - turn)
    if Rules.is_checkmate(turn, board):
        print(f'team {abs(1 - turn)} win')
        break
    elif Rules.is_draw(turn, board):
        print(f'draw')
        break
    print(f'Turn {turn}')
    valid_moves = Rules.get_all_valid_moves(turn, board)
    print(f'All valids moves: {len(valid_moves)}')
    for mv in valid_moves:
        print(mv)
    print('*****')
    while True:
        try:
            s = input(f'your move: ')
            s = [int(x) for x in list(s)]
            move = Move(Spot(s[0], s[1]), Spot(s[2], s[3]))
            if not Rules.is_valid_move(turn, move, board):
                continue
            board = Rules.get_next_state(move, board)
            board.display()
            print(f'last move: {board.last_mv}')
            print(f'n_moves: {board.n_mvs}')
            print('===================================')
            break
        except Exception as e:
            pass

    turn = abs(1 - turn)
    if Rules.is_checkmate(turn, board):
        print(f'team {abs(1 - turn)} win')
        break
    elif Rules.is_draw(turn, board):
        print(f'draw')
        break
