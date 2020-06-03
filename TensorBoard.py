import time
import chess
import copy
import numpy as np
# board = chess.Board()
# board.push_san("e4")

# board.push_san("e5")

# board.push_san("Qh5")

# board.push_san("Nc6")

# board.push_san("Bc4")

# board.push_san("Nf6")

# board.push_uci("g1f3")
# board.push_uci('d7d5')
# board.push_uci('e1g1')
# board.push_uci('d5e4')
# board.push_uci('d2d3')
# board.push_uci('e4e3')
# board.push_uci('d3d4')
# board.push_uci('e3e2')
# board.push_uci('h5f7')

mapping_position = {
    'k': 0, 'q': 1, 'r': 2, 'b': 3, 'n': 4, 'p': 5,
    'K': 6, 'Q': 7, 'R': 8, 'B': 9, 'N': 10, 'P': 11
}

mapping_uci_index = {
    '8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7,
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7
}

mapping_index_uci_row = {
    0: '8', 1: '7', 2: '6', 3: '5', 4: '4', 5: '3', 6: '2', 7: '1'
}

mapping_index_uci_col = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'
}

class TensorChessBoard(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.is_checkmate = self.board.is_checkmate()
        self.is_game_over = self.board.is_game_over()
        self.is_draw = self.is_game_over and (not self.is_checkmate)
        self.turn = self.board.turn

    def encode_board(self):
        board_str = str(self.board)
        tensor = np.zeros((8, 8, 12))
        s = board_str.split('\n')
        s = [x.split(' ') for x in s]
        s = np.array(s)
        for row in range(8):
            for col in range(8):
                if s[row][col] == '.':
                    continue
                c = mapping_position[s[row][col]]
                tensor[row][col][c] = 1
        return tensor

    def encode_moves(self):
        tensor = np.zeros((8, 8, 64))
        moves = [x.uci() for x in self.board.legal_moves]
        for move in moves:
            start = move[0:2]
            end = move[2:4]
            start_row = mapping_uci_index[start[1]]
            start_col = mapping_uci_index[start[0]]

            end_row = mapping_uci_index[end[1]]
            end_col = mapping_uci_index[end[0]]

            c = end_row * 8 + end_col
            tensor[start_row][start_col][c] = 1
        return tensor

    def encode(self):
        s1 = self.encode_board()
        s2 = self.encode_moves()
        return np.concatenate((s1, s2), axis=2)

    def get_next_state(self, move):
        board = copy.deepcopy(self.board)
        board.push_uci(move)
        return TensorChessBoard(board=board)

    @staticmethod
    def decode_moves(tensor: np.array, board: chess.Board):
        """[summary]

        Arguments:
            tensor {np.array} -- 8 x 8 x 64
        """
        s = np.argwhere(tensor == 1)
        moves_uci = []
        legal_moves = [x.uci() for x in board.legal_moves]
        promoted_moves = [x for x in legal_moves if len(x) == 5]
        for start_row, start_col, c in s:
            end_row = c // 8
            end_col = c % 8
            start = mapping_index_uci_col[start_col] + mapping_index_uci_row[start_row]
            end = mapping_index_uci_col[end_col] + mapping_index_uci_row[end_row]
            move = start + end
            for mv in promoted_moves:
                if move in mv:
                    move = move + 'q'
                    break
            moves_uci.append(move)
        return moves_uci

    def display(self):
        s = str(self.board)
        s = s.split('\n')
        for u in s:
            print(u)
        print()

    @staticmethod
    def decode_moves_from_index(index: int, legal_moves: list):
        """[summary]
        index = 512 * row + 64 * col + channel
        index in {0, 1, 2, ... 4095}
        row, col in {0, 1, 2, ... 7}
        channel in {0, 1, 2, ... 63}
        Arguments:
            tensor {np.array} -- 8 x 8 x 64
        """
        s1 = time.time() * 1000
        start_row = index // 512
        start_col = (index % 512) // 64
        channel = (index % 512) % 64
        end_row = channel // 8
        end_col = channel % 8
        s2 = time.time() * 1000
        promoted_moves = [x for x in legal_moves if len(x) == 5]
        start = mapping_index_uci_col[start_col] + mapping_index_uci_row[start_row]
        end = mapping_index_uci_col[end_col] + mapping_index_uci_row[end_row]
        move = start + end
        for mv in promoted_moves:
            if move in mv:
                move = move + 'q'
                break
        return move

    @staticmethod
    def encode_move(move: str):
        tensor = np.zeros((8, 8, 64))
        start = move[0:2]
        end = move[2:4]
        start_row = mapping_uci_index[start[1]]
        start_col = mapping_uci_index[start[0]]

        end_row = mapping_uci_index[end[1]]
        end_col = mapping_uci_index[end[0]]

        c = end_row * 8 + end_col
        tensor[start_row][start_col][c] = 1
        return tensor

# tensor_board = TensorChessBoard()
# board = tensor_board.encode_board()
# moves = tensor_board.encode_moves()
# tensor = tensor_board.encode()

# s = tensor_board.get_next_state('b1c3')
