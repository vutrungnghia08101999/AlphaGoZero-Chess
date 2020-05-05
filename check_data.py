import os
import numpy as np
import pickle

from chess_rules.TensorBoard import TensorBoard, Move

def read_pkl(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def decode(index: int) -> Move:
    assert index >= 0 and index < 8 * 8 * 78
    s = np.zeros((4992))
    s[index] = 1
    s = s.reshape(8, 8, 78)
    move = TensorBoard.decode_tensor_to_moves(s)
    assert len(move) == 1
    return move[0]


ROOT = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning/5'

files = list(os.listdir(ROOT))
for filename in files[0:]:
    s = read_pkl(os.path.join(ROOT, filename))
    max_idx = np.argmax(s[-1][1])
    mv = decode(max_idx)
    first_value = s[0][2]
    s = TensorBoard.decode_tensor_to_board(s[-1][0][:, :, 0:26])
    s['board'].display()
    print('First value', first_value)
    print('Expect move: ', mv)
    print('Current Turn: ', s['turn'])
    print('*********************')
