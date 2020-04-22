import argparse
import numpy as np
import pickle
from tqdm import tqdm

from TensorBoard import TensorBoard

DATA_TEST = '/media/vutrungnghia/New Volume/ArtificialIntelligence/dataset/supervised-learning/small_data.pkl'
parser = argparse.ArgumentParser()
parser.add_argument('--s', type=int)
parser.add_argument('--e', type=int)
args = parser.parse_args()

with (open(DATA_TEST, "rb")) as openfile:
    dataset = pickle.load(openfile)

X, Y, Z = dataset[0]
assert sum(sum(sum(Y))) == 20
assert sum(sum(sum(Z))) == 1
x, y, z = np.argwhere(Z == 1)[0]
assert Y[x][y][z] == 1
X_board = X[:, :, 0:12]
X_last_board = X[:, :, 12:24]
X_turn = X[:, :, 24][0][0]
X_n_mvs = X[:, :, 25][0][0]

for (state, valid_actions, expect_action) in tqdm(dataset[1:]):
    assert sum(sum(sum(valid_actions))) < 86
    assert sum(sum(sum(expect_action))) == 1
    x, y, z = np.argwhere(expect_action == 1)[0]
    assert valid_actions[x][y][z] == 1

    board = state[:, :, 0:12]
    last_board = state[:, :, 12:24]
    turn = state[:, :, 24][0][0]
    n_mvs = state[:, :, 25][0][0]

    assert int(n_mvs) - int(X_n_mvs) == 1
    assert abs(int(turn) - int(X_turn)) == 1
    assert sum(sum(sum(last_board == X_board))) == 8 * 8 * 12
    X_board = board
    X_last_board = last_board
    X_turn = turn
    X_n_mvs = n_mvs

dataset = dataset[args.s:args.e]
for i in range(len(dataset)):
    x = TensorBoard.decode_tensor_to_board(dataset[i][0])
    x['board'].display()
    print('n_mvs: ' + str(x['board'].n_mvs))
    print('turn: ' + str(x['turn']))
    a = TensorBoard.decode_tensor_to_moves(dataset[i][1])
    for u in a:
        print(u)
    print('****')
    b = TensorBoard.decode_tensor_to_moves(dataset[i][2])
    for u in b:
        print(u)
    print('========================')
