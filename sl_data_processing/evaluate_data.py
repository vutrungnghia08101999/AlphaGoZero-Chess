import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

from sl_data_processing.utils import read_yaml

configs = read_yaml('sl_data_processing/configs.yml')

for folder in configs['evaluate_data']['folders']:
    white_won = 'white' in folder
    path = os.path.join(configs['evaluate_data']['root'], folder)
    print(path)
    print('=================================')
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'rb') as f:
            games = pickle.load(f)
        print(filename)
        print(f'white won: {white_won}')

        for k, game in tqdm(games.items()):
            assert len(game) >= 50
            # print(len(game))
            first_state, first_valid_moves, first_expect_move = game[0]
            assert sum(sum(sum(first_valid_moves))) > 0 and sum(sum(sum(first_valid_moves))) < 532
            assert sum(sum(sum(first_expect_move))) == 1
            s = np.argwhere(first_expect_move == 1)
            assert s.shape[0] == 1 and s.shape[1] == 3
            assert first_valid_moves[s[0][0]][s[0][1]][s[0][2]] == 1
            l_last_board = first_state[:, :, 12:24]
            l_board = first_state[:, :, 0:12]
            l_turn = first_state[:, :, 24][0][0]
            l_nmvs = first_state[:, :, 25][0][0]
            assert sum(sum(sum(l_last_board))) == 32
            assert sum(sum(sum(l_board))) == 32
            if white_won:
                assert l_turn == 1
                assert l_nmvs % 2 == 0
            else:
                assert l_turn == 0
                assert l_nmvs % 2 == 1

            for state, valid_moves, expect_move in game[1:]:
                assert sum(sum(sum(valid_moves))) > 0 and sum(sum(sum(valid_moves))) < 532
                assert sum(sum(sum(expect_move))) == 1
                s = np.argwhere(expect_move == 1)
                assert s.shape[0] == 1 and s.shape[1] == 3
                assert valid_moves[s[0][0]][s[0][1]][s[0][2]] == 1
                last_board = state[:, :, 12:24]
                board = state[:, :, 0:12]
                turn = state[:, :, 24][0][0]
                nmvs = state[:, :, 25][0][0]

                assert turn == l_turn
                assert nmvs == l_nmvs + 2
                if white_won:
                    assert turn == 1
                    assert nmvs % 2 == 0
                else:
                    assert turn == 0
                    assert nmvs % 2 == 1
                n = sum(sum(sum(last_board != l_board)))
                # print(n)
                assert n <= 4 and n >= 2

                l_last_board = last_board
                l_board = board
                l_turn = turn
                l_nmvs = nmvs

print('Evaluate data successfully!')
