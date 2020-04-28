import logging
import os
import pickle
from tqdm import tqdm

from alphazero.utils import read_yaml

logging.basicConfig(level=logging.INFO)

configs = read_yaml('alphazero/configs.yml')

for iteration in configs['evaluate_data']:
    path = os.path.join(configs['dataroot'], str(iteration) + '.pkl')

    logging.info(f'Load {path}')
    with open(path, 'rb') as f:
        games = pickle.load(f)
    for k, game in games.items():
        logging.info(f'Game: {k}')
        first_board, first_pi, first_value = game[0]
        flag = first_value
        first_state = first_board[:, :, 0:26]
        first_valid_moves = first_board[:, :, 26:104]

        s = (first_pi > 0) + 0
        assert sum(sum(sum(s == first_valid_moves))) == 8 * 8 * 78
        assert abs(sum((sum(sum(first_pi)))) - 1) < 1e-2

        l_last_board = first_state[:, :, 12:24]
        l_board = first_state[:, :, 0:12]
        l_turn = first_state[:, :, 24][0][0]
        l_nmvs = first_state[:, :, 25][0][0]
        assert sum(sum(sum(l_last_board))) == 32
        assert sum(sum(sum(l_board))) == 32
        assert l_turn == 1
        assert l_nmvs == 0

        for board, pi, value in tqdm(game[1:]):
            state = board[:, :, 0:26]
            valid_moves = board[:, :, 26:104]

            s = (pi > 0) + 0
            assert sum(sum(sum(s == valid_moves))) == 8 * 8 * 78
            assert abs(sum((sum(sum(pi)))) - 1) < 1e-2

            last_board = state[:, :, 12:24]
            board = state[:, :, 0:12]
            turn = state[:, :, 24][0][0]
            nmvs = state[:, :, 25][0][0]

            assert sum(sum(sum(last_board == l_board))) == 8 * 8 * 12
            assert abs(int(turn) - int(l_turn)) == 1
            assert nmvs - l_nmvs == 1

            l_last_board = last_board
            l_board = board
            l_turn = turn
            l_nmvs = nmvs

        if flag == 0:  # draw or finished after n_moves
            for _, _, value in game:
                assert value == 0
        elif flag == 1:  # white won
            for i in range(len(game)):
                _, _, value = game[i]
                if i % 2 == 0:
                    assert value == 1
                else:
                    assert value == -1
        elif flag == -1:  # black won
            for i in range(len(game)):
                _, _, value = game[i]
                if i % 2 == 0:
                    assert value == -1
                else:
                    assert value == 1
        else:
            raise RuntimeError(f'{flag} is invalid value')

logging.info('Evaluate Sucessfully')
