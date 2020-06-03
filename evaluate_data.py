import argparse
import logging
import os
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str)
parser.add_argument('--last_iter', type=int)
args = parser.parse_args()

logging.basicConfig(filename=f'logs/{args.last_iter + 1}/eval_data.txt',
                    filemode='w',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info('\n\n********* EVALUATE DATA *********\n\n')
logging.info(args._get_kwargs())

PATH = os.path.join(args.dataroot, str(args.last_iter + 1))
for filename in os.listdir(PATH):
    filepath = os.path.join(PATH, filename)

    logging.info(f'Load {filepath}')
    with open(filepath, 'rb') as f:
        game = pickle.load(f)
    first_board, first_pi, first_value = game[0]
    flag = first_value
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

    logging.info(f'Value of team 1: {flag} - team 0: {-1 * flag}')
logging.info('Evaluate Sucessfully\n\n')
