import argparse
import logging
import os
import pickle

from sl_data_processing.utils import read_yaml, save_yaml

logging.basicConfig(filename='sl_data_processing/logs.txt',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str)
args = parser.parse_args()

configs = read_yaml('sl_data_processing/configs.yml')

s = open(
    os.path.join(configs['preprocess_data']['input'], args.time + '.pgn')).read()

logging.info(f'TIME: {args.time}')
# get number of moves
d = s.split('PlyCount "')
d = d[1:]
n_moves = [int(x.split('"')[0]) for x in d]

# get results
d = s.split('Result "')
d = d[1:]
results = [x.split('"')[0] for x in d]

assert len(results) == len(n_moves)

# get moves
s = s.replace('x', '')
s = s.replace('+', '')
s = s.replace('#', '')
s = s.replace('=Q', '')
s = s.split('\n\n')
data = []
for i in range(len(s)):
    if i % 2 == 1:
        data.append(s[i])

games = []
for game in data:
    s = game.split('{')
    s = s[0]
    s = s.split(' ')
    t = []
    for u in s:
        if '.' in u or not u:
            continue
        t.append(u)
    games.append(t)

# make sure games and n_moves is correspond
assert len(games) == len(n_moves)
for i in range(len(games)):
    assert len(games[i]) == n_moves[i]

logging.info(f'No.games: {len(n_moves)}')
logging.info(f'Average moves per game: {sum(n_moves)/len(n_moves)}')

# get results and matches wich n_moves >= 100
dataset = []
for i in range(len(games)):
    if len(games[i]) < 100:
        continue
    if results[i] == '1/2-1/2':
        continue
    s = {'result': results[i], 'moves': games[i]}
    dataset.append(s)

logging.info(f'No.games after remove draw and < 100 moves: {len(dataset)}')

# make sure white moves first
for match in dataset:
    first_move = match['moves'][0]
    assert '3' in first_move or '4' in first_move

# filter promote to knight, rook and bishop because I don't code for it
filtered_dataset = []
for match in dataset:
    flag = False
    for move in match['moves']:
        if '=' in move:
            flag = True
        if flag:
            break
    if not flag:
        filtered_dataset.append(match)

logging.info(f'No.games after filter promoted to knight, rook and bishop: {len(filtered_dataset)}')

# make sure there is no special synbol in moves excep 'O-O-O' and 'O-O'
for match in filtered_dataset:
    flag = False
    for move in match['moves']:
        if move == 'O-O-O' or move == 'O-O':
            continue
        s = list(move)
        for u in s:
            if not u.isalpha() and not u.isdigit():
                flag = True
                break
        if flag:
            raise RuntimeError(f'{move} is not valid')

white_1_won = []
black_0_won = []
for match in filtered_dataset:
    if match['result'] == '0-1':
        black_0_won.append(match['moves'])
    elif match['result'] == '1-0':
        white_1_won.append(match['moves'])
    else:
        raise RuntimeError(f'{match} is invalid match')

logging.info(f'No.games white-1 won: {len(white_1_won)}')
logging.info(f'No.games black-0 won: {len(black_0_won)}')

white_file = os.path.join(configs['preprocess_data']['output'], args.time + '_white_1_won.pkl')
black_file = os.path.join(configs['preprocess_data']['output'], args.time + '_black_0_won.pkl')

logging.info(f'save white 1 won at: {white_file}')
logging.info(f'save white 1 won at: {black_file}')
with open(white_file, 'wb') as f:
    pickle.dump(white_1_won, f)

with open(black_file, 'wb') as f:
    pickle.dump(black_0_won, f)

logging.info('completed')
