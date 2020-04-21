from supervised_learning.utils import read_yaml, save_yaml

configs = read_yaml('supervised_learning/configs.yml')

s = open(configs['preprocess_data']['input']).read()

d = s.split('PlyCount "')
d = d[1:]
n_moves = [int(x.split('"')[0]) for x in d]

s = s.replace('x', '')
s = s.replace('+', '')
s = s.replace('#', '')
s = s.replace('=Q', '')
s = s.split('\n\n')
data = []
for i in range(len(s)):
    if i % 2 == 1:
        data.append(s[i])

matches = []
for match in data:
    s = match.split('{')
    s = s[0]
    s = s.split(' ')
    t = []
    for u in s:
        if '.' in u or not u:
            continue
        t.append(u)
    matches.append(t)

assert len(matches) == len(n_moves)
for i in range(len(matches)):
    assert len(matches[i]) == n_moves[i]

max_v = -1000
min_v = 1000
s_4 = []
filter_matches = []
for match in matches:
    flag = False
    for move in match:
        if '=' in move:
            flag = True
        if flag:
            break
    if not flag:
        filter_matches.append(match)

for match in filter_matches:
    flag = False
    for move in match:
        if move == 'O-O-O' or move == 'O-O':
            continue
        s = list(move)
        for u in s:
            if not u.isalpha() and not u.isdigit():
                flag = True
                break
        if flag:
            print(move)
            break

dataset = {}
for i in range(len(filter_matches)):
    if len(filter_matches[i]) > 40:
        dataset[i] = filter_matches[i]

save_yaml(configs['preprocess_data']['output'], dataset)
