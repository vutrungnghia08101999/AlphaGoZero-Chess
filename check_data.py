import os
import pickle

from chess_rules.TensorBoard import TensorBoard

def read_pkl(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


ROOT = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning/4'

files = list(os.listdir(ROOT))
for filename in files[0:]:
    s = read_pkl(os.path.join(ROOT, filename))
    s = TensorBoard.decode_tensor_to_board(s[-1][0][:, :, 0:26])
    s['board'].display()
