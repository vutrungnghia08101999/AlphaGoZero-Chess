import copy
import math
import numpy as np

import torch

from chess_rules.TensorBoard import TensorBoard, Board
from mcts.ModeInference import predict
from sl_traning.model import ChessModel
from mcts.Node import Node

model = ChessModel()
checkpoint = torch.load(
    '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/SL/checkpoint_20-04-26_14_06_38_39.pth',
    map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
tensor_board = TensorBoard(Board(), Board(), 1)

root = Node(1, tensor_board, model, parent=None)
