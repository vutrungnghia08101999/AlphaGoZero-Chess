import argparse
import logging
import os
import numpy as np
import pickle
import time
from tqdm import tqdm
import copy

import torch

from chess_rules.ChessObjects import Board, Move
from chess_rules.TensorBoard import TensorBoard
from main.MCTS import MCTSNode
from main.model import ChessModel

model = ChessModel()
model.eval()

tensor_board = TensorBoard(Board(), Board(), 1)
root = MCTSNode(
    move=None,
    model=model,
    index=-1,
    perspective=tensor_board.turn,
    parent=None,
    tensor_board=tensor_board,
    )
root.expand_and_backpropagate()
s = root.traverse()
s.expand_and_backpropagate()
