import numpy as np

import torch
import torch.nn.functional as F

from sl_traning.model import ChessModel
from chess_rules.TensorBoard import TensorBoard, Move, Board

def decode(index: int) -> Move:
    assert index >= 0 and index < 8 * 8 * 78
    s = np.zeros((4992))
    s[index] = 1
    s = s.reshape(8, 8, 78)
    move = TensorBoard.decode_tensor_to_moves(s)
    assert len(move) == 1
    return move[0]

def predict(model: ChessModel, tensor_board: TensorBoard) -> list:
    encoded_board = tensor_board.encode_board_to_tensor()
    encoded_board = torch.tensor(encoded_board, dtype=torch.float)
    encoded_board = encoded_board.view(-1, 8, 8, 26)
    with torch.no_grad():
        pred = model(encoded_board)
    pred = pred.squeeze()
    valid_moves = tensor_board.encode_actions_to_tensor()
    valid_moves = torch.tensor(valid_moves).flatten()
    pred = pred - (valid_moves == 0) * 10
    pred = F.softmax(pred, dim=0)
    pred = pred * (valid_moves == 1)
    pred = pred + (1e-10) * (valid_moves == 1)
    n_mvs = int(sum(valid_moves))
    probs, indices = torch.topk(pred, k=n_mvs, dim=0, largest=True)
    predictions = []
    for i in range(n_mvs):
        move = decode(indices[i])
        s = tensor_board.get_next_state(move)
        prob = probs[i]
        predictions.append((move, prob, s))
    return predictions
