#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
import encoder_decoder_c4 as ed
from connect_board import board as c_board
import copy
import torch
import torch.multiprocessing as mp
from alpha_net_c4 import ConnectNet
import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# def save_as_pickle(filename, data):
#     completeName = os.path.join("./datasets/",\
#                                 filename)
#     with open(completeName, 'wb') as output:
#         pickle.dump(data, output)

# def load_pickle(filename):
#     completeName = os.path.join("./datasets/",\
#                                 filename)
#     with open(completeName, 'rb') as pkl_file:
#         data = pickle.load(pkl_file)
#     return data

class UCTNode():
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_idxes = []
        
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
          current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.actions(); c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000 # mask all illegal actions
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self,board,move):
        board.drop_piece(move)
        return board
            
    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = O wins
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent
        
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

def UCT_search(game_state, num_reads,net,temp):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in tqdm(range(num_reads)):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float()#.cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if leaf.game.check_winner() == True or leaf.game.actions() == []: # if somebody won or draw
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
    return root

def do_decode_n_move_pieces(board,move):
    board.drop_piece(move)
    return board

def get_policy(root, temp=1):
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))

def MCTS_self_play(connectnet, num_games, iteration):
    logging.info(f'Self-play {num_games} games at iteration {iteration}')
    output = os.path.join('dataset', f"iteration_{iteration}")
    os.makedirs(output, exist_ok=True)
        
    for game in range(0, num_games):
        logging.info(f'game: {game}/{num_games}')
        current_board = c_board()
        logging.info('\n' + str(current_board.current_board))
        logging.info(f'Turn: {current_board.player}')
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.actions() != []:
            if move_count < 11:
                t = 1
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            n_simulations = 777
            logging.info(f"Search next move with MCTS and {n_simulations} simulations")
            root = UCT_search(current_board,n_simulations,connectnet,t)
            policy = get_policy(root, t)
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy)) # decode move and move piece(s)
            dataset.append([board_state,policy])
            logging.info('\n' + str(current_board.current_board))
            logging.info(f'Turn: {current_board.player}')
            if current_board.check_winner() == True: # if somebody won
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        # # del dataset
        # with open(os.path.join(output, f"game_{game}"), 'wb') as output:
        #     pickle.dump(dataset_p, output)
        # save data after a game


def run_MCTS(num_games, iteration):
    net = ConnectNet()
    net.eval()

    best_model = os.path.join('models', 'best_model.pth')
    if os.path.isfile(best_model):
        checkpoint = torch.load(best_model)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        torch.save({'state_dict': net.state_dict()}, best_model)
    
    with torch.no_grad():
        MCTS_self_play(net, num_games, iteration)
