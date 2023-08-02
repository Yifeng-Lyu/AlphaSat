# The MIT License (MIT)
#
# Copyright (c) 2023 Yifeng Lyu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import networkx as nx
import datetime
import time
import sys
from copy import deepcopy
from collections import deque
import random as rm
import torch.optim as optim
from Env import *
from Tree_structure import *
import json
import os
from multiprocessing import Manager, Process, Pipe, Pool
import torch.multiprocessing as mp


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class ResidualBlock(nn.Module):   # residual blocks
    def __init__(self, out_channels): 
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x 
        out = self.relu2(out)
        return out

class Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)        
        self.res_layer = self.make_residual_layers(4, 1)        
        self.act_fc1 = nn.Linear(state_dim, 100)
        self.act_fc2 = nn.Linear(100, action_dim)             
        self.val_conv1 = nn.Conv2d(1, 1, kernel_size=1)
        self.val_fc1 = nn.Linear(state_dim, 64)
        self.val_fc2 = nn.Linear(64, 1)  
       
    def make_residual_layers(self, blocks, out_channels):  #  residual layers
        layers = []
        for i in range(blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, state_input):
        state_input = state_input.view(-1, 1, NUM_ORBITS*NUM_SATS_PER_ORBIT, NUM_ORBITS*NUM_SATS_PER_ORBIT)
        x = self.res_layer(state_input)
        
        x_act = x.view(-1, NUM_ORBITS*NUM_SATS_PER_ORBIT*NUM_ORBITS*NUM_SATS_PER_ORBIT)     
        x_act = self.act_fc1(x_act)  
        x_act = F.log_softmax(self.act_fc2(x_act))

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, NUM_ORBITS*NUM_SATS_PER_ORBIT*NUM_ORBITS*NUM_SATS_PER_ORBIT)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file="../output/initialization_" + config + "/model.pth", use_gpu=True):              
        self.use_gpu = use_gpu
        self.state_dim =  TOTAL_NUM_SATS*TOTAL_NUM_SATS           
        self.action_dim = ACTION_DIM     
        self.board_width = TOTAL_NUM_SATS
        self.board_height = TOTAL_NUM_SATS
        self.l2_const = 1e-4  

        if self.use_gpu:
            self.policy_value_net = Net(self.state_dim, self.action_dim).cuda()
        else:
            self.policy_value_net = Net(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

    def policy_value(self, state_batch):  
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_forward(self, board):   # predict the action probability and value of the state by correponding networks
        legal_positions = board.availables_test
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(board.current_state())).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(board.current_state())).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):  # loss functions
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()  
        torch.save(net_params, model_file)

class Initialization():
    def __init__(self, init_model=None):

        self.board_width = TOTAL_NUM_SATS
        self.board_height = TOTAL_NUM_SATS
        self.board = Board()
        self.env = Env(self.board)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate 
        self.temp = 1.0  # the temperature param
        self.n_MCTS = 100  # num of simulations for each link
        self.c_puct = 5
        self.buffer_size = 1000
        self.batch_size = 128
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5  
        self.kl_targ = 0.02
        self.check_freq = 1
        self.episodes = 100 

        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.self_agent = MCTSAgent(self.policy_value_net.policy_value_forward, c_puct=self.c_puct, n_MCTS=self.n_MCTS, is_MCTS=1)

    def collect_MCTS_data(self, index):  #  collect the data 
        temp=1e-3
        winner, MCTS_data = self.env.start_MCTS(self.self_agent, temp, index, ACTION_DIM)
        self.data_buffer.extend(MCTS_data)  

    def policy_update(self):   # update the neural network with collected date
        while len(self.data_buffer) < self.batch_size:
            self.batch_size = int(len(self.data_buffer)/2)
        mini_batch = rm.sample(self.data_buffer, self.batch_size)  
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs): 
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))                      
        return loss, entropy

    def run(self):
        for i in range(self.episodes): 
            self.collect_MCTS_data(i)  
            loss, entropy = self.policy_update()
    
if __name__ == '__main__':     
    initialization = Initialization()
    initialization.run()
