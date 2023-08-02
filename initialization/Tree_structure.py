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

import numpy as np
import copy
import sys
from multiprocessing import Manager,  Process, Pipe, Pool, managers
from collections import defaultdict


Select_Value_dict = {}
Step_now = 0
number_of_pipes_workers = 10
parent_pipes_workers = []
child_pipes_workers = []
for i in range(number_of_pipes_workers):
    parent_pipe, child_pipe = Pipe() 
    parent_pipes_workers.append(parent_pipe)
    child_pipes_workers.append(child_pipe)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def proc_send_parallel_workers(state_copy, pipe, i):   
    result = []
    result.append(i)
    pipe.send(result)
    return

node_dict = defaultdict(list)

class TreeNode(object):   #  The tree node for the mcts process
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} 
        self._n_visits = 0
        self._n_visits_now = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._Select_Value = self._Q + self._u
        self.action_num = 0

    def expand(self, action_priors):  # expand the leaf node
        cal = 0
        for action, prob in action_priors:
            cal += 1
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
                self._children[action].action_num = action
  
    def select(self, c_puct, state):  # select the child node
        if id(self) in node_dict and node_dict[id(self)][0] in self._children:
            return node_dict[id(self)][0], self._children[node_dict[id(self)][0]]
        else:
            action_num, node = max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
            node_dict[id(self)].append(action_num)
            node_dict[id(self)].append(self._Q + self._u)
            return action_num, node
        
    def update(self, leaf_value): # update attributes of nodes  
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        if not self._parent:
            pass
        elif not node_dict[id(self._parent)]:
            pass
        else:
            self._u = (5 * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
            if node_dict[id(self._parent)]:             
                if self._Q + self._u > node_dict[id(self._parent)][1]:
                    node_dict[id(self._parent)].append(self.action_num)
                    node_dict[id(self._parent)].append(self._Q + self._u)

    def update_recursive(self, leaf_value):   # update attributes of nodes recursively 
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):   # get the value of a child node
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):  # determine whether it is the leaf node 
        return self._children == {}

    def is_root(self): # determine whether it is the root node 
        return self._parent is None

class MCTS(object):  
    def __init__(self, policy_value_forward, c_puct, n_MCTS):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_forward
        self._c_puct = c_puct
        self._n_MCTS = n_MCTS

    def _step(self, state, n, index, mcts):   # the process to build and update the tree
        node = self._root
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct, state)                     
            state.add_link(action, 1)
        action_probs, leaf_value = self._policy(state)  
        end, delay_result, ratio, average_degree, delay_num, penalty = state.game_end(0, mcts)     
        if not end:        
            node.expand(action_probs)
        else:
            if delay_result: 
                leaf_value = (-0.5 * delay_num + (1 - 0.5) * ratio * 100)/100      
                node.update_recursive(leaf_value)  

    def call_back(self, return_list):
        leaf_value = (-0.5 * return_list[1] + (1 - 0.5) * return_list[0] * 100)/100 
        node.update_recursive(leaf_value) 

    def get_link_probs(self, mcts, state, index, temp=1e-3):  
        global mem_acts
        global mem_visits

        for n in range(self._n_MCTS):
            state_copy = copy.deepcopy(state)
            self._step(state_copy, n, index, mcts)   # multiple times
       
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]       
        if len(act_visits) == 0:              
            acts, visits = mem_acts, mem_visits
        else:
            acts, visits = zip(*act_visits)
            mem_acts = acts
            mem_visits = visits
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-3))
        return acts, act_probs

    def update_with_link(self, last_link): 
        if last_link in self._root._children:
            self._root = self._root._children[last_link]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

class MCTSAgent(object):  

    def __init__(self, policy_value_function, c_puct, n_MCTS, is_MCTS):
        self.mcts = MCTS(policy_value_function, c_puct, n_MCTS)
        self._is_MCTS = is_MCTS

    def get_action(self, board, index, temp, ACTION_DIM, return_prob=1):  # get the next action 
        link_probs = np.zeros(ACTION_DIM,)
        acts, probs = self.mcts.get_link_probs(self.mcts, board, index, temp)
        link_probs[list(acts)] = probs
        link = np.random.choice(acts, p=probs)
        self.mcts.update_with_link(link)
        return link, link_probs

