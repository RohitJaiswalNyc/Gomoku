"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
import numpy as np
import pylab as pl
import torch as tr
import itertools as it
import matplotlib.pyplot as pt
import random
from scipy.signal import correlate
import gomoku as gm
import pickle

device = "cuda" if tr.cuda.is_available() else "cpu"

class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.state_log = []
        self.action_log = []
        self.loss_history = []
        self.random_choice = 0
        self.random_choice_decelerator = 0.995
        self.targ = None
        self.gamma = 0.1
        self.learning_rate = 1e-1
        self.loss_sum = 0

        self.valid_actions = None
        self.num_filters = 5 # number of different patterns scanned across the image
        self.kernel_size = 3 # size of each filter

        self.linear_relu_stack = tr.nn.Sequential(
            tr.nn.Conv2d(3,out_channels=128,kernel_size=self.kernel_size), # 1 input channel
            tr.nn.ReLU(),
            tr.nn.Conv2d(128,out_channels=128,kernel_size=self.kernel_size), # 1 input channel
            tr.nn.ReLU(),
            tr.nn.Conv2d(128,out_channels=64,kernel_size=self.kernel_size), # 1 input channel
            tr.nn.ReLU(), # relu(x) = max(x, 0)
            tr.nn.Flatten(),
            tr.nn.Linear(5184, 15*15),  # 225 output neurons (1 per digit)

        ).to(device)
        self.loss_fn = tr.nn.MSELoss()
        self.optimizer = tr.optim.Adadelta(self.linear_relu_stack.parameters(),lr=self.learning_rate)



    def save_nn(self):
        tr.save({
            'model_state_dict': self.linear_relu_stack.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }, 'nn.pt')

    def save_moves(self):
        with open('listfile', 'ab') as fp:
            pickle.dump(self.loss_sum, fp)
        self.loss_sum = 0

    def load_nn(self):
        model = self.linear_relu_stack
        optimizer = self.optimizer
        checkpoint = tr.load('nn.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        return model,optimizer,loss

    def trans(self):
        x = self.loss_history
        Y = x
        X = []
        for i in range(len(x)):
            X.append(i)

        return X,Y

    def show(self):
        x,y = self.trans()
        #
        pt.subplot(1,1,1)
        pt.plot(x,y,'ro')
        pt.plot(x,y)
        #
        pt.tight_layout()
        pt.show()
        self.loss_history = []

    def train_nn(self,targ):
        # targ is the total score which indicates the reward for every action
        # decelerates with reward_decelerator
        self.targ = targ
        model,optimizer,loss_fn = self.load_nn()
        # use the model in training mode
        model.train()
        targ = float(targ)

        index = 0

        # for every action in each state
        self.save_moves()

        for state in reversed(self.state_log):
            reward = targ

            x = tr.tensor(state)
            x = x.type(tr.FloatTensor)
            x = x.view(1,3,15,15)
            out = model(x)
            out = out.view(225)
            action = self.action_log[index]
            board_index = action[0]*15 + action[1]
            targ_t = out.clone()
            targ_t[board_index] = reward

            # Compute prediction error
            loss = loss_fn(out.type(tr.FloatTensor), targ_t.type(tr.FloatTensor))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            self.loss_sum += loss
            # Progress update
            self.loss_history.append(loss)
            index += 1
            reward *= self.gamma

        # self.show()
        self.state_log = []
        self.action_log = []
        self.save_nn()
        self.random_choice *= self.random_choice_decelerator
        pass

    def clear_non_valid(self,q_val):
        for index in range(225):
            row = index//15
            col = index % 15
            if (row,col) not in self.valid_actions:
                q_val[index] = -1
        return q_val

    def q_learn_score(self,state):
        r = random.randint(0,1)
        if r < self.random_choice:
            ind = random.randint(0,len(self.valid_actions)-1)
            self.state_log.append(state)
            self.action_log.append(self.valid_actions[ind])
            return 0,self.valid_actions[ind]

        model,optimizer,loss_fn = self.load_nn()
        state = np.array(state).astype(np.single)
        x = tr.tensor(state)
        x = x.view(1,3,15,15)


        model.eval()
        q_val = model(x)
        q_val = q_val.view(225)
        q_val = tr.nn.functional.softmax(q_val,dim=0)

        q_val = self.clear_non_valid(q_val)
        action = q_val.argmax(dim=0)
        row = action//15
        col = action % 15

        row_f = int(row)
        col_f = int(col)
        action = (row_f,col_f)
        self.state_log.append(state)
        self.action_log.append(action)
        return 0,action

    def minimax(self,state):
        score,action = self.q_learn_score(state.board)
        return score,action

nn = NeuralNetwork()


class Submission:
    def __init__(self, board_size, win_size, max_depth=2):
        self.max_depth = max_depth
        # nn.save_nn()
        pass

    def __call__(self, state,flag=True,final_score=0):
        ### Replace with your implementation
        if flag:
            nn.valid_actions = state.valid_actions()
            score, action = nn.minimax(state)
            return action
        else:
            nn.train_nn(final_score)
            return 0

