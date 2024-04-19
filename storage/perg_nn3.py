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

device = "cuda" if tr.cuda.is_available() else "cpu"

class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.state_log = []
        self.action_log = []
        self.loss_history = []
        self.random_choice = 0.001
        self.random_choice_decelerator = 0.995
        self.targ = None
        self.gamma = 0.6
        self.learning_rate = 1e-1
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
            tr.nn.Linear(81, 15*15),  # 225 output neurons (1 per digit)
        ).to(device)
        self.loss_fn = tr.nn.MSELoss()
        self.optimizer = tr.optim.Adadelta(self.linear_relu_stack.parameters(),lr=self.learning_rate)

    def save_nn(self):
        tr.save({
            'model_state_dict': self.linear_relu_stack.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }, '../code/other/nn.pt')

    def load_nn(self):
        model = self.linear_relu_stack
        optimizer = self.optimizer
        checkpoint = tr.load('../code/other/nn.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        return model,optimizer,loss

    def show(self):
        pt.subplot(1,3,1)
        pt.hist(self.loss_history, ec='k')
        pt.ylabel("Frequency")
        pt.xlabel("loss")
        pl.plot(self.targ)
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
        for state in reversed(self.state_log):
            reward = targ

            x = tr.tensor(state)
            x = x.type(tr.FloatTensor)
            out = model(x)
            action = self.action_log[index]
            board_index = action[0]*15 + action[1]
            targ_t = out.clone()
            for i in range(64):
                targ_t[i][board_index] = reward

            # Compute prediction error
            loss = loss_fn(out.type(tr.FloatTensor), targ_t.type(tr.FloatTensor))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            # Progress update
            self.loss_history.append(loss)
            index += 1
            reward *= self.gamma


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
                q_val[index] = 0
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

        model.eval()
        q_val = model(x)
        q_val = q_val.sum(dim=0)
        q_val = tr.softmax(q_val,dim=0)
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

    def turn_bound(self,state):

        is_max = state.is_max_turn()
        fewest_moves = state.board[gm.EMPTY].sum() # moves to a tie game

        # use correlations to extract possible routes to a non-tie game
        corr = state.corr
        min_routes = (corr[:,gm.EMPTY] + corr[:,gm.MIN] == state.win_size)
        max_routes = (corr[:,gm.EMPTY] + corr[:,gm.MAX] == state.win_size)
        # also get the number of turns in each route until game over
        min_turns = 2*corr[:,gm.EMPTY] - (0 if is_max else 1)
        max_turns = 2*corr[:,gm.EMPTY] - (1 if is_max else 0)
        # check if there is a shorter path to a game-over state
        if min_routes.any():
            moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)
        if max_routes.any():
            moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)

        # return the shortest path found to a game-over state
        return fewest_moves

    # helper to find empty position in pth win pattern starting from (r,c)
    def find_empty(self,state, p, r, c):
        if p == 0: # horizontal
            return r, c + state.board[gm.EMPTY, r, c:c+state.win_size].argmax()
        if p == 1: # vertical
            return r + state.board[gm.EMPTY, r:r+state.win_size, c].argmax(), c
        if p == 2: # diagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
            return r + offset, c + offset
        if p == 3: # antidiagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
            return r - offset, c + offset
        # None indicates no empty found
        return None

    # fast look-aheads to short-circuit the minimax search when possible
    def look_ahead(self,state):

        # if current player has a win pattern with all their marks except one empty, they can win next turn
        player = state.current_player()
        sign = +1 if player == gm.MAX else -1
        magnitude = state.board[gm.EMPTY].sum() # no +1 since win comes after turn
        # NN

        # check if current player is one move away to a win
        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size-1))
        if idx.shape[0] > 0:
            # find empty position they can fill to win, it is an optimal action
            p, r, c = idx[0]
            action = self.find_empty(state, p, r, c)
            return sign * magnitude, action

        # else, if opponent has at least two such moves with different empty positions, they can win in two turns
        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        loss_empties = set() # make sure the 2+ empty positions are distinct
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            loss_empties.add(pos)
            if len(loss_empties) > 1: # just found a second empty
                score = -sign * (magnitude - 1) # opponent wins an extra turn later
                return score, pos # block one of their wins with next action even if futile

        # return 0 to signify no conclusive look-aheads
        return 0, None

    def minimax(self,state, max_depth, alpha=-np.inf, beta=np.inf):
        # check fast look-ahead before trying minimax if game is sure to end
        score, action = self.look_ahead(state)
        if score != 0:
            return score, action

        # check for game over base case with no valid actions
        if state.is_game_over():
            # NN
            return state.current_score(), None

        # have to try minimax, prepare the valid actions
        # should be at least one valid action if this code is reached
        actions = state.valid_actions()

        # prioritize actions near non-empties but break ties randomly
        # NN
        rank = -state.corr[:, 1:].sum(axis=(0,1)) - np.random.rand(*state.board.shape[1:])
        rank = rank[state.board[gm.EMPTY] > 0] # only empty positions are valid actions
        scrambler = np.argsort(rank)
        # check for max depth base case

        if max_depth == 0:
            # NN
            score,action = self.q_learn_score(state.board)
            return score,action

        # alpha-beta pruning
        best_action = None
        if state.is_max_turn():
            bound = -np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth-1, alpha, beta)

                if utility > bound: bound, best_action = utility, action
                if bound >= beta: break
                alpha = max(alpha, bound)

        else:
            bound = +np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth-1, alpha, beta)

                if utility < bound: bound, best_action = utility, action
                if bound <= alpha: break
                beta = min(beta, bound)

        return bound, best_action


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
            score, action = nn.minimax(state, self.max_depth)
            return action
        else:
            nn.train_nn(final_score)
            return 0







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

device = "cuda" if tr.cuda.is_available() else "cpu"

class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.state_log = []
        self.action_log = []
        self.loss_history = []
        self.random_choice = 0.001
        self.random_choice_decelerator = 0.995
        self.targ = None
        self.gamma = 0.6
        self.learning_rate = 1e-1
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
            tr.nn.Linear(81, 15*15),  # 225 output neurons (1 per digit)
        ).to(device)
        self.loss_fn = tr.nn.MSELoss()
        self.optimizer = tr.optim.Adadelta(self.linear_relu_stack.parameters(),lr=self.learning_rate)

    def save_nn(self):
        tr.save({
            'model_state_dict': self.linear_relu_stack.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }, '../code/other/nn.pt')

    def load_nn(self):
        model = self.linear_relu_stack
        optimizer = self.optimizer
        checkpoint = tr.load('../code/other/nn.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        return model,optimizer,loss

    def show(self):
        pt.subplot(1,3,1)
        pt.hist(self.loss_history, ec='k')
        pt.ylabel("Frequency")
        pt.xlabel("loss")
        pl.plot(self.targ)
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
        for state in reversed(self.state_log):
            reward = targ

            x = tr.tensor(state)
            x = x.type(tr.FloatTensor)
            out = model(x)
            action = self.action_log[index]
            board_index = action[0]*15 + action[1]
            targ_t = out.clone()
            for i in range(64):
                targ_t[i][board_index] = reward

            # Compute prediction error
            loss = loss_fn(out.type(tr.FloatTensor), targ_t.type(tr.FloatTensor))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            # Progress update
            self.loss_history.append(loss)
            index += 1
            reward *= self.gamma


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
                q_val[index] = 0
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

        model.eval()
        q_val = model(x)
        q_val = q_val.sum(dim=0)
        q_val = tr.softmax(q_val,dim=0)
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

    def turn_bound(self,state):

        is_max = state.is_max_turn()
        fewest_moves = state.board[gm.EMPTY].sum() # moves to a tie game

        # use correlations to extract possible routes to a non-tie game
        corr = state.corr
        min_routes = (corr[:,gm.EMPTY] + corr[:,gm.MIN] == state.win_size)
        max_routes = (corr[:,gm.EMPTY] + corr[:,gm.MAX] == state.win_size)
        # also get the number of turns in each route until game over
        min_turns = 2*corr[:,gm.EMPTY] - (0 if is_max else 1)
        max_turns = 2*corr[:,gm.EMPTY] - (1 if is_max else 0)
        # check if there is a shorter path to a game-over state
        if min_routes.any():
            moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)
        if max_routes.any():
            moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)

        # return the shortest path found to a game-over state
        return fewest_moves

    # helper to find empty position in pth win pattern starting from (r,c)
    def find_empty(self,state, p, r, c):
        if p == 0: # horizontal
            return r, c + state.board[gm.EMPTY, r, c:c+state.win_size].argmax()
        if p == 1: # vertical
            return r + state.board[gm.EMPTY, r:r+state.win_size, c].argmax(), c
        if p == 2: # diagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
            return r + offset, c + offset
        if p == 3: # antidiagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
            return r - offset, c + offset
        # None indicates no empty found
        return None

    # fast look-aheads to short-circuit the minimax search when possible
    def look_ahead(self,state):

        # if current player has a win pattern with all their marks except one empty, they can win next turn
        player = state.current_player()
        sign = +1 if player == gm.MAX else -1
        magnitude = state.board[gm.EMPTY].sum() # no +1 since win comes after turn
        # NN

        # check if current player is one move away to a win
        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size-1))
        if idx.shape[0] > 0:
            # find empty position they can fill to win, it is an optimal action
            p, r, c = idx[0]
            action = self.find_empty(state, p, r, c)
            return sign * magnitude, action

        # else, if opponent has at least two such moves with different empty positions, they can win in two turns
        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        loss_empties = set() # make sure the 2+ empty positions are distinct
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            loss_empties.add(pos)
            if len(loss_empties) > 1: # just found a second empty
                score = -sign * (magnitude - 1) # opponent wins an extra turn later
                return score, pos # block one of their wins with next action even if futile

        # return 0 to signify no conclusive look-aheads
        return 0, None

    def minimax(self,state, max_depth, alpha=-np.inf, beta=np.inf):
        # check fast look-ahead before trying minimax if game is sure to end
        score, action = self.look_ahead(state)
        if score != 0:
            return score, action

        # check for game over base case with no valid actions
        if state.is_game_over():
            # NN
            return state.current_score(), None

        # have to try minimax, prepare the valid actions
        # should be at least one valid action if this code is reached
        actions = state.valid_actions()

        # prioritize actions near non-empties but break ties randomly
        # NN
        rank = -state.corr[:, 1:].sum(axis=(0,1)) - np.random.rand(*state.board.shape[1:])
        rank = rank[state.board[gm.EMPTY] > 0] # only empty positions are valid actions
        scrambler = np.argsort(rank)
        # check for max depth base case

        if max_depth == 0:
            # NN
            score,action = self.q_learn_score(state.board)
            return score,action

        # alpha-beta pruning
        best_action = None
        if state.is_max_turn():
            bound = -np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth-1, alpha, beta)

                if utility > bound: bound, best_action = utility, action
                if bound >= beta: break
                alpha = max(alpha, bound)

        else:
            bound = +np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax(child, max_depth-1, alpha, beta)

                if utility < bound: bound, best_action = utility, action
                if bound <= alpha: break
                beta = min(beta, bound)

        return bound, best_action


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
            score, action = nn.minimax(state, self.max_depth)
            return action
        else:
            nn.train_nn(final_score)
            return 0







