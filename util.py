import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from gym.envs.registration import register, spec

def to_onehot(size,value):
    """1 hot encoding for observed state"""
    return np.eye(size)[value]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # mse bc we predict reward values, not actions
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            
            # scenario 1: game isn't complete - retrain model
            # NN is a q function
            if not done:
                # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions from next state)]
                # reward : whats my reward, if i take the proposed action?
                # max(q(next, all actions)) : whats my best move from my next state? (2 hops away from curr state)
                # 2nd term: all possible actions - this is 2 steps into the future
                # check out "demystifying deep reinforcement learning" blog
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
                
            # what action probabilities would have we predicted from the curr state?
            target_f = self.brain.predict(state)
            
            # we know the actual outcome, so update target probabilities
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        
        # decrease epsilon over time
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay