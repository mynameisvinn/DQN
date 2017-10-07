from collections import deque
import random

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD

class Actor(object):
    """
    actor learns to predict a set of "deltas", which represent 
    the difference in value between a new state and the current 
    state. the actor will choose the action that will result in 
    the largest delta.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.brain = self._build_model()
        self.epsilon = 1
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.n_batch = 32
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(164, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # output is action space, not a single value prediction
        a_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=a_optimizer)
        return model
    
    def remember(self, experience):
        self.memory.append(experience)

    def predict(self, state):
        return self.brain.predict(state)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict(state)[0])
    
    def replay(self):
        """
        actor learns the deltas between current state and next state.
        
        for every current state, the actor predicts value deltas for each
        possible action.
        
        the actor no longer moves to the highest value state, but takes the
        action with the highest value difference. 
        """
        if len(self.memory) > self.n_batch:
            X_train = []
            y_train = []
        
            minibatch = random.sample(self.memory, self.n_batch)
            for memory in minibatch:
                m_orig_state, m_action, m_value = memory
                old_qval = self.brain.predict(m_orig_state.reshape(1, self.state_size,) )
                y = np.zeros((1, self.action_size))
                y[:] = old_qval[:]
                y[0][m_action] = m_value
                X_train.append(m_orig_state.reshape((self.state_size,)))
                y_train.append(y.reshape((self.action_size,)))
                
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            self.brain.fit(X_train, y_train, batch_size=self.n_batch, epochs=1, verbose=0)
        else:
            pass
        
        # decrease epsilon over time
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay