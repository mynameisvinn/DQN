import random
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class Critic(object):
    """
    critics assign values to each possible state. 
    
    to train a critic on a current state, take a random action.
    then, have the critic predict the value ("target value") of 
    the new state. the training set is then: 
    
    {X: curr_state, y: target_value}
    
    notice how the critic's learning is completely action-agnostic. 
    it does not know about actions in its training set, or which
    action is responsible for the target value.
    
    attributes
    ----------
    state_size : int
    action_size : int
    memory : list
    brain : keras model object
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.brain = self._build_model()
    
    def _build_model(self):
        """
        construct model to predict value for a given state. a state value, by itself,
        does not mean anything; it is the relative value that matters to the actor.

        the critic will suggest values to the actor; and it is up to the actor to select
        the correct action
        """
        model = Sequential()
        model.add(Dense(164, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(1, activation='linear'))
        c_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=c_optimizer)
        return model
    
    def predict(self, state):
        return self.brain.predict(state)
    
    def remember(self, experience):
        self.memory.append(experience)
        
    def replay(self, n_batch):
        """
        online model training through experience replay. by training the model with
        randomly selected memories, we avoid model overfitting.
        """
        if len(self.memory) > n_batch:
            
            # prepare dataset for batch training
            minibatch = random.sample(self.memory, n_batch)
            X_train = []
            y_train = []
    
            for batch in minibatch:
                m_state, m_value = batch
                X_train.append(m_state.reshape((self.state_size,)))
                y_train.append(np.array(m_value).reshape((1,)))
                
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            self.brain.fit(X_train, y_train, batch_size=n_batch, epochs=1, verbose=0)
        else:
            pass
    
    def plot_value(self, initial_state):
        obs_sqr = int(math.sqrt(self.state_size))
        np_w_cri_r = np.zeros((obs_sqr, obs_sqr))
        working_state = initial_state.copy()
        for x in range(0, obs_sqr):
            for y in range(0, obs_sqr):
                my_state = working_state.copy()
                
                my_state[x,y] = 1  # Place the player at a given X/Y location.

                # And now have the critic model predict the state value
                # with the player in that location.
                value = self.brain.predict(my_state.reshape(1, self.state_size))
                np_w_cri_r[x,y] = value
        np_w_cri_r.shape
        plt.pcolor(np_w_cri_r)
        plt.title("Value Network")
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.draw()