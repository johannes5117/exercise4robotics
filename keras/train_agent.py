# basic imports
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import time

# keras imports
import keras
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

### hyperparameters
# learning rate of the optimizer, here adaptive
# learning_rate = 0.2

# Q learning discount factor [0 = only weight current state, 1 = weight future reward only]
gamma = 0.99

# E-greedy exploration [0 = no exploration, 1 = strict exploration]
epsilon = 1
epsilon_min = 0.2
epsilon_decay = 0.99


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    loss = tf.reduce_sum(tf.square(selected_q - target_q))    
    return loss

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


historyLength = opt.hist_len
num_classes = 5  # 0 = no action / 1 = up / 2 = down / 3 = left / 4 = right

### define network here
print('... setting up Qnet ...')
print('input shape:\t', opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength)
qnet = Sequential()
qnet.add(Conv2D(32, kernel_size=(5, 5),
                activation='relu',
                input_shape=(opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength)))
qnet.add(Conv2D(64, (5, 5), activation='relu'))
#qnet.add(Dropout(0.25))
qnet.add(Flatten())
qnet.add(Dense(128, activation='relu'))
#qnet.add(Dropout(0.5))
qnet.add(Dense(num_classes, activation='softmax'))

qnet.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])





'''
# train the CNN
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(validation_data_x, validation_data_y, verbose=0)
'''
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # NOTE:
                    # You should prepare your network training here. I suggest to put this into a
                    # class by itself but in general what you want to do is roughly the following
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
                    # setup placeholders for states (x) actions (u) and rewards and terminal values
                    x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
                    u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
                    ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
                    xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
                    r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
                    term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

                    # get the output from your network
                    Q = my_network_forward_pass(x)
                    Qn =  my_network_forward_pass(xn)

                    # calculate the loss
                    loss = Q_loss(Q, u, Qn, ustar, r, term)

                    # setup an optimizer in tensorflow to minimize the loss
"""





# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**3
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

# for e in range(opt.eval_nepisodes):

for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:
        if state.terminal:
            print('target reached!')
        else:
            print('early stop')
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #       remember
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    if np.random.rand() <= epsilon:
        # this just gets a random action
        action = randrange(opt.act_num)
        action_onehot = trans.one_hot_action(action)
        #print('random action:\t', action)
    else:
        # make a Qnet prediction here based on the current state <x>
        input_state = np.reshape(state_with_history, (1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength))
        q_actions = qnet.predict(input_state, batch_size=None, verbose=0)
        # get the action which corresponds to the max Q value (action = argmax)
        action = np.argmax(q_actions)
        action_onehot = trans.one_hot_action(action)
        #print('Qnet action:\t', action)

    next_state = sim.step(action)

    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    
    
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    # state_batch           : gives current state
    # next_state_batch      : gives next state either predicted by Qnet or at random
    # reward_batch          : gives reward of next_state
    # terminal_batch        : gives whether next_state is goalstate

    # argmax gives index, amax gives max value along axis or value
    action_axis = 1

    # reshape the batches from (32, 3600) -> (32, 30, 30, 4) as input for Qnet
    state_batch = np.reshape(state_batch, (opt.minibatch_size, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength))
    next_state_batch = np.reshape(next_state_batch, (opt.minibatch_size, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength))

    # debug
    # print('state shape:\t', state_batch.shape)
    # print('next_state shape:\t', next_state_batch.shape)
    # print('action shape:\t', action_batch.shape)

    # TODO: sum up over the batches
    # predict the Q values for the actions a' from the next_state s' 
    q_next_batch = qnet.predict(next_state_batch, batch_size = opt.minibatch_size)
    #print('Q prediction shape:\n', q_next_batch.shape)
    # get the next_action
    next_action_batch = np.argmax(q_next_batch, action_axis)
    
    # implementing the Q function below
    # apply the reshape trick here: (32,) -> (32, 1)
    q_star = np.reshape(np.amax(q_next_batch, action_axis),(opt.minibatch_size, 1))
    q_target_batch = reward_batch + (1-terminal_batch) * gamma * q_star
    # print('Q target shape:\t', q_target_batch.shape)

    # get the current Q values the network predicts for the active state
    q_current_batch = qnet.predict(state_batch, batch_size = opt.minibatch_size)
    #print('Q shape:\n', q_current_batch.shape)

    action_index_batch = np.argmax(action_batch, action_axis)
    # print('action index:\n', action_index_batch)
    # remember the actual Q values before training step for loss calculation
    q_selected_batch = q_current_batch[np.arange(opt.minibatch_size), action_index_batch].reshape((opt.minibatch_size, 1))
    
    # update the Q(s,a) value for the action a taken from state s to s'  /  gradually add future rewards using the Q function
    q_current_batch[np.arange(opt.minibatch_size), action_index_batch] = q_target_batch[:,0]  # dim trick here as well
    #print('q current shape:\n', q_current_batch[np.arange(opt.minibatch_size), action_index_batch].shape)

    # fit the Qnet to the evolved Q values
    qnet.fit(state_batch, q_current_batch, epochs=1, verbose=0)

    # calculate the loss for output
    loss = np.sum(np.power(q_target_batch - q_selected_batch, 2))

    # print('loss shape:\t', loss.shape)
    # print(loss)
    print('step: {:>4}/{} | loss: {:.5f}'.format(step, steps, loss))

    ### finished here


    # TODO train me here
    # this should proceed as follows:
    # 1) pre-define variables and networks as outlined above
    # 1) here: calculate best action for next_state_batch
    # TODO:
    # action_batch_next = CALCULATE_ME
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss 
    #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))

    
    # TODO every once in a while you should test your agent here so that you can track its performance

    if opt.disp_on:
        plt.ion()
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        # time.sleep(0.1)
        plt.pause(opt.disp_interval)
        plt.draw()


# 2. perform a final test of your model and save it
# TODO
