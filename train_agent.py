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


### HYPERPARAMETERS

# learning rate of the optimizer, here adaptive
learning_rate = 0.2

# learning rate of the Q function [0 = no learning, 1 = only consider rewards]
alpha = 0.5

# Q learning discount factor [0 = only weight current state, 1 = weight future reward only]
gamma = 0.8

# E-greedy exploration [0 = no exploration, 1 = strict exploration]
epsilon = 1
epsilon_min = 0.2
epsilon_decay = 0.9999

training_start = 200  # total number of steps after which network training starts
training_interval = 5    # number of steps between subsequent training steps

batch_processing = True
use_convolutions = False

# debug output
action_output = False
dimensions_output = False
q_output = False
stats_output = True
state_txt_output = False


### helper functions
def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# reformat data for network input
def reshapeInputData(input_batch, no_batches):
    if use_convolutions:
        input_batch = input_batch.reshape((no_batches, historyLength, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz))
        # reformat input data if convolutions are used (consistent with visual map)
        input_batch = np.rot90(input_batch, axes=(1, 2))
        input_batch = np.rot90(input_batch, axes=(2, 3))
        # rotate mapview 180 degree
        input_batch = np.rot90(input_batch, axes=(1, 2))
        input_batch = np.rot90(input_batch, axes=(1, 2))
    else:
        input_batch = input_batch.reshape((no_batches, historyLength * opt.pob_siz * opt.cub_siz * opt.pob_siz * opt.cub_siz))
    return input_batch
    

# export state with history to file for debugging
def saveStateAsTxt(state_array):
    state_array[state_array > 200] = 4
    state_array[state_array > 100] = 3
    state_array[state_array >  50] = 2
    state_array[state_array >  10] = 1

    state_array = reshapeInputData(state_array, 1)
    
    # append history, most recent state is last
    string = ''
    for i in range(historyLength):

        # consistent with visualization
        string += str(np.array(state_array[0,:,:,i], dtype=np.uint8)) + '\n\n'

    with open('state_export.txt', 'w') as textfile:
        print(string, file=textfile)


### INITIALIZATION
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
num_classes = 5


### NETWORK DEFINITION
print('\n... setting up Qnet ...')
print('input shape:\t{}*{}*{}\n'.format(opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength))

qnet = Sequential()
# qnet.add(Conv2D(64, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=(opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, historyLength)))
# qnet.add(Flatten())
qnet.add(Dense(128, activation='relu', input_shape=(opt.pob_siz * opt.cub_siz * opt.pob_siz * opt.cub_siz * historyLength,)))
qnet.add(Dense(256, activation='relu'))
qnet.add(Dense(256, activation='relu'))
qnet.add(Dense(num_classes, activation='softmax'))

qnet.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

configstr = '... hyperparams ...\ngamma:\t\t{}\nepsilon:\t{}\nepsilon min:\t{}\nepsilon decay:\t{}\nbatch processing:\t{}\n'.format(
    gamma, epsilon, epsilon_min, epsilon_decay, batch_processing)
print(configstr)
print('> network compiled')


### 
### TRAINING
###

print('\n... training ...')

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)


# some statistics
loss = 0
stats = np.zeros((opt.eval_nepisodes, 4))
step = 0
max_step = 0
max_last = 0

### go over some episodes, apply learning
for e in range(opt.eval_nepisodes):

    ### reset the environment
    # reset the game
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    # and reset the history
    state_with_history[:] = 0
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)

    ### go until goal reached
    while True:
        ### goal check or max_step
        if state.terminal or (step - max_last) >= opt.early_stop:
            max_step = step
            break
        
        # increase the counter in each iteration globally
        step += 1

        ### take action here
            # movement correspondencies
            # 0 : stay
            # 1 : up
            # 2 : down
            # 3 : left
            # 4 : right
        if np.random.rand() <= epsilon:
            # this just gets a random action
            action = randrange(opt.act_num)
            action_onehot = trans.one_hot_action(action)
            if action_output: print('random action:\t', action)
        else:
            # make a Qnet prediction here based on the current state s
            input_state = reshapeInputData(state_with_history, 1)
            q_actions = qnet.predict(input_state, verbose=0)
            # get the action which corresponds to the max Q value (action = argmax)
            action = np.argmax(q_actions)
            action_onehot = trans.one_hot_action(action)
            if action_output: print('Qnet action:\t', action)

        next_state = sim.step(action)

        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state


        # output current state to txt
        if state_txt_output: saveStateAsTxt(state_with_history)

        ### print map here
        if opt.disp_on:
            # plt.ion()
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)
            plt.draw()


        ### training step: fit Qnet to Q values
        if (step > training_start) and (step % training_interval == 0):
            # state_batch           : gives current state
            # next_state_batch      : gives next state either predicted by Qnet or at random
            # reward_batch          : gives reward of next_state
            # terminal_batch        : gives whether next_state is goalstate
            state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

            loss = 0
            if not batch_processing:
                # iterate over the individual batches
                for i in range(opt.minibatch_size):
                    state_i = state_batch[i]
                    actions_i = action_batch[i]
                    next_state_i = next_state_batch[i]
                    reward_i = reward_batch[i]
                    terminal_i = terminal_batch[i]

                    # formatting
                    state_i = reshapeInputData(state_i, 1)
                    next_state_i = reshapeInputData(next_state_i, 1)
                    action_index = np.argmax(actions_i)

                    # get the q values for [s, s']
                    q_actual = qnet.predict(state_i)
                    q_next = qnet.predict(next_state_i)

                    # calculate new q value
                    q_target = reward_i + ((1 - terminal_i[0]) * gamma * np.amax(q_next))
                    q_updated = np.copy(q_actual)  # IMPORTANT: otherwise only reference, e.g. both would be identical
                    q_updated[0, action_index] = (1 - alpha) * q_updated[0, action_index] + alpha * q_target

                    if q_output: print('q actual:\t{}\nq target:\t{}'.format(q_actual[0,:], q_updated[0,:]))

                    # fit qnet to new q value
                    qnet.fit(state_i, q_updated, epochs=1, verbose=0)

                    # compute loss for stats
                    loss += np.sum(np.power(np.abs(q_updated - q_actual), 2), 1)[0]


            else:
                # argmax gives index, amax gives max value along axis or value
                action_axis = 1

                # reshape the batches as input for Qnet
                state_batch = reshapeInputData(state_batch, opt.minibatch_size)
                next_state_batch = reshapeInputData(next_state_batch, opt.minibatch_size)
                action_index_batch = np.argmax(action_batch, action_axis)
                
                # get the current Q values the network predicts for the active state
                q_current_batch = qnet.predict(state_batch, batch_size = opt.minibatch_size)
                # predict the Q values for the actions a' from the next_state s' 
                q_next_batch = qnet.predict(next_state_batch, batch_size = opt.minibatch_size)
                
                # implementing the Q function below apply the reshape trick: (32,) -> (32, 1)
                q_star = np.reshape(np.amax(q_next_batch, action_axis),(opt.minibatch_size, 1))
                q_target_batch = (reward_batch) + (1-(terminal_batch)) * gamma * np.copy(q_star)
                
                # update the Q(s,a) value for the action a taken from state s to s'  /  gradually add future rewards using the Q function
                q_updated_batch = np.copy(q_current_batch)
                q_updated_batch[np.arange(opt.minibatch_size), action_index_batch] = (1 - alpha) \
                    * q_updated_batch[np.arange(opt.minibatch_size), action_index_batch] + alpha * np.copy(q_target_batch[:,0])  # dim trick here as well
                
                if q_output: print('q actual:\n{}\nq target:\n{}'.format(q_current_batch, q_updated_batch))

                ### fit the Qnet to the evolved Q values
                # this implicitly computes the squared error loss of (target - prediction)**2 by fitting the network output 
                # of qnet(state_batch) -> Q values to those Q values with future rewards taken into account
                # the smaller the loss, the closer the Qnet predictions get to the true Q values
                qnet.train_on_batch(state_batch, q_updated_batch)
                
                loss = np.mean(np.sum(np.power(q_updated_batch - q_current_batch, 2), action_axis))

            # debug
            if dimensions_output:
                print('\n...shapes...')
                print('state shape:\t\t', state_batch.shape)
                print('next_state shape:\t', next_state_batch.shape)
                print('action shape:\t\t', action_batch.shape)
                print('Q prediction shape:\t', q_next_batch.shape)
                print('Q target shape:\t\t', q_target_batch.shape)
                print('Q shape:\t\t', q_current_batch.shape)
                print('action index:\t\t', action_index_batch)
                print('Q current shape:\t', q_current_batch[np.arange(opt.minibatch_size), action_index_batch].shape)

            # update epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    steps = max_step - max_last
    max_last = max_step

    print('episode: {:>4}/{} | loss: {:>8.4f} | steps: {:>4} | epsilon {:>3.3f}'.format(e, opt.eval_nepisodes, loss, steps, epsilon))
    stats[e,:] = [e, loss, steps, epsilon]


    ### finished here


### save model and stats
print('... training ended ...\n')
qnet.save('qnet.h5')
print('> qnet saved')
if stats_output:
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = 'stats' + timestamp + '.txt'
    configstr += '\n' + qnet.to_yaml()
    np.savetxt(filename, stats, fmt='%1.3f', footer=configstr)
    print('> statistics exported')
print('\n...finished!')