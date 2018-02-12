#!/usr/bin/env python

#multi_agent.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#10 February 2018
#
#helper functions for an experiment with Q-learning on system having multiple agents 

#imports
import numpy as np
import random
import copy
from collections import deque

#initialize the environment = dict containing all constants that describe the system
def initialize_environment(rn_seed, max_moves, N_buckets, N_agents):
    random.seed(rn_seed)
    actions = range(N_buckets)
    acts = ['move to ' + str(action) for action in actions]
    environment = {'rn_seed':rn_seed, 'max_moves':max_moves, 'actions':actions,
        'acts':acts, 'N_buckets':N_buckets, 'N_agents':N_agents}
    return environment

def initialize_state(environment):
    N_buckets = environment['N_buckets']
    N_agents = environment['N_agents']
    all_buckets = range(N_buckets)
    agent_buckets = np.random.choice(all_buckets, size=N_agents, replace=False)
    agent_buckets.sort()
    agents = [{'bucket':bucket} for bucket in agent_buckets]
    state = {'agents':agents}
    state['next_agent'] = np.random.randint(0, N_agents)
    return state

def get_reward(state):
    reward = 0.0
    for agent in state['agents']:
        reward += agent['bucket']
    return reward

#move agent
def update_state(state, environment, action):
    state_next = copy.deepcopy(state)
    agents = state_next['agents']
    agent_idx = state_next['next_agent']
    agent = agents[agent_idx]
    agent['bucket'] = action
    agent_idx += 1
    if (agent_idx >= N_agents):
        agent_idx = 0
    state_next['next_agent'] = agent_idx
    return state_next

#check game state = running, or too many moves
def get_game_state(N_turn, environment):
    game_state = 'running'
    max_moves = environment['max_moves']
    if (N_turn >= max_moves):
        game_state = 'max_moves'
    return game_state

#convert state into a numpy array agent locations
def state2vector(state, environment):
    N_buckets = environment['N_buckets']
    v = np.zeros((1,N_buckets), dtype=float)
    for agent in state['agents']:
        v[0, agent['bucket']] = 1.0
    return v

#play game per strategy
def play_game(environment, strategy, model=None):
    state = initialize_state(environment)
    max_moves = environment['max_moves']
    N_buckets = environment['N_buckets']
    memories = deque(maxlen=max_moves+1)
    N_turn = 0
    game_state = get_game_state(N_turn, environment)
    while (game_state == 'running'):
        if (strategy == 'random'):
            action = np.random.choice(environment['actions'])
        if (strategy == 'smart'):
            state_vector = state2vector(state, environment)
            Q = model.predict(state_vector, batch_size=1)
            action = np.argmax(Q)
        state_next = update_state(state, environment, action)
        reward = get_reward(state_next)
        game_state = get_game_state(N_turn, environment)
        memory = (state, action, reward, state_next, game_state)
        memories.append(memory)
        N_turn += 1
        state = copy.deepcopy(state_next)
    return memories   

#build neural network
def build_model(N_inputs, N_neurons, N_outputs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import RMSprop
    model = Sequential()
    model.add(Dense(N_neurons, input_shape=(N_inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(N_neurons))
    model.add(Activation('relu'))
    model.add(Dense(N_outputs))
    model.add(Activation('linear'))
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

#train model via a Q-learning algorithm that was adapted from
#source code that Outlace published at http://outlace.com/rlpart3.html
def train(environment, model, N_games, gamma, memories, batch_size, debug=False):
    epsilon = 1.0
    for N_game in range(N_games):
        state = initialize_state(environment)
        state_vector = state2vector(state, environment)
        N_inputs = state_vector.shape[1]
        experience_replay = True
        N_turn = 1
        if (N_game > N_games/10):
            #agent executes random actions for first 10% games, after which epsilon ramps down to 0.1
            if (epsilon > 0.1):
                epsilon -= 1.0/(N_games/2)
        game_state = get_game_state(N_turn, environment)
        while (game_state == 'running'):
            state_vector = state2vector(state, environment)
            #predict this turn's possible rewards Q
            Q = model.predict(state_vector, batch_size=1)
            #choose best action
            if (np.random.random() < epsilon):
                #choose random action
                action = np.random.choice(environment['actions'])
            else:
                #choose best action
                action = np.argmax(Q)
            #get next state
            state_next = update_state(state, environment, action)
            state_vector_next = state2vector(state_next, environment)
            #predict next turn's possible rewards
            Q_next = model.predict(state_vector_next, batch_size=1)
            max_Q_next = np.max(Q_next)
            reward = get_reward(state_next)
            game_state = get_game_state(N_turn, environment)
            #add next turn's discounted reward to this turn's predicted reward
            Q[0, action] = reward
            if (game_state == 'running'):
                Q[0, action] += gamma*max_Q_next
            else:
                if (debug):
                    print '======================='
                    print 'game number = ', N_game
                    print 'turn number = ', N_turn
                    final_act = 'agent ' + str(state['next_agent']) + ' ' + environment['acts'][action]
                    print 'final act = ', final_act
                    print 'final reward = ', reward
                    print 'epsilon = ', epsilon
                    print 'game_state = ', game_state
                else:
                    print '.',
            if (experience_replay):
                #train model on randomly selected past experiences
                memories.append((state, action, reward, state_next, game_state))
                memories_sub = random.sample(memories, batch_size)
                statez = [m[0] for m in memories_sub]
                actionz = [m[1] for m in memories_sub]
                rewardz = [m[2] for m in memories_sub]
                statez_next = [m[3] for m in memories_sub]
                game_statez = [m[4] for m in memories_sub]
                state_vectorz_list = [state2vector(s, environment) for s in statez]
                state_vectorz = np.array(state_vectorz_list).reshape(batch_size, N_inputs)
                Qz = model.predict(state_vectorz, batch_size=batch_size)
                state_vectorz_next_list = [state2vector(s, environment) for s in statez_next]
                state_vectorz_next = np.array(state_vectorz_next_list).reshape(batch_size, N_inputs)
                Qz_next = model.predict(state_vectorz_next, batch_size=batch_size)
                for idx in range(batch_size):
                    reward = rewardz[idx]
                    max_Q_next = np.max(Qz_next[idx])
                    action = actionz[idx]
                    Qz[idx, action] = reward
                    if (game_statez[idx] == 'running'):
                        Qz[idx, action] += gamma*max_Q_next
                model.fit(state_vectorz, Qz, batch_size=batch_size, epochs=1, verbose=0)
            else:
                #teach model about current action & reward
                model.fit(state_vector, Q, batch_size=1, epochs=1, verbose=0)
            state = copy.deepcopy(state_next)
            N_turn += 1
    return model

#generate memories of playing multiple random games
def make_memories(environment, strategy, N_games):
    memories_list = []
    N_memories = 0
    for N_game in range(N_games):
        memories = play_game(environment, strategy)
        memories_list += [memories]
        N_memories += len(memories)
    memories = deque(maxlen=N_memories)
    for game_memories in memories_list:
        for m in game_memories:
            memories.append(m)
    return memories

#initialize
rn_seed = 14
N_agents = 3
N_buckets = 5
max_moves = 100
environment = initialize_environment(rn_seed, max_moves, N_buckets, N_agents)
print 'environment = ', environment
state = initialize_state(environment)
print 'state = ', state
reward = get_reward(state)
print 'reward = ', reward
state_vector = state2vector(state, environment)
print 'state_vector = ', state_vector

#build model
N_inputs = N_buckets
N_outputs = N_buckets
N_neurons = N_inputs*N_outputs
model = build_model(N_inputs, N_neurons, N_outputs)

#play 100 games making random actions, and stash moves in memories queue
N_games = 100
strategy = 'random'
memories = make_memories(environment, strategy, N_games)
print 'number of memories = ', len(memories)

#train model
N_games = 100
gamma = 0.85                              #discount for future rewards
batch_size = 100                          #number of memories used during experience-replay
print 'batch_size = ', batch_size
debug = True                              #set debug=True to see stats about each game's final turn
print 'training model',
trained_model = train(environment, model, N_games, gamma, memories, batch_size, debug=debug)
print '\ntraining done.'

#play one smart game
strategy = 'smart'
memories = play_game(environment, strategy, model=trained_model)
for m in memories:
    print m

cat, bug, actions, rewards, bug_distances, bug_direction_angles, cat_direction_angles, turns = \
    memories2arrays(memories)
cumulative_rewards = rewards.cumsum()
mean_cat_bug_separations = bug_distances.mean()
memories_smart = memories
print 'strategy = ', strategy
print 'final cumulative_rewards = ', cumulative_rewards[-1]
print 'mean_cat_bug_separations = ', mean_cat_bug_separations


