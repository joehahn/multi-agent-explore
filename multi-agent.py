#!/usr/bin/env python

#multi-agent.py
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
    environment = {'rn_seed':rn_seed, 'max_moves':max_moves, 'actions':actions,
        'N_buckets':N_buckets, 'N_agents':N_agents}
    return environment

def initialize_agents(environment):
    N_buckets = environment['N_buckets']
    N_agents = environment['N_agents']
    all_buckets = range(N_buckets)
    agent_buckets = np.random.choice(all_buckets, size=N_agents, replace=False)
    agent_buckets.sort()
    agents = [{'bucket':bucket} for bucket in agent_buckets]
    return agents

#def initialize_state(environment):
#    N_buckets = environment['N_buckets']
#    N_agents = environment['N_agents']
#    agents = initialize_agents(N_agents, N_buckets)
#    state = {'agents':agents}

def get_reward(agents):
    reward = 0.0
    for agent in agents:
        reward += agent['bucket']
    return reward

#check game state = running, or too many moves
def get_game_state(N_turn, environment):
    game_state = 'running'
    max_moves = environment['max_moves']
    if (N_turn >= max_moves):
        game_state = 'max_moves'
    return game_state

#generate state vector from list of agents
def agents2state_vector(agents, environment):
    N_buckets = environment['N_buckets']
    v = np.zeros((1,N_buckets), dtype=float)
    for agent in agents:
        v[0, agent['bucket']] = 1.0
    return v

#convert state_vector to list of agents
def state_vector2agents(state_vector):
    agents = []
    for idx in range(state_vector.shape[1]):
        if (state_vector[0, idx] > 0.5):
            agents += [{'bucket':idx}]
    return agents

#play game per strategy
def play_game(environment, strategy, model=None):
    agents = initialize_agents(environment)
    max_moves = environment['max_moves']
    memories = deque(maxlen=max_moves+1)
    N_turn = 0
    game_state = get_game_state(N_turn, environment)
    while (game_state == 'running'):
        if (strategy == 'random'):
             agents_next = initialize_agents(environment)
        reward = get_reward(agents_next)
        game_state = get_game_state(N_turn, environment)
        memory = (agents, reward, agents_next, game_state)
        memories.append(memory)
        print N_turn, reward
        N_turn += 1
        agents = agents_next
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
    for N_game in range(N_training_games):
        agents = initialize_agents(environment)
        state_vector = get_state_vector(agents, environment)
        N_inputs = state_vector.shape[1]
        experience_replay = True
        N_turn = 0
        if (N_game > N_games/10):
            #agent executes random actions for first 10% games, after which epsilon ramps down to 0.1
            if (epsilon > 0.1):
                epsilon -= 1.0/(N_games/2)
        game_state = get_game_state(N_turn, environment)
        while (game_state == 'running'):
            agents_copy = copy.deepcopy(agents)
            for agent in agents:
                state_vector = get_state_vector(agents, environment)
                #predict this turn's possible rewards Q
                Q = model.predict(state_vector, batch_size=1)
                #choose best action
                if (np.random.random() < epsilon):
                    #move agent to random bucket...note that agents is updated
                    agent['bucket'] = np.random.choice(environment['actions'])
                else:
                    #agent moves to best bucket...note that agents is updated
                    agent['bucket'] = np.argmax(Q)
                #update state_vector
                state_vector_next = get_state_vector(agents, environment)
                #predict next turn's possible rewards
                Q_next = model.predict(state_vector_next, batch_size=1)
                max_Q_next = np.max(Q_next)
                reward = get_reward(agents)
                game_state = get_game_state(N_turn, environment)
                #add next turn's discounted reward to this turn's predicted reward
                Q[0, agent['bucket']] = reward
                if (game_state == 'running'):
                    Q[0, agent['bucket']] += gamma*max_Q_next
            agents_next = copy.deepcopy(agents)
            if (experience_replay):
                #train model on randomly selected past experiences
                memory = (agents_copy, reward, agents_next, game_state)
                memories.append(memory)
                memories_sub = random.sample(memories, batch_size)
                agentz = [m[0] for m in memories_sub]
                rewardz = [m[1] for m in memories_sub]
                agentz_next = [m[2] for m in memories_sub]
                game_statez = [m[3] for m in memories_sub]
                state_vectorz_list = [get_state_vector(a, environment) for a in agentz]
                state_vectorz = np.array(state_vectorz_list).reshape(batch_size, N_inputs)
                Qz = model.predict(state_vectorz, batch_size=batch_size)
                state_vectorz_next_list = [get_state_vector(a, environment) for a in agentz_next]
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
            state = state_next
            N_moves += 1
    return model


#initialize
rn_seed = 12
N_agents = 3
N_buckets = 5
max_moves = 100
environment = initialize_environment(rn_seed, max_moves, N_buckets, N_agents)
print 'environment = ', environment

#build model
agents = initialize_agents(environment)
state_vector = get_state_vector(agents, environment)
N_inputs = state_vector.shape[1]
N_outputs = N_agents
N_neurons = N_inputs*N_outputs
model = build_model(N_inputs, N_neurons, N_outputs)

#play random game
strategy = 'random'
memories = play_game(environment, strategy)
for m in memories:
    print m

N_training_games = 10
gamma = 0.85                              #discount for future rewards
batch_size = 100                          #number of memories used during experience-replay
print 'batch_size = ', batch_size
debug = True                              #set debug=True to see stats about each game's final turn
print 'training model',
trained_model = train(environment, model, N_training_games, max_distance, gamma, memories, batch_size, debug=debug)
print '\ntraining done.'


