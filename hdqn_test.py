#!/usr/bin/env python
"""Run SDP Environment with hDQN."""

import argparse
import os
import random

import numpy as np
import objectives
import policy
import hdqn
import sdp


from objectives import mean_huber_loss
from hdqn import HQNAgent
from sdp import StochasticDecisionProcess


# debuging parameters 
#agent parameters
#GAMMA = 0.99
#ALPHA = 25e-5 #finished
#NUM_EPISODES=10 #finished
#REPLAY_BUFFER_SIZE = 100 #not specified
#BATCH_SIZE = 2 #not specified
#TARGET_UPDATE_FREQ = 1#not specified
#NUM_BURNIN = 2 #not specified
#TRAIN_FREQ=2 #not specified
#ANNEAL_NUM_STEPS = 5 #finished for the metacontroller, check what adaptive anneal means for the controller
#EVAL_FREQ=1



#environment parameters
#INITIAL_STATE=1
#TERMINAL_STATE=0
#NUM_STATES=6
#BONUS_STATE=5
#ACTION_SUCCESS_PROB=0.5

#Evaluation Parameters
#EVAL_NUM_EPISODES=1 #finished


#agent parameters
GAMMA = 0.99
ALPHA = 25e-5 #finished
#NUM_EPISODES=100000 #finished 
NUM_EPISODES=30000 #for linear model
#NUM_EPISODES=60000 #for deep model
REPLAY_BUFFER_SIZE = 1000000 #not specified
BATCH_SIZE = 64 #not specified
TARGET_UPDATE_FREQ = 1000#not specified
NUM_BURNIN = 30000 #not specified for linear model
#NUM_BURNIN=60000 #for deep model
#NUM_BURNIN = 100000 #not specified
TRAIN_FREQ=1 #not specified
ANNEAL_NUM_STEPS = 50000 #finished for the metacontroller, check what adaptive anneal means for the controller
EVAL_FREQ=1000



#environment parameters
INITIAL_STATE=1
TERMINAL_STATE=0
NUM_STATES=6
BONUS_STATE=5
ACTION_SUCCESS_PROB=0.5
RESET_FIT_LOGS=1000 #for counting the number of visits per state

#Evaluation Parameters
EVAL_NUM_EPISODES=10 #finished

def main():  # noqa: D103

    hdqn_agent=HQNAgent(
                 controller_network_type='Linear',
                 metacontroller_network_type='Linear',
                 #controller_network_type='Deep',
                 #metacontroller_network_type='Deep',
                 state_shape=(1,NUM_STATES),
                 goal_shape=(1,NUM_STATES),
                 num_actions=2,
                 num_goals=NUM_STATES,
                 controller_burnin_policy=policy.UniformRandomPolicy(2),
                 metacontroller_burnin_policy=policy.UniformRandomPolicy(NUM_STATES),
                 controller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(2,1.0,0.1,ANNEAL_NUM_STEPS),
                 metacontroller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_STATES,1.0,0.1,ANNEAL_NUM_STEPS),
                 controller_testing_policy=policy.GreedyEpsilonPolicy(0.05,2),
                 metacontroller_testing_policy=policy.GreedyEpsilonPolicy(0.05,2),
                 controller_gamma=GAMMA,
                 metacontroller_gamma=GAMMA,
                 controller_alpha=ALPHA,
                 metacontroller_alpha=ALPHA,
                 controller_target_update_freq=TARGET_UPDATE_FREQ,
                 metacontroller_target_update_freq=TARGET_UPDATE_FREQ,
                 controller_batch_size=BATCH_SIZE,
                 metacontroller_batch_size=BATCH_SIZE,
                 controller_optimizer='Adam',
                 metacontroller_optimizer='Adam',
                 controller_loss_function=mean_huber_loss,
                 metacontroller_loss_function=mean_huber_loss,
                 eval_freq=EVAL_FREQ,
                 controller_num_burnin=NUM_BURNIN,
                 metacontroller_num_burnin=NUM_BURNIN
                 )
    
    #training environment
    env = StochasticDecisionProcess(initial_state=INITIAL_STATE,\
              terminal_state=TERMINAL_STATE,\
              num_states=NUM_STATES,\
              bonus_state=BONUS_STATE,\
              act_suc_prob=ACTION_SUCCESS_PROB
             )
    #evaluation environment
    eval_env = StochasticDecisionProcess(initial_state=INITIAL_STATE,\
              terminal_state=TERMINAL_STATE,\
              num_states=NUM_STATES,\
              bonus_state=BONUS_STATE,\
              act_suc_prob=ACTION_SUCCESS_PROB
              )
              
    hdqn_agent.fit(env, eval_env, NUM_EPISODES,EVAL_NUM_EPISODES,1000)

if __name__ == '__main__':
    main()
