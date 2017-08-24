#!/usr/bin/env python
"""Environment of Stochastic Decision Process."""
import random
import numpy as np

INITIAL_STATE=1
TERMINAL_STATE=0
NUM_STATES=6
BONUS_STATE=5
ACTION_SUCCESS_PROB=0.5

class StochasticDecisionProcess:
    
    def __init__(
            self,
            initial_state=INITIAL_STATE,\
            terminal_state=TERMINAL_STATE,\
            num_states=NUM_STATES,\
            bonus_state=BONUS_STATE,\
            act_suc_prob=ACTION_SUCCESS_PROB
            ):
        
        self.initial_state=initial_state
        self.terminal_state=terminal_state
        self.num_states=num_states
        self.bonus_state=bonus_state
        self.act_suc_prob=act_suc_prob
        
        self.visited = False
        self.current_state = initial_state
        
        self.visit_counts=np.zeros(self.num_states)
    
    def reset(self):
        self.visited = False
        self.current_state = self.initial_state
        return self.current_state
    
    
    def step(self, action):
        
        #apply action
        
        #"right" selected
        if action == 1:
            if random.random() < self.act_suc_prob and self.current_state <self.num_states:
                if self.current_state !=self.num_states-1:
                    self.current_state += 1
                    self.visit_counts[self.current_state]+=1
                    #print'INSIDE THE ENVIRONMENT'
                    #print self.visit_counts
                else:
                    self.current_state -= 1
                    self.visit_counts[self.current_state]-=1
                    
            else:
                if self.current_state !=0:
                    self.current_state -= 1
                    self.visit_counts[self.current_state]+=1
                    #print'INSIDE THE ENVIRONMENT'
                    #print self.visit_counts
        #"left" selected
        else:
            if self.current_state !=0:
                self.current_state -= 1
                self.visit_counts[self.current_state]+=1
                #print'INSIDE THE ENVIRONMENT'
                #print self.visit_counts
                
        #check if the 'bonus" state has been reached
        if self.current_state == self.bonus_state:
            self.visited = True

        #check if the state is terminal
        if self.current_state == self.terminal_state:
            if self.visited:
                #print 'BINGO'
                return self.current_state, 1.00, True
            else:
                return self.current_state, 1.00/100.00, True
        else:
            return self.current_state, 0.0, False
        
    def reset_fit_logs(self):
            self.visit_counts=np.zeros(self.num_states)
        
        


