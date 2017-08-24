from collections import namedtuple
import random
import numpy as np


Experience = namedtuple('Experience', 'state, action, reward, next_state, terminal')
DEBUG=0

class RingBuffer(object):
    
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]
    

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]
    def __str__(self):
        return 'start pointer {0} length {1} max length {2}'.format(self.start, self.length, self.maxlen)
        
    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

class ReplayMemory:
    """Interface for replay memories.
       The memory is not compressed in the sense that:
       i) it stores both the current and next state of an experience sample
       ii) the state and next_state will contain more than one experience samples in case of
           a history length greather than one
    """
    
    def __init__(self, max_size):
        """Setup memory.
        """
        
        self.max_size=max_size
        self.actions = RingBuffer(max_size)
        self.rewards = RingBuffer(max_size)
        self.terminal = RingBuffer(max_size)
        self.states = RingBuffer(max_size)
        self.next_states =  RingBuffer(max_size)
        
        pass

    @property
    def nb_entries(self):
        return self.states.length

    def append(self, state, action, reward,next_state,is_terminal):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminal.append(is_terminal)
    
        if DEBUG:
            print self.states
            print self.next_states
            print self.actions
            print self.rewards
            print self.terminal


    def sample(self, batch_size, indices=None):
        
        if indices is None:
        #if indices is None draw random samples, each index refers to the last frame of next_state
            indices = random.sample(xrange(0, self.nb_entries), batch_size)

        assert 0<=np.min(indices)                         
        assert np.max(indices)< self.nb_entries
        assert len(indices) == batch_size
            
        samples = []

        for idx in indices:
            samples.append(Experience(state=self.states[idx], action=self.actions[idx], reward=self.rewards[idx],next_state=self.next_states[idx], terminal=self.terminal[idx]))
            
        assert len(samples) == batch_size
        return samples
    

    def clear(self):
        del self.states
        del self.next_states
        del self.actions
        del self.rewards
        del self.terminals

        self.close()

#class CompressedReplayMemory:
