"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np
#import attr

DEBUG = 0

class Policy:
    """Base class representing an MDP policy.

    Policies are used by the agent to choose actions.

    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select(self):
        """Used by agents to select actions.

        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.

    This is provided as a reference on how to use the policy class.

    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.

    Raises
    ------
    ValueError:
      If num_actions <= 0
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select(self):
        """Return a random action index.

        This policy cannot contain others (as they would just be ignored).

        Returns
        -------
        int:
          Action index in range [0, num_actions)
        """
        action = np.random.randint(0, self.num_actions)
	if DEBUG:
		print 'In uniform policy: action {0}'.format(action)
	return action

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select(self, q_values):  # noqa: D102
        action = np.argmax(q_values)
        if DEBUG:
            print 'Q-values'
            print q_values
            print 'In greedy policy: action {0}'.format(action)
	return action


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.

    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.

    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    def __init__(self, epsilon,num_actions):
        self.epsilon = epsilon
        self.num_actions=num_actions

    def select(self, q_values):
        """Run Greedy-Epsilon for the given Q-values.

        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.

        Returns
        -------
        int:
          The action index chosen.
        """
        rand_num = np.random.rand()
        if rand_num < self.epsilon:
            action =  np.random.randint(self.num_actions)
            if DEBUG:
                print 'GreedyEpsilonPolicy: select randomly'
            # print 'random'
        else:
            action = np.argmax(q_values)
            #print 'greedy'
            #print q_values
	
        if DEBUG:
            print 'GreedyEpsilon: epsilon {0} action {1}'.format(self.epsilon, action)
        
        return action


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self,num_actions,start_value, end_value,
                 num_steps):  # noqa: D102
        self.policy=GreedyEpsilonPolicy(start_value,num_actions)
        self.end_value = end_value
        self.start_value = start_value
        self.num_steps = num_steps

    def select(self, q_values, num_update):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        Any:
          Selected action.
        """
	# Linear annealed epsilon=x: f(x) = ax + b.
        a = -float(self.start_value - self.end_value) / float(self.num_steps)
        b = float(self.start_value)
        self.policy.epsilon = max(self.end_value, a * float(num_update) + b)
        #print 'Annealed Epsilon {0}'.format(self.policy.epsilon)
        action = self.policy.select(q_values)
        if DEBUG:
            print 'LinearDecay: epsilon {0} action {1}'.format(self.policy.epsilon,action)
        return action

    def reset(self):
	"""Start the decay over at the start value."""
        self.policy.epsilon=start_value
