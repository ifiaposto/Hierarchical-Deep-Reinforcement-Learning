"""Main DQN agent."""


import tensorflow as tf
from keras import optimizers
from keras.layers import (Activation, Dense, Flatten, concatenate, Merge, merge, Input,Multiply)
from keras.callbacks import History
from keras.models import Model
import numpy as np

import inspect


from objectives import mean_huber_loss
from memory import RingBuffer
from memory import ReplayMemory
import utils
import policy 

DEBUG=0
TENSORBOARD_PLOTS=1
MATLAB_PLOTS=1

GAMMA = 0.99
ALPHA = 25e-5 #finished
NUM_EPISODES=12000 #finished
REPLAY_BUFFER_SIZE = 1000000 #not specified, found in Ethan's code
BATCH_SIZE = 16 #not specified
TARGET_UPDATE_FREQ = 1000#not specified
NUM_BURNIN = 1000 #not specified
TRAIN_FREQ=2 #not specified
SAVE_FREQ=1# TODO: adjust it later
ANNEAL_NUM_STEPS = 50000 #finished for the metacontroller, check what adaptive anneal means for the controller
EVAL_FREQ=1,

EVAL_NUM_EPISODES=10#finished




def create_deep_model(input_shape,num_outputs,act_func='relu'):
    
    
    
    input_state = [Input(input_shape[i]) for i in range(len(input_shape))]
    input_mask = Input(shape=(num_outputs,))
            
    #does it flatten the full list?
    flat_input_state = [Flatten()(input_state[i]) for i in range(len(input_shape))]
    if len(flat_input_state)>1:
        merged_input_state=merge(flat_input_state, mode='concat')
    else:
        merged_input_state=flat_input_state[0]
    
    b1=Dense(30)(merged_input_state)
    b2 = Activation(act_func)(b1)
    
    c1=Dense(30)(b2)
    c2=Activation(act_func)(c1)
    
    d1=Dense(30)(c2)
    d2=Activation(act_func)(d1)
    
    e1=Dense(num_outputs)(d2)
    e2=Activation('relu')(e1)
    
    f = Multiply()([e2,input_mask])
    model = Model(inputs=input_state+[input_mask], outputs=[f])
                          
    return model
    


def create_linear_model(input_shape,num_outputs,act_func):  # noqa: D103
    """Create a linear network for the Q-network model.
                
    Parameters
    ----------
    input_shape: a list with the shapes of each input.
    num_outputs: the number of outputs.
                
    Returns
    -------
    a keras linear model with the output masked.
    """
            
    input_state = [Input(input_shape[i]) for i in range(len(input_shape))]
    input_mask = Input(shape=(num_outputs,))
            
    #does it flatten the full list?
    b = [Flatten()(input_state[i]) for i in range(len(input_shape))]
    if len(b)>1:
        c=Dense(num_outputs)(merge(b, mode='concat'))
    else:
        c=Dense(num_outputs)(b[0])
    e = Activation (act_func)(c)
    f = Multiply()([e,input_mask])
    model = Model(inputs=input_state+[input_mask], outputs=[f])
            
    return model

def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
            
    Parameters
    ----------
    step: int
    Training step (sets the position on x-axis of tensorboard graph.
    name: str
    Name of variable. Will be the name of the graph in tensorboard.
    value: float
    The value of the variable at this step.
    writer: tf.FileWriter
    The tensorboard FileWriter instance.
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer=writer.add_summary(summary, step)
        
class HQNAgent:
    
    #learning module, used for the controller and metacontroller
    class Module:

        def __init__(self,
                     network_type,
                     module_type,
                     state_shape,
                     num_choices,
                     burnin_policy,
                     training_policy,
                     testing_policy,
                     num_burnin=NUM_BURNIN,
                     gamma=GAMMA,
                     alpha=ALPHA,
                     optimizer='Adam',
                     loss_function = mean_huber_loss,
                     target_update_freq=TARGET_UPDATE_FREQ,
                     batch_size=BATCH_SIZE,
                     mem_size=REPLAY_BUFFER_SIZE):
            
            
            #network parameters
            self.network_type = network_type
            self.module_type = module_type
            self.state_shape = state_shape
            self.num_choices = num_choices
            
            #learning parameters
            #the discounting reward factor
            self.gamma = gamma
            #the learning rate
            self.alpha = alpha
            #the tensorflow optimizer to be used
            self.optimizer = optimizer
            #the loss function to be minimized
            self.loss_function=loss_function
            
            #after how many updates the target network will be synced
            self.target_update_freq = target_update_freq
            #umber of samples used to initialize the memory
            self.num_burnin=num_burnin
            #the batch_size for the parameters update
            self.batch_size = batch_size
            
            #agent's policies
            self.burnin_policy=burnin_policy
            self.testing_policy=testing_policy
            self.training_policy=training_policy
            
            
            #auxiliary variables for the training
            
            #number of parameter updates
            self.num_updates= 0            
            #number of interactions with the learning samples
            self.num_samples=0
            #number of episodes used for training
            self.num_train_episodes=0
    
            #modules's components
            
            #the network
            if self.network_type=='Linear':
                self.network=create_linear_model(input_shape=self.state_shape,num_outputs=self.num_choices,act_func='relu')
                self.target_network=create_linear_model(input_shape=self.state_shape,num_outputs=self.num_choices,act_func='relu')
            else:
                self.network=create_deep_model(input_shape=self.state_shape,num_outputs=self.num_choices,act_func='relu')
                self.target_network=create_deep_model(input_shape=self.state_shape,num_outputs=self.num_choices,act_func='relu')
                
            #print self.state_shape
            #self.network=self.create_linear_model(input_shape=self.state_shape,num_outputs=self.num_choices,act_func='relu')
            self.network.compile(loss=self.loss_function, optimizer = self.optimizer)
            #the target network
  
            #the replay memory
            self.memory	= ReplayMemory(mem_size)
                                                    
            #tensorboard logistics
            self.writer=tf.summary.FileWriter('./logs_'+module_type+'_'+network_type)
                                                
                                        
        def calc_q_values(self, states):
            """Given a state (or batch of states) calculate the Q-values.
            states: list of states
            choices:list with the mask of choices for each state
            Return
            ------
            Q-values for the state(s)
            """
        
            
            batch_size=len(states)
            states_batch=[np.zeros((batch_size, )+self.state_shape[i]) for i in range(len(self.state_shape))]
   
            for idx in range(batch_size):
                for in_idx in range(len(self.state_shape)):
                    assert states[idx][in_idx].shape==self.state_shape[in_idx] 
                    states_batch[in_idx][idx]=states[idx][in_idx] 
        

            q_values=self.network.predict(states_batch+[np.ones((batch_size,self.num_choices))],batch_size=1)
            assert q_values.shape==(batch_size,self.num_choices)
            if DEBUG:
                print '{0} Calculate q values for the batch {1}'.format(self.module_type,states_batch+[np.ones((batch_size,self.num_choices))])
                print 'The Q-values are: {0}'.format(q_values)
                        
            return q_values

        def select(self,policy,**kwargs):
            """Select the output(goal/action) based on the current state.
            Returns
            --------
            selected choice (goal or action)
            """
            policy_args=inspect.getargspec(policy.select)[0]
            #print 'Policy args'
            #print policy_args
            
            if len(policy_args)>1:
                if 'num_update' in policy_args:
                    choice=policy.select(self.calc_q_values([kwargs['state']]), self.num_samples)
                    if DEBUG:
                        print 'q values {0} for the state {1} and num of samples {2}. selected {3}'.format(kwargs['state'],self.calc_q_values([kwargs['state']]),self.num_samples,choice)
                else:
                    choice=policy.select(self.calc_q_values([kwargs['state']]))
                    if DEBUG:
                        print 'q values {0} for the state {1}. selected {2}'.format(kwargs['state'],self.calc_q_values([kwargs['state']]),choice)
            else:
                choice=policy.select()
         
            assert 0<=choice<self.num_choices

            return choice
            
            
        def update_policy(self,agent_writer=None):
            """Updates the modules's policy.
            """

            # sample the memory replay to get samples of experience <state, action, reward, next_state, is_terminal>
            exp_samples = self.memory.sample(self.batch_size)
            assert len(exp_samples)==self.batch_size
  
            #process the experience samples
            
            #try this: state_batch=[np.zeros((self.batch_size, )+self.state_shape[i][1:]) for i in range(len(self.state_shape))]
            state_batch=[np.zeros((self.batch_size, )+self.state_shape[i]) for i in range(len(self.state_shape))]
        
            #the next state-input batches
            next_state_batch=[np.zeros((self.batch_size, )+self.state_shape[i]) for i in range(len(self.state_shape))]
        
            #input mask needed to chose only one q-value
            mask_batch=np.zeros((self.batch_size,self.num_choices))
        
            #the q-value which corresponds to the applied action of the sample
            target_q_batch=np.zeros((self.batch_size,self.num_choices))
        
        
            for sample_idx in range(self.batch_size):
                for in_idx in range(len(self.state_shape)):
                    assert exp_samples[sample_idx].state[in_idx].shape==self.state_shape[in_idx] 
                    assert exp_samples[sample_idx].next_state[in_idx].shape==self.state_shape[in_idx]
                
                    state_batch[in_idx][sample_idx]=exp_samples[sample_idx].state[in_idx]
                    next_state_batch[in_idx][sample_idx]=exp_samples[sample_idx].next_state[in_idx] 
                
                #activate the output of the applied action
                mask_batch[sample_idx, exp_samples[sample_idx].action] = 1
                
            if DEBUG:
                print 'Update policy'
                print 'Batch with current states'
                print state_batch
                print 'Batch with next states'
                print next_state_batch
                print 'Action Mask Batch'
                print mask_batch
        
            #on the next state, chose the best predicted q-value on the fixed-target network
            predicted_q_batch = self.target_network.predict(next_state_batch+ [np.ones((self.batch_size,self.num_choices))],batch_size=self.batch_size)
            if DEBUG:
                print 'Predicted Q values {0}'.format(predicted_q_batch)
                
            assert predicted_q_batch.shape==(self.batch_size,self.num_choices)
                
            best_q_batch = np.amax(predicted_q_batch, axis=1)
            assert best_q_batch.shape==(self.batch_size,)
            
            #compute the target q-value r+gamma*max{a'}(Q(nextstat,a',qt)
            for sample_idx in range(self.batch_size):
                target_q_batch[sample_idx, exp_samples[sample_idx].action] = exp_samples[sample_idx].reward + self.gamma*best_q_batch[sample_idx]
            
            if DEBUG:
                print 'Train for target q {0} and state {1} and mask batch {2}'.format(target_q_batch,state_batch,mask_batch)
    
            #if DEBUG:
                #print 'Train Network for {0}'.format(self.module_type)
            loss = self.network.train_on_batch(x=state_batch+[mask_batch], y=target_q_batch)
            if DEBUG:
                print 'Just Trained Network for {0}'.format(self.module_type)

            save_scalar(self.num_updates, 'Loss for {0}'.format(self.module_type),  loss, self.writer)
            if agent_writer is not None:
                save_scalar(self.num_updates, 'Loss for {0}'.format(self.module_type), loss, agent_writer)
        
            #update the target network
            if self.num_updates>0 and self.num_updates % self.target_update_freq == 0:
                #if DEBUG:
                #    print 'Update the target network for {0}'.format(self.module_type)
                if DEBUG:
                    print 'Update target network'
                utils.get_hard_target_model_updates(self.target_network, self.network)
                if DEBUG:
                    print 'Just Updated the target network for {0}'.format(self.module_type)


        def save_model(self):
            self.network.save_weights('{0}_source_{1}.weight'.format(self.module_type,self.network_type))
            self.target_network.save_weights('{0}_target_{1}.weight'.format(self.module_type,self.network_type))



    """Class implementing Hierarchical Q-learning Agent.

    """
    def __init__(self,
                 controller_network_type,
                 metacontroller_network_type,
                 state_shape,#state[0] is the history  length
                 goal_shape,
                 num_actions,
                 num_goals,
                 controller_training_policy,
                 metacontroller_training_policy,
                 controller_testing_policy,
                 metacontroller_testing_policy,
                 controller_burnin_policy,
                 metacontroller_burnin_policy,
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
                 metacontroller_num_burnin=NUM_BURNIN,
                 ):

        #agent's description
        self.state_shape=state_shape
        self.goal_shape=goal_shape
        self.num_actions=num_actions
        self.num_goals=num_goals
        
        #agent's parameters
        self.eval_freq=eval_freq
        

        
        #agent's learning modules
        self.controller=self.Module(network_type=controller_network_type,
                                module_type='controller',
                                state_shape=[self.state_shape,self.goal_shape],
                                num_choices=num_actions,
                                burnin_policy=controller_burnin_policy,
                                training_policy=controller_training_policy,
                                testing_policy=controller_testing_policy,
                                gamma=controller_gamma,
                                alpha=controller_alpha,
                                optimizer=controller_optimizer,
                                loss_function = controller_loss_function,
                                batch_size=controller_batch_size,
                                num_burnin=controller_num_burnin)
                                                            
        self.metacontroller=self.Module(network_type=metacontroller_network_type,
                                    module_type='metacontroller',
                                    state_shape=[self.state_shape],
                                    num_choices=num_goals,
                                    burnin_policy=metacontroller_burnin_policy,
                                    training_policy=metacontroller_training_policy,
                                    testing_policy=metacontroller_testing_policy,
                                    gamma=metacontroller_gamma,
                                    alpha=metacontroller_alpha,
                                    optimizer=metacontroller_optimizer,
                                    loss_function = metacontroller_loss_function,
                                    batch_size=metacontroller_batch_size,
                                    num_burnin=metacontroller_num_burnin)
        
        #tensorbboard logistics
        self.sess = tf.Session()
        self.controller.writer.add_graph(tf.get_default_graph())
        self.metacontroller.writer.add_graph(tf.get_default_graph())
        self.writer=tf.summary.FileWriter('./logs_hdqn_{0}'.format(controller_network_type))
                

    def env_preprocess(self,state):
        vector = np.zeros(self.state_shape)
        vector[0,state] = 1.0
        return vector
        #return np.expand_dims(vector, axis=0)
    
  
    def goal_preprocess(self,goal):
        #print self.goal_shape
        vector = np.zeros(self.goal_shape)
        vector[0,goal] = 1.0
        return vector
        #return np.expand_dims(vector, axis=0)
    
    #compute the intrinsic reward achieved
    @staticmethod
    def get_intrinsic_reward(goal,state):
        return 1.0 if goal == state else 0.0
    
    @staticmethod
    def goal_reached(goal,state):
        return goal==state
        
    def fit(self, env, eval_env, num_episodes,eval_num_episodes,reset_env_fit_logs):
        """Fit your model to the provided environment.
        
        Parameters
        ----------
        env: agent's environment
        eval_env: copy of agent's environment used for the evaluation
        num_episodes: number of episodes for the training
        """
        #replay memories (of controller and metacontroller) burnin  
        
        print 'Start Burn-in'
        for episode_idx in range(num_episodes):
            #start new episode
            
            #get initial state
            state=env.reset()
            assert state is not None
            proc_state=self.env_preprocess(state)
            assert proc_state is not None
            
            #print 'Burnin episode: {0}'.format(episode_idx)
            
            if DEBUG:
                print 'Burnin New Episode {0}'.format(episode_idx)
                print 'Initial State {0}'.format(proc_state)
            
            #select next goal 
            goal=self.metacontroller.select(self.metacontroller.burnin_policy)
            assert goal is not None     
            proc_goal=self.goal_preprocess(goal)
            assert proc_goal is not None
            
            
        
            #new episode
            while True: 
                
                #the first state of the new goal
                proc_state0=proc_state
                #the total environmental reward achieved by setting this goal
                extrinsic_reward=0
                
                #new goal
                if DEBUG:
                    print 'Next goal {0}'.format(proc_goal)
                 
                #goal has not been reached and the episode has not finished 
                while True:
                    
                    #select next action given the current goal
                    action=self.controller.select(self.controller.burnin_policy)
                    assert action is not None
                    
                    if DEBUG:
                        print 'Next action {0}'.format(action)
                    
                    #apply the action to the environment, get reward and nextstate
                    next_state, reward, is_terminal= env.step(action)
                    assert next_state is not None
                    proc_next_state=self.env_preprocess(next_state)
                    assert proc_next_state is not None
                  
                    #compute the internal and external reward
                    extrinsic_reward +=reward
                    intrinsic_reward=self.get_intrinsic_reward(goal,next_state)
                    if DEBUG:
                        print 'Next state {0} {1} : Goal {2} {3} Intrinsic Reward {4} Extrinsic Reward {5} Action{6}'.format(proc_next_state,next_state, proc_goal, goal,intrinsic_reward,extrinsic_reward,action)
                        
                    #store the experience in the controller's memory
                    #self.controller.num_samples+=1
                    self.controller.memory.append([proc_state,proc_goal],\
                                                   action,\
                                                   intrinsic_reward,\
                                                   [proc_next_state,proc_goal],\
                                                   is_terminal)

                    proc_state=proc_next_state
                    if is_terminal or self.goal_reached(next_state,goal):
                        if DEBUG:
                            print 'New goal and maybe new episode'
                        break
                                    
                #store the experience in the metacontroller's memory
                #self.metacontroller.num_samples+=1
                self.metacontroller.memory.append([proc_state0], \
                                                   goal, \
                                                   extrinsic_reward,\
                                                   [proc_next_state],\
                                                   is_terminal)
                if is_terminal:
                    #start new episode
                    if DEBUG:
                        print 'Start new episode {0}'.format(episode_idx)
                    break
                else:
                    #select next goal 
                    if DEBUG:
                        print 'Start new goal within the same episode'
                        
                    goal=self.metacontroller.select(self.metacontroller.burnin_policy)
                    proc_goal=self.goal_preprocess(goal)
                    assert proc_goal is not None
         
        #start training the networks
        env.reset_fit_logs()
        total_extrinsic_reward=0
        total_intrinsic_reward=0
        goal_num_samples=np.zeros(self.num_goals)
        
        for self.num_train_episodes in range(num_episodes):
            #start new episode
            print 'Training episode: {0}'.format(self.num_train_episodes)
            
            #check if it's time to evaluate the agent
            if self.num_train_episodes>0 and self.num_train_episodes % self.eval_freq == 0:
                #compute the reward, average episode length achieved in new episodes by the current agent
                self.evaluate(eval_env,eval_num_episodes)
            
            #get initial state
            state=env.reset()
            assert state is not None
            proc_state=self.env_preprocess(state)
            assert proc_state is not None
            
            if DEBUG:
                print 'Training New Episode {0}'.format(episode_idx)
                #print 'Initial State {0}'.format(proc_state)
            
                            
            #select next goal 
            #print 'Select goal {0}'.format(self.metacontroller.num_samples)
            goal=self.metacontroller.select(policy=self.metacontroller.training_policy,state=[proc_state],num_update=self.metacontroller.num_samples) 
            
            assert goal is not None     
            proc_goal=self.goal_preprocess(goal)
            assert proc_goal is not None
                        

            #new episode
            while True: 
                
                proc_state0=proc_state
                extrinsic_reward=0
                
                #new goal
                if DEBUG:
                    print 'Next goal {0}'.format(proc_goal)
                 
                #goal has not been reached and the episode has not finished 
                while True:
                    
                    #select next action given the current goal
                    #print 'Select action for goal {0} {1}'.format(goal,goal_num_samples[goal])
                    action=self.controller.select(policy=self.controller.training_policy,state=[proc_state,proc_goal],num_update=goal_num_samples[goal])

                    #apply the action to the environment, get reward and nextstate
                    next_state, reward, is_terminal= env.step(action)
                    assert next_state is not None
                    proc_next_state=self.env_preprocess(next_state)
                    assert proc_next_state is not None
                  
                    #compute the internal and external reward
                    extrinsic_reward +=reward
                    total_extrinsic_reward +=reward
                    intrinsic_reward=self.get_intrinsic_reward(goal,next_state)
                    if DEBUG:
                        print 'Next state {0} {1} : Goal {2} {3} Intrinsic Reward {4} Extrinsic Reward {5} Action{6}'.format(proc_next_state,next_state, proc_goal, goal,intrinsic_reward,extrinsic_reward,action)
                        
                    total_intrinsic_reward+=intrinsic_reward
                    
                    
                    #store the experience in the controller's memory
                    self.controller.num_samples+=1
                    self.controller.memory.append([proc_state,proc_goal],\
                                                   action,\
                                                   intrinsic_reward,\
                                                   [proc_next_state,proc_goal],\
                                                   is_terminal)
                    
  
                                                                
                    #update the weights of the controller's network                            
                    self.controller.update_policy(self.writer)
                    self.controller.num_updates += 1
                    
                    #update the weights of the metacontroller's network  
                    self.metacontroller.update_policy(self.writer)
                    self.metacontroller.num_updates += 1
            
                    #update the tensorboard training metrics
                    #save_scalar(self.metacontroller.num_updates,'total training extrinsic reward',total_extrinsic_reward,self.metacontroller.writer)
                    #save_scalar(self.controller.num_updates,'total training intrinsic reward',total_intrinsic_reward,self.controller.writer)
                    save_scalar(self.metacontroller.num_updates,'Total Training Extrinsic Reward',total_extrinsic_reward,self.writer)
                    save_scalar(self.controller.num_updates,'Total Training Intrinsic Reward',total_intrinsic_reward,self.writer)
   


                    #check if it's time to store the controller's model
                    if self.controller.num_updates>0  and self.controller.num_updates % SAVE_FREQ==0:
                        self.controller.save_model()
                        
                    if self.metacontroller.num_updates>0  and self.controller.num_updates % SAVE_FREQ==0:
                        self.controller.save_model()
                                                
                    proc_state=proc_next_state
                    if is_terminal or self.goal_reached(next_state,goal):
                        goal_num_samples[goal]+=1
                        if DEBUG:
                            print 'New goal and maybe new episode'
                              
                        break
                                    
                #store the experience in the metacontroller's memory
                self.metacontroller.num_samples+=1
                self.metacontroller.memory.append([proc_state0], \
                                                   goal, \
                                                   extrinsic_reward,\
                                                   [proc_next_state],\
                                                   is_terminal)
                
                if is_terminal:
                    #start new episode
                    if (self.num_train_episodes+1) % reset_env_fit_logs==0:
                        for state_idx in range(env.num_states):
                            save_scalar(self.num_train_episodes,'Visit Counts for state {0}'.format(state_idx),env.visit_counts[state_idx],self.writer)
                        env.reset_fit_logs()
                        #print 'Visit counts for each state '
                        #print env.visit_counts
                    
                    if DEBUG:
                        print 'Start new episode {0}'.format(self.num_train_episodes)
                    break
                else:
                    #select next goal 
                    if DEBUG:
                        print 'Start new goal within the same episode'
                    goal=self.metacontroller.select(policy=self.metacontroller.training_policy,state=[proc_state],num_update=goal_num_samples[goal])   
                    proc_goal=self.goal_preprocess(goal)
                
    def evaluate(self,env,num_episodes):
        """
        Evaluate the performance of the agent in the environment.
        Parameters
        ----------
        env: agent's environment
        num_episodes: number of episodes for the testing
        """
    
        total_reward = 0
        episode_length = 0
        
        print 'Start Evaluation'
        for episode_idx in range(num_episodes):
            
            
            #start new episode
            
            print 'Total reward {0}'.format(total_reward)
            state=env.reset()
            assert state is not None
            proc_state=self.env_preprocess(state)
            assert proc_state is not None
            
            if DEBUG:
                print 'Evaluating New Episode {0}'.format(episode_idx)
                print 'Initial State {0}'.format(proc_state)
                
            #print 'Initial State {0}'.format(proc_state)
                            
            #select next goal 
            goal=self.metacontroller.select(policy=self.metacontroller.testing_policy,state=[proc_state])      
            #goal=6
            proc_goal=self.goal_preprocess(goal)
            
            
            #new episode
            while True: 
                
                state_0=proc_state

                if DEBUG:
                    print 'Next goal {0}'.format(proc_goal)
                #new goal
                while True:
                    action=self.controller.select(policy=self.controller.testing_policy,state=[proc_state,proc_goal])
                    
                    #apply the action to the environment, get reward and nextstate
                    next_state, reward, is_terminal= env.step(action)
                    assert next_state is not None
                    proc_next_state=self.env_preprocess(next_state)
                    assert proc_next_state is not None
                    
                    #print 'Previous State {0} Next state {1} : Goal {2} Extrinsic Reward {3} Action {4}'.format(proc_state,proc_next_state, proc_goal,reward, action)
                    if DEBUG:
                    #print '
                    #print 'Reward {0}'.format(reward)
                    #print 'Action {0}'.format(action)
                    #print 'Previous State {0} Next state {1} : Goal {2} Extrinsic Reward {3} Action {4}'.format(proc_state,proc_next_state, proc_goal,action, reward)
                        print 'Previous State {0} Next state {1} : Goal {2} Extrinsic Reward {3} Action {4}'.format(proc_state,proc_next_state, proc_goal,reward, action)
                    
                    
                    #compute the internal and external reward
       
                    total_reward +=reward

                    episode_length+=1

                    if is_terminal or self.goal_reached(next_state,goal):
                        break
                                    
                    state = next_state
                    proc_state=proc_next_state
                                        
                if is_terminal:
                    #start new episode
                    break
                else:
                    #select next goal 
                    goal=self.metacontroller.select(policy=self.metacontroller.testing_policy,state=[proc_state])  
                    proc_goal=self.goal_preprocess(goal)
                    #goal=0

        #update the tensorboard logistics
        #save_scalar(self.controller.num_updates,'Total Testing Reward',total_reward,self.controller.writer)
        #save_scalar(self.controller.num_updates,'Testing Episode Length ',episode_length/num_episodes,self.controller.writer)
        #save_scalar(self.controller.num_updates,'Testing Total Reward',total_reward,self.metacontroller.writer)
        #save_scalar(self.controller.num_updates,'Testing episode length ',episode_length/num_episodes,self.metacontroller.writer)
        save_scalar(self.num_train_episodes,'Testing Total Reward',total_reward,self.writer)
        save_scalar(self.num_train_episodes,'Testing Episode Length ',episode_length/num_episodes,self.writer)
