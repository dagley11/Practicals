# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey3 import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, alpha = .95, gamma = .7, bins = 5):
        self.num_bins = bins
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 4
        self.i = 0
        self.state = None
        self.Q = {}
        self.key = None
        self.action = 0
        self.alpha = alpha
        self.Gamma = gamma
        self.rand = .1
        self.vel_bins = np.linspace(-48,29,self.num_bins)
        self.monkey_bot_bins = np.linspace(0,343,self.num_bins)
        self.tree_top_bins = np.linspace(211,346,self.num_bins)
        self.tree_bot_bins = np.linspace(11,140,self.num_bins)
        self.gravity_bins = np.linspace(-27,-1,self.num_bins)
        self.dist_bins = np.linspace(-115,460,self.num_bins)
        

    def reset(self):        
        self.gravity = -10
        self.i = 0
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        self.last_state = self.state
        self.last_key = self.key
        self.last_action = self.action
        self.state = state
        
        if self.i == 1:
            self.infer_gravity()
        self.state['gravity'] = self.gravity

        self.bin_states(self.num_bins)
        self.convert_to_string()

        if self.key in self.Q:
            if self.Q[self.key][0] > self.Q[self.key][1]:
                #print('dont jump')
                self.action  = 0
            elif self.Q[self.key][1] > self.Q[self.key][0]:
                #print('jump')
                self.action = 1
            else:
                if npr.random() < self.rand:
                    self.action = 1
                else:
                    self.action = 0
        else:
            self.Q[self.key]=[0,0]
            if npr.random() < self.rand:
                self.action = 1
            else:
                self.action = 0

        

        return self.action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if self.i > 0:
            if self.last_action == 0:
                self.Q[self.last_key][0] = self.Q[self.last_key][0] + self.alpha * (reward + self.Gamma*max(self.Q[self.key]) - self.Q[self.last_key][0])
            elif self.last_action == 1:
                self.Q[self.last_key][1] = self.Q[self.last_key][1] + self.alpha * (reward + self.Gamma*max(self.Q[self.key]) - self.Q[self.last_key][1])

    def bin_states(self, bin_num):
        if self.state['gravity'] <= self.gravity_bins[0]:
            self.state['gravity'] = 0
        elif self.state['gravity'] > self.gravity_bins[bin_num-1]:
            self.state['gravity'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['gravity'] > self.gravity_bins[i] and self.state['gravity'] <= self.gravity_bins[i+1]:
                self.state['gravity'] = i

        if self.state['monkey']['bot'] <= self.monkey_bot_bins[0]:
            self.state['monkey']['bot'] = 0
        elif self.state['monkey']['bot'] > self.monkey_bot_bins[bin_num-1]:
            self.state['monkey']['bot'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['monkey']['bot'] > self.monkey_bot_bins[i] and self.state['monkey']['bot'] <= self.monkey_bot_bins[i+1]:
                self.state['monkey']['bot'] = i

        if self.state['monkey']['vel'] <= self.vel_bins[0]:
            self.state['monkey']['vel'] = 0
        elif self.state['monkey']['vel'] > self.vel_bins[bin_num-1]:
            self.state['monkey']['vel'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['monkey']['vel'] > self.vel_bins[i] and self.state['monkey']['vel'] <= self.vel_bins[i+1]:
                self.state['monkey']['vel'] = i

        if self.state['tree']['bot'] <= self.tree_bot_bins[0]:
            self.state['tree']['bot'] = 0
        elif self.state['tree']['bot'] > self.tree_bot_bins[bin_num-1]:
            self.state['tree']['bot'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['tree']['bot'] > self.tree_bot_bins[i] and self.state['tree']['bot'] <= self.tree_bot_bins[i+1]:
                self.state['tree']['bot'] = i

        if self.state['tree']['top'] <= self.tree_top_bins[0]:
            self.state['tree']['top'] = 0
        elif self.state['tree']['top'] > self.tree_top_bins[bin_num-1]:
            self.state['tree']['top'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['tree']['top'] > self.tree_top_bins[i] and self.state['tree']['top'] <= self.tree_top_bins[i+1]:
                self.state['tree']['top'] = i

        if self.state['tree']['dist'] <= self.dist_bins[0]:
            self.state['tree']['dist'] = 0
        elif self.state['tree']['dist'] > self.dist_bins[bin_num-1]:
            self.state['tree']['dist'] = bin_num
        else:
            for i in range(1,bin_num-1):
               if self.state['tree']['dist'] > self.dist_bins[i] and self.state['tree']['dist'] <= self.dist_bins[i+1]:
                self.state['tree']['dist'] = i
    
    def convert_to_string(self):
        self.key = 'g' + str(self.state['gravity']) \
        + 'b' + str(self.state['monkey']['bot']) \
        + 'v' + str(self.state['monkey']['vel']) \
        + 't' + str(self.state['tree']['bot']) \
        + 'u' + str(self.state['tree']['top']) \
        + 'd' + str(self.state['tree']['dist'])

    def infer_gravity(self):
        self.gravity = self.last_state['monkey']['vel'] - self.state['monkey']['vel']
 

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            learner.action_callback(swing.get_state())
            #hist.append(learner.state)
            learner.i += 1
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    

    # Empty list to save history.
    

    # Run games. 
    '''
    for bins in [5,10,20,50,75,100]:
        agent = Learner(bins=bins)
        hist = []
        run_games(agent, hist, 1000, 0)

        # Save history. 
        #out = np.load('hist.npy')
        print(bins, np.mean(np.array(hist)))
        #np.save('hist',np.array(hist))

    for alpha in [1,.9,.8,.7,.6,.5]:
        agent = Learner(alpha=alpha)
        hist = []
        run_games(agent, hist, 1000, 0)

        # Save history. 
        #out = np.load('hist.npy')
        print(alpha, np.mean(np.array(hist)))
        #np.save('hist',np.array(hist))
    
    for gamma in [1,.9,.8,.7,.6,.5]:
        agent = Learner(gamma=gamma)
        hist = []
        run_games(agent, hist, 1000, 0)

        # Save history. 
        #out = np.load('hist.npy')
        print(gamma, np.mean(np.array(hist)))
        #np.save('hist',np.array(hist))

    
    agent = Learner(gamma=.8, bins = 75, alpha = .6)
    hist = []
    run_games(agent, hist, 1000, 0)

    # Save history. 
    #out = np.load('hist.npy')
    print(np.mean(np.array(hist)))
    #np.save('hist',np.array(hist))
    '''


    agent = Learner(gamma=.8, bins = 75, alpha = .6)
    hist = []
    run_games(agent, hist, 10000, 0)

        # Save history. 
        #out = np.load('hist.npy')
    np.save('hist1',np.array(hist))  

    agent = Learner(gamma=.8, bins = 150, alpha = .6)
    hist = []
    run_games(agent, hist, 10000, 0)

        # Save history. 
        #out = np.load('hist.npy')
    np.save('hist2',np.array(hist))  
