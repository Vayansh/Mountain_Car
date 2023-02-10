import gym
import cv2 as cv
import random
import numpy as np

# Hyperparameters 
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_step = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500* EPSILON_MIN/max_num_step
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 50

#  Q_learner class to learn game
class Q_learner(object):              
    def __init__(self,env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bin = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bin
        self.action_shape = env.action_space.n
        
        self.Q_table = np.zeros((self.obs_bin+1,self.obs_bin+1,self.action_shape))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1
        
    def discrete(self,obs):
        if isinstance(obs[1],dict):
            obs = obs[0]
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))
    
    def get_action(self,obs):
        discreted_obs = self.discrete(obs)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q_table[discreted_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])
        
    def learn(self,obs,action,reward,next_obs):
        discreted_obs = self.discrete(obs)
        discreted_next_obs = self.discrete(next_obs)
        td_target = reward + self.gamma* np.max(self.Q_table[discreted_next_obs])
        td_error = td_target - self.Q_table[discreted_obs][action]
        self.Q_table[discreted_obs][action] += self.alpha* td_error
        
        
    
# Agent class to play game    
class agent:                            
    def __init__(self):
        self.env = gym.make('MountainCar-v0',render_mode = "rgb_array")
        self.env.metadata['render_fps'] = 30
        self.action = self.env.action_space.n
        self.MAX_STEP = 50
        self.env.reset()
        self.learner = Q_learner(self.env)
        
    def random_action(self):
        done = False
        score = 0
        s = self.env.reset()
        self.MAX_STEP = 50
        while not done:
            self.MAX_STEP-=1
            if self.MAX_STEP <= 0:
                break
            a = random.choice([0,1,2])
            state,reward,done,_,_ = self.env.step(a)
            arr = self.env.render()
            cv.imshow('window',arr)
            cv.waitKey(100)
            score += reward
        cv.destroyAllWindows()
        print(score)
    
    def train(self):
        best_reward = -float('inf')
        for episodes in range(MAX_NUM_EPISODES):
            done = False
            s = self.env.reset()
            s = s[0]
            total_reward = 0
            while not done:
                action = self.learner.get_action(s)
                next_s,reward,done,_,_ = self.env.step(action)
                self.learner.learn(s,action,reward,next_s)
                s = next_s
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episode: {} reward: {} best_reward: {} eps: {}".format(episodes,total_reward,best_reward,self.learner.epsilon))
        return np.argmax(self.learner.Q_table,axis=2)
            
    def test(self,policy):
        done = False
        s = self.env.reset()
        total_reward = 0
        while not done:
            action = policy[self.learner.discrete(s)]
            next_s,reward,done,_,_ = self.env.step(action)
            arr = self.env.render()
            cv.imshow('window',arr)
            cv.waitKey(50)
            s = next_s
            total_reward += reward
        print(total_reward)    
    
        
if __name__ == '__main__':
    ag = agent()
    MAX_EPISODE = 10
    # learned_policy = ag.train()
    # np.save('MountainCar_policy.npy',learned_policy)
    learned_policy = np.load('MountainCar_policy.npy')
    for i in range(MAX_EPISODE):
        ag.test(learned_policy)   
    ag.env.close()