from osim.env.run import RunEnv
from itertools import chain
import numpy as np

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


class myRunEnv(RunEnv):
    """docstring for myenv"""
    # STATE_TOES_L = 28
    # STATE_TOES_R = 30
    ninput = 41
    body_loc = 22
    nposition = 14
    def __init__(self, visualize = True, max_obstacles = 3):
        super(myRunEnv, self).__init__(visualize, max_obstacles)
        self.prev_body = np.zeros(self.nposition)
        # print(self.ninput)
        # self.f = open('temp_rewards', 'w+')

    def get_observation(self):
        super(myRunEnv, self).get_observation()
        self.curr_body = np.array(self.current_state[self.body_loc : self.body_loc + self.nposition])
        self.current_state += list((self.curr_body - self.prev_body)/0.01)
        self.prev_body = self.curr_body
        return self.current_state


    def get_observation_space_shape(self):
        return (self.observation_space.shape[0] + self.nposition, )

    # def compute_reward(self):
    #   reward = super(myenv, self).compute_reward()
    #   f = self.f
    #   left, right = self.current_state[self.STATE_TOES_L], self.current_state[self.STATE_TOES_R]
    #   # f.write("Orig Reward {0}".format(reward))
    #   # if left > right:
    #   #   reward += self.current_state[self.STATE_TOES_R] - self.last_state[self.STATE_TOES_R]
    #   # else:
    #   #   reward += self.current_state[self.STATE_TOES_L] - self.last_state[self.STATE_TOES_L]
    #   # f.write(" {0}\n".format(reward))
    #   f.write("left: {0} right: {1}\n".format(left, right))
    #   return reward
