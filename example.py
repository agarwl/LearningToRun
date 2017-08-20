# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import ModelIntervalCheckpoint

from osim.env import *

from keras.optimizers import RMSprop

import argparse
import math

# from newRun import MyRunEnv
from myRunEnv import myRunEnv



# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=5000000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()

# Load walking environment
# env = RunEnv(args.visualize)


env = myRunEnv(args.visualize)
env.reset(difficulty=0, seed=42)

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

observation_space_shape = env.get_observation_space_shape()
reg = 1e-6
# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + observation_space_shape))
actor.add(Dense(128, kernel_regularizer=l2(reg)))
actor.add(LeakyReLU())
actor.add(Dense(64, kernel_regularizer=l2(reg)))
actor.add(LeakyReLU())
actor.add(Dense(nb_actions, kernel_regularizer=l2(reg)))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + observation_space_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(512,kernel_regularizer=l2(reg))(x)
x = LeakyReLU()(x)
x = Dense(256,kernel_regularizer=l2(reg))(x)
x = LeakyReLU()(x)
x = Dense(1, kernel_regularizer=l2(reg))(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.995, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.


num = 1000
filepath = "shallow_model"
checkpointer = ModelIntervalCheckpoint(filepath, interval = nallsteps//num, verbose = 1)


if args.train:
  agent.load_weights(args.model)
	agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit,
	log_interval=20000, callbacks=[checkpointer])
    # After training is done, we save the final weights.
	agent.save_weights(args.model, overwrite=True)

if not args.train:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)