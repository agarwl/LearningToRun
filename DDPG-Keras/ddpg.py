# from gym_torcs import TorcsEnv
from osim.env import RunEnv
import numpy as np
import random
import argparse
import keras as K
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import os
from parallel_env import ei

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import multiprocessing

OU = OU()       #Ornstein-Uhlenbeck Process
NUM = 4

def f((i,action)):
    x = env[i].step(action)
    return x

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.995
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 18  #Steering/Acceleration/Brake
    state_dim = 55  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.0
    episode_count = 10000
    max_steps = 10000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    pool = multiprocessing.Pool(NUM)
    # for _ in range(1000):

    # Generate a Torcs environment
    # env = RunEnv(visualize=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))


        # s_t = env.reset(difficulty = 2, seed=42)
        s_t_arr = [None]*NUM
        done = [False]*NUM
        for k in range(NUM):
            s_t_arr[k] = env[k].reset(k)
        s_t_arr = np.asarray(s_t_arr)
        a_t = np.zeros((NUM, action_dim))

        total_reward = 0.
        for j in range(max_steps):
            # for s_t in s_t_arr:
                # s_t = np.reshape(s_t, (1, -1))
            loss = 0
            epsilon -= 1.0 / EXPLORE
            noise_t = np.zeros((NUM, action_dim))

            a_t_original = actor.model.predict(s_t_arr)
            # print(a_t_original)
            noise_t = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0],  0.0 , 0.60, 0.30)
            a_t = a_t_original + noise_t
            # a_t =  np.reshape(a_t.shape[0], -1)
            o = pool.map(f, zip(xrange(NUM), a_t))

            for k in range(NUM):
                ob, r_t, done[k], info =  o[k] #env.step(a_t)
                if step % 2 == 0:
                    s_t1 = np.reshape(ob, -1)
                    buff.add(s_t_arr[k], a_t[k], r_t, s_t1, done[k])      #Add replay buffer
                    s_t_arr[k] = s_t1
                total_reward += r_t

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            # total_reward += r_t
            # s_t_arr = s_t1
            if step % (max_steps//5) == 0:
                print("Episode", i, "Step", step, "Reward", r_t, "Loss", loss, "Total_reward", total_reward)

            step += 1
            if np.any(done):
                break

        if np.mod(i, 20) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    env = [None]*NUM
    for i in range(NUM):
        env[i] = ei()
    playGame(1)
