import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.regularizers import l2

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        observation_space_shape = state_size#(state_size,1)
        print(observation_space_shape)
        S = Input(shape=[observation_space_shape])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Brake = Dense(action_dim, activation='sigmoid')(h1) #,init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        model = Model(inputs = S,outputs = Brake)
        return model, model.trainable_weights, S

        # model = Sequential()
        # model.add(Flatten(input_shape=(1,) + observation_space_shape))
        # model.add(Dense(128, kernel_regularizer=l2(reg)))
        # model.add(LeakyReLU())
        # model.add(Dense(64, kernel_regularizer=l2(reg)))
        # model.add(LeakyReLU())
        # model.add(Dense(nb_actions, kernel_regularizer=l2(reg)))
        # model.add(Activation('sigmoid'))
        # print(model.summary())
        # return model, model.trainable_weights, model.input