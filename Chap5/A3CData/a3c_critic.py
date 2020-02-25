# A3C Critic data parallelism

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import tensorflow as tf


## critic network
def build_network(state_dim):
    state_input = Input((state_dim,))
    h1 = Dense(64, activation='relu')(state_input)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)
    v_output = Dense(1, activation='linear')(h3)
    model = Model(state_input, v_output)
    #model.summary()
    model._make_predict_function()  # class 안에서 def가 정의되면 필요없음
    return model


class Global_Critic(object):
    """
        Global Critic Network for A3C: V function approximator
    """
    def __init__(self, state_dim, action_dim, learning_rate):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.model = build_network(state_dim)

        self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    ## train the critic network run by worker
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            # loss function and its gradient
            v_values = self.model(states)
            loss = tf.reduce_sum(tf.square(td_targets-v_values))
        dj_dphi = tape.gradient(loss, self.model.trainable_variables)

        # gradient clipping
        dj_dphi, _ = tf.clip_by_global_norm(dj_dphi, 40) #40

        # gradients
        grads = zip(dj_dphi, self.model.trainable_variables)

        self.critic_optimizer.apply_gradients(grads)


    ## save critic weights
    def save_weights(self, path):
        self.model.save_weights(path)


    ## load critic wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5')



class Worker_Critic(object):
    """
        Critic Network for A3C: V function approximator
    """
    def __init__(self, state_dim):

        self.model = build_network(state_dim)
