# A3C Global agent and local agents for training and evaluation
# data parallelism

import gym

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import threading
import multiprocessing

from a3c_actor import Global_Actor, Worker_Actor
from a3c_critic import Global_Critic, Worker_Critic

# shared global parameters across all workers
global_episode_count = 0
global_step = 0
global_episode_reward = []  # save the results


class A3Cagent(object):

    """
        Global network
    """
    def __init__(self, env_name):

        # training environment
        self.env_name = env_name
        self.WORKERS_NUM = multiprocessing.cpu_count() #4

        # hyperparameters
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.ENTROPY_BETA = 0.01

        # get state dimension
        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        # get action dimension
        action_dim = env.action_space.shape[0]
        # get action bound
        action_bound = env.action_space.high[0]

        # create global actor and critic networks
        self.global_actor = Global_Actor(state_dim, action_dim, action_bound, self.ACTOR_LEARNING_RATE,
                                         self.ENTROPY_BETA)
        self.global_critic = Global_Critic(state_dim, action_dim, self.CRITIC_LEARNING_RATE)


    def train(self, max_episode_num):

        workers = []

        # create worker
        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(A3Cworker(worker_name, self.env_name, self.global_actor,
                                     self.global_critic, max_episode_num))


         # create worker (multi-agents) and do parallel training
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        np.savetxt('./save_weights/pendulum_epi_reward.txt', global_episode_reward)
        print(global_episode_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()


class A3Cworker(threading.Thread):

    """
        local agent network (worker)
    """
    def __init__(self, worker_name, env_name, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        #self.lock = threading.Lock()

        # hyperparameters
        self.GAMMA = 0.95
        self.t_MAX = 4 # t-step prediction

        self.max_episode_num = max_episode_num

        # environment
        self.env = gym.make(env_name)
        self.worker_name = worker_name

        # global network sharing
        self.global_actor = global_actor
        self.global_critic = global_critic


        # get state dimension
        self.state_dim = self.env.observation_space.shape[0]
        # get action dimension
        self.action_dim = self.env.action_space.shape[0]
        # get action bound
        self.action_bound = self.env.action_space.high[0]

        # create local actor and critic networks
        self.worker_actor = Worker_Actor(self.state_dim, self.action_dim, self.action_bound)
        self.worker_critic = Worker_Critic(self.state_dim)

        # initial transfer global network parameters to worker network parameters
        self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
        self.worker_critic.model.set_weights(self.global_critic.model.get_weights())


    ## computing Advantages and targets: y_k = r_k + gamma*V(s_k+1), A(s_k, a_k)= y_k - V(s_k)
    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets


    ## convert (list of np.array) to np.array
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack


    # train each worker
    def run(self):

        global global_episode_count, global_step
        global global_episode_reward  # total episode across all workers

        print(self.worker_name, "starts ---")

        while global_episode_count <= int(self.max_episode_num):

            # initialize batch
            batch_state, batch_action, batch_reward = [], [], []

            # reset episode
            step, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset() # shape of state from gym (3,)

            while not done:

                # visualize the environment
                #self.env.render()
                # pick an action (shape of gym action = (action_dim,) )
                action = self.worker_actor.get_action(state)
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)

                # change shape (state_dim,) -> (1, state_dim), same to action, next_state
                state = np.reshape(state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                action = np.reshape(action, [1, self.action_dim])

                # append to the batch
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward+8)/8) # <-- normalization
                #batch_reward.append(reward)

                # update state and step
                state = next_state
                step += 1
                episode_reward += reward[0]

                # if batch is full or episode ends, start to train global on batch
                if len(batch_state) == self.t_MAX or done:

                    # extract states, actions, rewards from batch
                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward)

                    # clear the batch
                    batch_state, batch_action, batch_reward = [], [], []

                    # compute n-step TD target and advantage prediction with global network
                    next_state = np.reshape(next_state, [1, self.state_dim])
                    next_v_value = self.global_critic.model.predict(next_state)
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    v_values = self.global_critic.model.predict(states)
                    advantages = n_step_td_targets - v_values


                    #with self.lock:
                    # update global critic
                    self.global_critic.train(states, n_step_td_targets)
                    # update global actor
                    self.global_actor.train(states, actions, advantages)

                    # transfer global network parameters to worker network parameters
                    self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
                    self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

                    # update global step
                    global_step += 1

                if done:
                    # update global episode count
                    global_episode_count += 1
                    ## display rewards every episode
                    print('Worker name:', self.worker_name, ', Episode: ', global_episode_count,
                          ', Step: ', step, ', Reward: ', episode_reward)

                    global_episode_reward.append(episode_reward)

                    ## save weights every episode
                    if global_episode_count % 10 == 0:
                        self.global_actor.save_weights("./save_weights/pendulum_actor.h5")
                        self.global_critic.save_weights("./save_weights/pendulum_critic.h5")

