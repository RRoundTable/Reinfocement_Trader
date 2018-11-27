"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃

Github: https://github.com/DongjunLee/dqn-tensorflow/blob/master/main.py
"""

from DeepQNet.DQN import *
from collections import deque
import logging
import os
import random
from typing import List

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import normalize
#import gym

from DeepQNet.DQN import DQN
from DeepQNet.environment import Env, Env2


flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episodes')
flags.DEFINE_integer('target_update_count', 1000, 'DQN Target Network update count')  # default 5
flags.DEFINE_integer('max_episode_count', 100, 'Number of maximum episodes')
flags.DEFINE_integer('input_size', 128, 'Input size.')
flags.DEFINE_integer('output_size', 3, 'output size. (Number of actions)')
flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divided evenly into the dataset sizes)')
flags.DEFINE_integer('frame_size', 1, 'Frame size. (Stack env\'s observation T-n ~ T')
flags.DEFINE_string('model_name', 'ConvNetv1', 'DeepLearning Network model name. (MLPv1, ConvNetv1, ConvNetv2')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_boolean('step_verbose', True, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
flags.DEFINE_float('transaction_cost', 0.0, 'transaction cost')



FLAGS = flags.FLAGS

logger = logging.getLogger()
# fileHandler = logging.FileHandler('./myLoggerTest.log')
streamHandler = logging.StreamHandler()
# logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)


def replay_train(behaviorDQN: DQN, targetDQN: DQN, train_batch: list):
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        behaviorDQN (DeepQNetwork): Behavior DQN that will be trained
        targetDQN (DeepQNetwork): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    # print(np.array([x[0] for x in train_batch]).shape)
    states = np.vstack([x[0] for x in train_batch])

    actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
    rewards = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch[:FLAGS.batch_size]])

    predict_result = targetDQN.predict(next_states)

    Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1)

    X = states
    y = behaviorDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return behaviorDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def csv_save(actions, rewards, losses, dir, filename):
    header = ["cum_reward", "rewards", "actions", "losses"]
    data_dict = dict()
    data_shape = np.array(rewards).shape
    rewards_reshaped = np.array(rewards).reshape([-1])
    cumulative_reward = np.cumsum(rewards_reshaped).reshape(data_shape)

    data_dict[header[0]] = cumulative_reward
    data_dict[header[1]] = rewards
    data_dict[header[2]] = actions
    # data_dict[header[3]] = losses

    df = pd.DataFrame(data_dict)

    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_csv(dir + filename)

def train(sess, env, mainDQN, targetDQN):
    logger.info("FLAGS configure.")
    logger.info(FLAGS.__flags)


    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)

    # store last games rewards
    consecutive_len = 100 # default value
    last_n_game_reward = deque(maxlen=consecutive_len)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    # initial copy q_net -> target_net
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    sess.run(copy_ops)

    global_step = 1

    history = []
    for episode in range(FLAGS.max_episode_count):
        action_list = []
        reward_list = []
        loss_list = []
        random_count = 0
        prev_random = 0
        max_count = 0
        prev_max = 0

        e = 1. / ((episode / 10) + 1)   # epsilon
        done = False
        step_count = 0
        state = env.reset()

        e_reward = 0
        model_loss = 0
        avg_reward = np.mean(last_n_game_reward)

        if FLAGS.frame_size > 1:
            state_with_frame = deque(maxlen=FLAGS.frame_size)

            for _ in range(FLAGS.frame_size):
                state_with_frame.append(state)

            state = np.array(state_with_frame)
            state = np.reshape(state, (1, FLAGS.input_size, FLAGS.frame_size))

        while not done:
            if np.random.rand() < e:
                action = env.action_sample()
                random_count +=1
            else:
                # Choose an action by greedily from the Q-network
                action = np.argmax(mainDQN.predict(state))
                max_count += 1

            # Get new state and reward from environment
            next_state, reward, done = env.step(action)

            if FLAGS.frame_size > 1:
                state_with_frame.append(next_state)

                next_state = np.array(state_with_frame)
                next_state = np.reshape(next_state, (1, FLAGS.input_size, FLAGS.frame_size))

            # Save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > FLAGS.batch_size:
                minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                model_loss = loss

                if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                    logger.info(f" - step_count : {step_count}, reward: {e_reward} loss: {loss},  random: {random_count-prev_random}, optimal: {max_count-prev_max}")
                    prev_max = max_count
                    prev_random = random_count
                loss_list.append(loss)
            else :
                loss_list.append(0)

            if step_count % FLAGS.target_update_count == 0:
                sess.run(copy_ops)

            state = next_state
            e_reward += reward
            step_count += 1

            # save model checkpoint
            if global_step % FLAGS.save_step_count == 0:
                checkpoint_path = FLAGS.model_name + "_f" + str(FLAGS.frame_size) + "_" + FLAGS.checkpoint_path + "global_step"
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                saver.save(sess, checkpoint_path, global_step=global_step)
                logger.info(f"save model for global_step: {global_step} ")

            global_step += 1

            action_list.append(action)
            reward_list.append(reward)

            # End of episode
        csv_save(action_list, reward_list, loss_list,
                 "DQN_data_{0}_tc_{1}_{2}_target_{3}".format(FLAGS.model_name, FLAGS.transaction_cost, env.name, FLAGS.target_update_count),
                 "/episode_{0}.csv".format(episode))
        logger.info(
            f"Episode: {episode} \treward: {e_reward}  \tloss: {model_loss}  \tconsecutive_{consecutive_len}_avg_reward: {avg_reward} \tepsilon:{e}" )

        # CartPole-v0 Game Clear Checking Logic
        last_n_game_reward.append(e_reward)

        if len(last_n_game_reward) == last_n_game_reward.maxlen:
            avg_reward = np.mean(last_n_game_reward)
    return saver


# def test(env, saver):
def test(sess, env, mainDQN, targetDQN, saver, online=False):
    logger.info("FLAGS configure.")
    logger.info(FLAGS.__flags)

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)

    # store last games rewards
    consecutive_len = 100 # default value
    last_n_game_reward = deque(maxlen=consecutive_len)

    # Load weights
    checkpoint_path = FLAGS.model_name + "_f" + str(FLAGS.frame_size) + "_" + FLAGS.checkpoint_path
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    # initial copy q_net -> target_net
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    sess.run(copy_ops)

    global_step = 1

    history = []

    action_list = []
    reward_list = []
    loss_list = []

    done = False
    step_count = 0
    state = env.reset()

    e_reward = 0
    model_loss = 0
    avg_reward = np.mean(last_n_game_reward)

    if FLAGS.frame_size > 1:
        state_with_frame = deque(maxlen=FLAGS.frame_size)

        for _ in range(FLAGS.frame_size):
            state_with_frame.append(state)

        state = np.array(state_with_frame)
        state = np.reshape(state, (1, FLAGS.input_size, FLAGS.frame_size))

    while not done:
        action = np.argmax(mainDQN.predict(state))

        # Get new state and reward from environment
        next_state, reward, done = env.step(action)

        if FLAGS.frame_size > 1:
            state_with_frame.append(next_state)

            next_state = np.array(state_with_frame)
            next_state = np.reshape(next_state, (1, FLAGS.input_size, FLAGS.frame_size))

        # Save the experience to our buffer
        if online:
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > FLAGS.batch_size:
                minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                model_loss = loss

                if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                    logger.info(f" - step_count : {step_count}, reward: {e_reward} loss: {loss}")
                loss_list.append(loss)
            else :
                loss_list.append(0)

            if step_count % FLAGS.target_update_count == 0:
                sess.run(copy_ops)
        else:
            if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
                logger.info(f" - step_count : {step_count}, reward: {e_reward}")

        state = next_state
        e_reward += reward
        step_count += 1

        global_step += 1

        action_list.append(action)
        reward_list.append(reward)

        # End of episode
    csv_save(action_list, reward_list, loss_list,
             "DQN_data_{0}_tc_{1}_{2}_target_{3}".format(FLAGS.model_name, FLAGS.transaction_cost, env.name, FLAGS.target_update_count),
             "/episode_{0}.csv".format("Test"))
    logger.info(
        f"Episode: Test \treward: {e_reward}  \tloss: {model_loss}  \tconsecutive_{consecutive_len}_avg_reward: {avg_reward}" )

    # CartPole-v0 Game Clear Checking Logic
    last_n_game_reward.append(e_reward)

    if len(last_n_game_reward) == last_n_game_reward.maxlen:
        avg_reward = np.mean(last_n_game_reward)


if __name__ == "__main__":
    print(tf.__version__)
    if FLAGS.model_name.startswith("MLP") and FLAGS.frame_size > 1:
        raise ValueError('do not support frame_size > 1 if model_name is MLP')

    raw_data = read_from_file('../dataset/chart.csv', length=130000)
    price_diff = raw_data[1:] - raw_data[:-1]
    actions = np.array([-1, 0, 1])

    norm2 = normalize(price_diff[:, np.newaxis], axis=0).ravel()
    env_train = Env2(normalized=norm2[:15000],
                    time_series=(price_diff[:15000], raw_data[:15000]),
                    actions=actions, input_size=FLAGS.input_size-1, transaction_cost=FLAGS.transaction_cost)
    env_test = Env2(normalized=norm2[15000:30000],
                   time_series=(price_diff[15000:30000], raw_data[15000:30000]),
                   actions=actions, input_size=FLAGS.input_size-1, transaction_cost=FLAGS.transaction_cost)


    with tf.Session() as sess:

        mainDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
                      learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
        targetDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
                        frame_size=FLAGS.frame_size, name="target")


        saver = train(sess,env_train, mainDQN, targetDQN)
        # test(env_test, saver)
        saver = tf.train.Saver()
        #test(sess, env_test, mainDQN, targetDQN, saver, online=False)

#
# """
# Double DQN (Nature 2015)
# http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# Notes:
#     The difference is that now there are two DQNs (DQN & Target DQN)
#     y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
#     Loss: (y_i - Q(state, action; 𝜃))^2
#     Every C step, 𝜃_target <- 𝜃
#
# Github: https://github.com/DongjunLee/dqn-tensorflow/blob/master/main.py
# """
#
# from DeepQNet.DQN import *
# from collections import deque
# import logging
# import os
# import random
# from typing import List
#
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# from sklearn.preprocessing import normalize
# #import gym
#
# from DeepQNet.DQN import DQN
# from DeepQNet.environment import Env
#
#
# flags = tf.app.flags
# flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate')
# flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episodes')
# flags.DEFINE_integer('target_update_count', 20, 'DQN Target Network update count')  # default 5
# flags.DEFINE_integer('max_episode_count', 100, 'Number of maximum episodes')
# flags.DEFINE_integer('input_size', 128, 'Input size.')
# flags.DEFINE_integer('output_size', 3, 'output size. (Number of actions)')
# flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divided evenly into the dataset sizes)')
# flags.DEFINE_integer('frame_size', 1, 'Frame size. (Stack env\'s observation T-n ~ T')
# flags.DEFINE_string('model_name', 'MLPv1', 'DeepLearning Network model name. (MLPv1, ConvNetv1, ConvNetv2')
# flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
# flags.DEFINE_boolean('step_verbose', True, 'verbose every step count')
# flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
# flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
# flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')
# flags.DEFINE_float('transaction_cost', 0.0005, 'transaction cost')
#
# FLAGS = flags.FLAGS
#
# logger = logging.getLogger()
# fileHandler = logging.FileHandler('./myLoggerTest.log')
# streamHandler = logging.StreamHandler()
# logger.addHandler(fileHandler)
# logger.addHandler(streamHandler)
# logger.setLevel(logging.INFO)
#
#
# def replay_train(behaviorDQN: DQN, targetDQN: DQN, train_batch: list):
#     """Trains `mainDQN` with target Q values given by `targetDQN`
#     Args:
#         behaviorDQN (DeepQNetwork): Behavior DQN that will be trained
#         targetDQN (DeepQNetwork): Target DQN that will predict Q_target
#         train_batch (list): Minibatch of replay memory
#             Each element is (s, a, r, s', done)
#             [(state, action, reward, next_state, done), ...]
#     Returns:
#         float: After updating `mainDQN`, it returns a `loss`
#     """
#     # print(np.array([x[0] for x in train_batch]).shape)
#     states = np.vstack([x[0] for x in train_batch])
#
#     actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
#     rewards = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])
#     next_states = np.vstack([x[3] for x in train_batch])
#     done = np.array([x[4] for x in train_batch[:FLAGS.batch_size]])
#
#     predict_result = targetDQN.predict(next_states)
#
#     Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1)
#
#     X = states
#     y = behaviorDQN.predict(states)
#     y[np.arange(len(X)), actions] = Q_target
#
#     # Train our network using target and predicted Q values on each episode
#     return behaviorDQN.update(X, y)
#
#
# def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
#     """Creates TF operations that copy weights from `src_scope` to `dest_scope`
#     Args:
#         dest_scope_name (str): Destination weights (copy to)
#         src_scope_name (str): Source weight (copy from)
#     Returns:
#         List[tf.Operation]: Update operations are created and returned
#     """
#     # Copy variables src_scope to dest_scope
#     op_holder = []
#
#     src_vars = tf.get_collection(
#         tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
#     dest_vars = tf.get_collection(
#         tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
#
#     for src_var, dest_var in zip(src_vars, dest_vars):
#         op_holder.append(dest_var.assign(src_var.value()))
#
#     return op_holder
#
#
# def csv_save(actions, rewards, losses, dir, filename):
#     header = ["cum_reward", "rewards", "actions", "losses"]
#     data_dict = dict()
#     data_shape = np.array(rewards).shape
#     rewards_reshaped = np.array(rewards).reshape([-1])
#     cumulative_reward = np.cumsum(rewards_reshaped).reshape(data_shape)
#
#     data_dict[header[0]] = cumulative_reward
#     data_dict[header[1]] = rewards
#     data_dict[header[2]] = actions
#     data_dict[header[3]] = losses
#
#     df = pd.DataFrame(data_dict)
#
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     df.to_csv(dir + filename)
#
# def train(sess, env):
#     logger.info("FLAGS configure.")
#     logger.info(FLAGS.__flags)
#
#     # store the previous observations in replay memory
#     replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
#
#     # store last games rewards
#     consecutive_len = 100 # default value
#     last_n_game_reward = deque(maxlen=consecutive_len)
#
#     with tf.Session() as sess:
#         mainDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
#                                learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
#         targetDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
#                                  frame_size=FLAGS.frame_size, name="target")
#
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver(tf.global_variables())
#
#         # initial copy q_net -> target_net
#         copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
#         sess.run(copy_ops)
#
#         global_step = 1
#
#         history = []
#         for episode in range(FLAGS.max_episode_count):
#             action_list = []
#             reward_list = []
#             loss_list = []
#
#             e = 1. / ((episode / 10) + 1)   # epsilon
#             done = False
#             step_count = 0
#             state = env.reset()
#
#             e_reward = 0
#             model_loss = 0
#             avg_reward = np.mean(last_n_game_reward)
#
#             if FLAGS.frame_size > 1:
#                 state_with_frame = deque(maxlen=FLAGS.frame_size)
#
#                 for _ in range(FLAGS.frame_size):
#                     state_with_frame.append(state)
#
#                 state = np.array(state_with_frame)
#                 state = np.reshape(state, (1, FLAGS.input_size, FLAGS.frame_size))
#
#             while not done:
#                 if np.random.rand() < e:
#                     action = env.action_sample()
#                 else:
#                     # Choose an action by greedily from the Q-network
#                     action = np.argmax(mainDQN.predict(state))
#
#                 # Get new state and reward from environment
#                 next_state, reward, done = env.step(action)
#
#                 if FLAGS.frame_size > 1:
#                     state_with_frame.append(next_state)
#
#                     next_state = np.array(state_with_frame)
#                     next_state = np.reshape(next_state, (1, FLAGS.input_size, FLAGS.frame_size))
#
#                 # Save the experience to our buffer
#                 replay_buffer.append((state, action, reward, next_state, done))
#
#                 if len(replay_buffer) > FLAGS.batch_size:
#                     minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
#                     loss, _ = replay_train(mainDQN, targetDQN, minibatch)
#                     model_loss = loss
#
#                     if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
#                         logger.info(f" - step_count : {step_count}, reward: {e_reward} loss: {loss}")
#                     loss_list.append(loss)
#                 else :
#                     loss_list.append(0)
#
#                 if step_count % FLAGS.target_update_count == 0:
#                     sess.run(copy_ops)
#
#                 state = next_state
#                 e_reward += reward
#                 step_count += 1
#
#                 # save model checkpoint
#                 if global_step % FLAGS.save_step_count == 0:
#                     checkpoint_path = FLAGS.model_name + "_f" + str(FLAGS.frame_size) + "_" + FLAGS.checkpoint_path + "global_step"
#                     if not os.path.exists(checkpoint_path):
#                         os.makedirs(checkpoint_path)
#
#                     saver.save(sess, checkpoint_path, global_step=global_step)
#                     logger.info(f"save model for global_step: {global_step} ")
#
#                 global_step += 1
#
#                 action_list.append(action)
#                 reward_list.append(reward)
#
#                 # End of episode
#             csv_save(action_list, reward_list, loss_list, "DQN_data{0}".format(FLAGS.model_name), "/episode_{0}.csv".format(episode))
#             logger.info(
#                 f"Episode: {episode} \treward: {e_reward}  \tloss: {model_loss}  \tconsecutive_{consecutive_len}_avg_reward: {avg_reward} \tepsilon:{e}" )
#
#             # CartPole-v0 Game Clear Checking Logic
#             last_n_game_reward.append(e_reward)
#
#             if len(last_n_game_reward) == last_n_game_reward.maxlen:
#                 avg_reward = np.mean(last_n_game_reward)
#     return saver
#
#
# # def test(env, saver):
# def test(sess, env):
#     logger.info("FLAGS configure.")
#     logger.info(FLAGS.__flags)
#
#     # store the previous observations in replay memory
#     replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
#
#     # store last games rewards
#     consecutive_len = 100 # default value
#     last_n_game_reward = deque(maxlen=consecutive_len)
#
#     with tf.Session() as sess:
#         mainDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
#                                learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
#         targetDQN = DQN(sess, FLAGS.model_name, FLAGS.input_size, FLAGS.output_size,
#                                  frame_size=FLAGS.frame_size, name="target")
#
#         saver = tf.train.Saver()
#         checkpoint_path = FLAGS.model_name + "_f" + str(FLAGS.frame_size) + "_" + FLAGS.checkpoint_path
#         saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
#
#         # initial copy q_net -> target_net
#         copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
#         sess.run(copy_ops)
#
#         global_step = 1
#
#         history = []
#
#         action_list = []
#         reward_list = []
#         loss_list = []
#
#         done = False
#         step_count = 0
#         state = env.reset()
#
#         e_reward = 0
#         model_loss = 0
#         avg_reward = np.mean(last_n_game_reward)
#
#         if FLAGS.frame_size > 1:
#             state_with_frame = deque(maxlen=FLAGS.frame_size)
#
#             for _ in range(FLAGS.frame_size):
#                 state_with_frame.append(state)
#
#             state = np.array(state_with_frame)
#             state = np.reshape(state, (1, FLAGS.input_size, FLAGS.frame_size))
#
#         while not done:
#             action = np.argmax(mainDQN.predict(state))
#
#             # Get new state and reward from environment
#             next_state, reward, done = env.step(action)
#
#             if FLAGS.frame_size > 1:
#                 state_with_frame.append(next_state)
#
#                 next_state = np.array(state_with_frame)
#                 next_state = np.reshape(next_state, (1, FLAGS.input_size, FLAGS.frame_size))
#
#             # Save the experience to our buffer
#             replay_buffer.append((state, action, reward, next_state, done))
#
#             if len(replay_buffer) > FLAGS.batch_size:
#                 minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
#                 loss, _ = replay_train(mainDQN, targetDQN, minibatch)
#                 model_loss = loss
#
#                 if FLAGS.step_verbose and step_count % FLAGS.step_verbose_count == 0:
#                     logger.info(f" - step_count : {step_count}, reward: {e_reward} loss: {loss}")
#                 loss_list.append(loss)
#             else :
#                 loss_list.append(0)
#
#             if step_count % FLAGS.target_update_count == 0:
#                 sess.run(copy_ops)
#
#             state = next_state
#             e_reward += reward
#             step_count += 1
#
#             global_step += 1
#
#             action_list.append(action)
#             reward_list.append(reward)
#
#             # End of episode
#         csv_save(action_list, reward_list, loss_list,
#                  "DQN_data_{0}_tc_{1}".format(FLAGS.model_name, FLAGS.transaction_cost),
#                  "/episode_{0}.csv".format("Test"))
#         logger.info(
#             f"Episode: Test \treward: {e_reward}  \tloss: {model_loss}  \tconsecutive_{consecutive_len}_avg_reward: {avg_reward}" )
#
#         # CartPole-v0 Game Clear Checking Logic
#         last_n_game_reward.append(e_reward)
#
#         if len(last_n_game_reward) == last_n_game_reward.maxlen:
#             avg_reward = np.mean(last_n_game_reward)
#
#
# if __name__ == "__main__":
#     if FLAGS.model_name.startswith("MLP") and FLAGS.frame_size > 1:
#         raise ValueError('do not support frame_size > 1 if model_name is MLP')
#
#     raw_data = read_from_file('../dataset/chart.csv', length=130000)
#     price_diff = raw_data[1:] - raw_data[:-1]
#     actions = np.array([-1, 0, 1])
#     window_size = 20000
#     slide = 5000
#     norm2 = normalize(price_diff[:, np.newaxis], axis=0).ravel()
#     env_train = Env(normalized=norm2[:1000],
#                     time_series=(price_diff[:1000], raw_data[:1000]),
#                     actions=actions, input_size=FLAGS.input_size)
#     env_test = Env(normalized=norm2[100000:],
#                    time_series=(price_diff[100000:], raw_data[100000:]),
#                    actions=actions, input_size=FLAGS.input_size)
#
#     saver = train(env_train)
#     # test(env_test, saver)
#     test(env_train)