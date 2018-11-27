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

import logging
import random
import os
import tensorflow as tf
import pandas as pd
from DQN import DQN
from collections import deque
import time

from environment import Env
from readData import *


# Switch Constants
list_S_MODE = ["train", "test"]
list_S_MODEL = ["MLPv1", "ConvNetv1", "ConvNetv2"]
list_S_STOCK = ["doosan", "hynix", "lg", "samsung"]

S_STOCK = list_S_STOCK[0]
S_MODEL = list_S_MODEL[1]
S_MODE = list_S_MODE[0]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', S_MODEL, 'Q-Network model name.')
flags.DEFINE_integer('raw_data_length', 6836, 'Number of rows of raw data from csv file')  # R (160212 or 160094)
flags.DEFINE_integer('seq_size', 20 if FLAGS.model_name != "MLPv1" else 1, 'Sequence size. (stack env\'s observation T-n ~ T)')  # S
FLAGS.raw_data_length = FLAGS.raw_data_length // (3 * FLAGS.seq_size) * (3 * FLAGS.seq_size) if FLAGS.model_name != "MLPv1" else FLAGS.raw_data_length  # R % 3S == 0
flags.DEFINE_integer('env_train_length', FLAGS.raw_data_length // 3 * 2, 'Number of rows of raw data used for training')  # T  must % S == 0
flags.DEFINE_integer('target_update_count', 20, 'DQN Target Network update count')  # U
flags.DEFINE_integer('num_epoch', 50, 'Number of train episodes')  # E  1000: overfitting
flags.DEFINE_integer('input_size', 13, 'Input dimension')   # D (start, high, low, end[5], amount[5])
flags.DEFINE_integer('output_size', 3, 'Number of actions')  # C
flags.DEFINE_integer('replay_memory_length', 200, 'Number of replay memory episodes')  # M is equal to final timestep + 1
flags.DEFINE_integer('batch_size', FLAGS.replay_memory_length - 1, 'Batch size (Must divided evenly into the dataset sizes)')  # B  is OK (B = M - 1)?
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')  # lr
flags.DEFINE_boolean('step_verbose', True, 'Verbose every step count')
flags.DEFINE_integer('step_verbose_count', 1000, 'Verbose step count')
flags.DEFINE_integer('save_episode_count', 1, 'Model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'Model save checkpoint_path')
flags.DEFINE_float('transaction_cost', 0.0, 'Transaction cost')  # tc
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate')  # g

####################################
"""
* Abbreviation List *
- R: raw_data_length
- T: env_train_length
- M: replay_memory_length
- E: num_epoch
- B: batch_size
- S: seq_size
- D: input_size
- C: output_size
- TC: transaction_cost
- LR: learning_rate
"""

R = FLAGS.raw_data_length
T = FLAGS.env_train_length
M = FLAGS.replay_memory_length
E = FLAGS.num_epoch
B = FLAGS.batch_size
S = FLAGS.seq_size
D = FLAGS.input_size
C = FLAGS.output_size
TC = FLAGS.transaction_cost
LR = FLAGS.learning_rate

####################################

logger = logging.getLogger()
fileHandler = logging.FileHandler('./myLoggerTest.log')
streamHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)


def replay_train(behaviorDQN, targetDQN, train_batch):
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

    states = np.array([x[0] for x in train_batch])   # [batch_size, seq_size, input_size]
    actions = np.array([x[1] for x in train_batch])  # [batch_size]
    rewards = np.array([x[2] for x in train_batch])  # [batch_size]
    next_states = np.array([x[3] for x in train_batch])  # [batch_size, seq_size, input_size]

    assert states.shape == (B, S, D) and\
        actions.shape == (B,) and\
        rewards.shape == (B,) and\
        next_states.shape == (B, S, D)

    next_Q = targetDQN.predict(next_states)  # [batch_size, output_size]
    assert next_Q.shape == (B, C)

    # target_Q = r_t + γ * max Q_{t+1}
    target_Q = rewards + FLAGS.discount_rate * np.max(next_Q, axis=1)
    assert target_Q.shape == (B,)

    x = states  # [batch_size, seq_len, input_size]
    target = behaviorDQN.predict(states)  # [batch_size, output_size]
    assert target.shape == (B, C)

    # for a, q in zip(actions, Q_target):
    #     print('a:', a, ' q:', q)

    # Batch update
    for i, (a, q) in enumerate(zip(actions, target_Q)):
        target[i, a] = q
    assert target.shape == (B, C)

    # Train our network using target and predicted Q values on each episode
    return behaviorDQN.update(x, target)


def get_copy_var_ops(dest_scope_name, src_scope_name):
    """
    Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def csv_save(actions, rewards, cumulated_reward, dir, filename, df):
    data_dict = dict()
    data_dict['action'] = actions
    data_dict['reward'] = rewards
    data_dict['cumulated_reward'] = cumulated_reward

    data_frame = pd.DataFrame(data_dict)

    df['cumulated_reward'] = data_frame['cumulated_reward']
    df['reward'] = data_frame['reward']
    df['action'] = data_frame['action']

    df = df.set_index('date')

    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_csv(dir + '/' + filename + '.csv')


def train(sess, env, mainDQN, targetDQN):
    ############################
    # 1. Initialize settings
    ############################

    logger.info("FLAGS configure.")
    logger.info(FLAGS.flag_values_dict())

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    # initial copy mainDQN to targetDQN
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    sess.run(copy_ops)

    ############################
    # 2. For loop for num_epoch episodes ends
    ############################

    greedy_cum_rewards = []
    global_timestep = 0  # step throughout the episodes
    for epoch in np.arange(FLAGS.num_epoch + 1):
        start_time = time.time()

        # Episode starts!
        random_action_cnt, prev_random_action, greedy_action_cnt, prev_greedy_action = 0, 0, 0, 0
        cumulated_reward, loss = 0, 0

        e = 1 / ((epoch / 10) + 1)  # epsilon decaying
        done = False  # isTerminalState

        timestep = S - 1  # t
        state = env.reset()  # initial state
        assert state.shape == (S, D)

        ############################
        # 3. While loop for one episode ends
        ############################
        
        cumulated_reward = 0
        loss_list = []
        action_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []
        reward_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []
        cumulated_reward_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []

        while not done:
            # epsilon greedy
            if np.random.rand() < e:
                action = env.action_sample()
                assert isinstance(action, int)
                random_action_cnt += 1
            else:
                # Choose an action by greedily from the Q-network
                action = mainDQN.greedy_action(state)  # [S, 13] → Scalar action
                assert isinstance(action, np.int64)
                greedy_action_cnt += 1

            # At current state, action 'action' and get next state and reward from environment
            next_state, reward, done = env.step(action)  # [S, 13], scalar, bool
            assert next_state.shape == (S, D) and isinstance(reward, np.float64) and isinstance(done, bool)

            # Save the experience to our buffer
            entry = (state, action, reward, next_state, done)  # (s_t, a_t, r_t, s_{t+1}, s_{t+1}isTerminalState)
            replay_buffer.append(entry)

            if len(replay_buffer) > FLAGS.batch_size:
                # Significance of replay buffer is that
                # it's learned not sequentially one by one, but at once using multiple data.

                minibatch = random.sample(replay_buffer, FLAGS.batch_size)  # [B, 5] ('list' type)
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if FLAGS.step_verbose and global_timestep % FLAGS.step_verbose_count == 0:
                    logger.info(
                        f"- timestep: {timestep:4} \t cumulated_reward: {cumulated_reward:8} \t loss: {loss:20} \t"
                        f" random_action_cnt: {random_action_cnt:3} \t greed_action_cnt: {greedy_action_cnt:3}")

                    # prev_greedy_action = greedy_action_cnt
                    # prev_random_action = random_action_cnt
                loss_list.append(loss)
            else:
                loss_list.append(0)  # train didn't happen

            if global_timestep % FLAGS.target_update_count == 0:
                sess.run(copy_ops)

            state = next_state

            # timestep++
            timestep += 1
            global_timestep += 1

            if done is False:
                action_list.append(action - 1)
                reward_list.append(reward)
                cumulated_reward += reward  # discount_rate is not included?
                cumulated_reward_list.append(cumulated_reward)
            else:
                action_list.append(0)
                reward_list.append(0)
                cumulated_reward_list.append(cumulated_reward)

        print(f"<Processing time per one episode: {time.time() - start_time:4}s>")

        # An episode was ended
        # Save model checkpoint
        if epoch % FLAGS.save_episode_count == 0:
            checkpoint_path = \
                FLAGS.model_name + "_s" + str(FLAGS.seq_size) + "_" + FLAGS.checkpoint_path + "global_timestep"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)

            saver.save(sess, checkpoint_path, global_step=epoch)
            logger.info(f"save model for episode: {epoch}")

        logger.info(
            f"Episode: {epoch:4} \tcumulated_reward: {cumulated_reward:7} \tloss: {loss:20} \t"
            f"epsilon:{e:20} \n")

        if epoch == 0:
            # epoch 0: No training
            greedy_cum_rewards.append(cumulated_reward)
        else:
            greedy_cum_reward = test(sess, env_train, mainDQN, saver, df, make_csv=False)
            greedy_cum_rewards.append(greedy_cum_reward)

        if epoch % 10 == 0:
            csv_save(action_list, reward_list, cumulated_reward_list,
                     "training_csv", f"{S_STOCK}-{S_MODEL}-{S_MODE}-{epoch}", df)

    greedy_cum_reward_df = pd.DataFrame(columns=['cumulated_reward'])
    greedy_cum_reward_df['cumulated_reward'] = greedy_cum_rewards
    greedy_cum_reward_df.index.name = 'epoch'
    greedy_cum_reward_df.to_csv(f"training_csv/{S_STOCK}-{S_MODEL}-{S_MODE}-{FLAGS.num_epoch}-greedy.csv")

    return saver


# Only one episode
def test(sess, env, mainDQN, saver, df, make_csv):
    ############################
    # 1. Initialize settings
    ############################
    sess.run(tf.global_variables_initializer())

    # Load weights
    checkpoint_path = FLAGS.model_name + "_s" + str(FLAGS.seq_size) + "_" + FLAGS.checkpoint_path
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    last_step = tf.train.latest_checkpoint(checkpoint_path).split('-')[1]

    state = env.reset()
    done = False

    cumulated_reward = 0
    action_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []
    reward_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []
    cumulated_reward_list = [-2 for _ in range(S - 1)] if S_MODEL != "MLPv1" else []

    while not done:
        action = mainDQN.greedy_action(state)
        next_state, reward, done = env.step(action)

        if done is False:
            action_list.append(action - 1)
            reward_list.append(reward)
            cumulated_reward += reward
            cumulated_reward_list.append(cumulated_reward)
            state = next_state
        else:
            action_list.append(0)
            reward_list.append(0)
            cumulated_reward_list.append(cumulated_reward)

    if make_csv is True:
        csv_save(action_list, reward_list, cumulated_reward_list, "result_csv", f"{S_STOCK}-{S_MODEL}-{S_MODE}-{last_step}", df)

    print("Result:", cumulated_reward)

    return cumulated_reward


def makeDataFrame():
    col = ['date', 'price', 'next_price', 'diff_price', 'action', 'reward', 'cumulated_reward']
    df = pd.DataFrame(columns=col)
    df_stock = pd.read_csv("dataset/day/" + S_STOCK + '.csv')

    if S_MODE == "train":
        df_stock = df_stock[:T]
    else:
        df_stock = df_stock[T:]
    df_stock = df_stock.reset_index()
    
    df['date'] = df_stock['date']
    df['price'] = df_stock['end']
    next_price = df_stock['end'][1:]
    next_price = next_price.append(next_price[-1:], ignore_index=True)
    df['next_price'] = next_price
    df['diff_price'] = df['next_price'] - df['price']

    return df


if __name__ == "__main__":
    print("tensorflow version:", tf.__version__)

    df = makeDataFrame()

    # raw_data: D features data with raw_data_length samples
    # end_data: only 'end' feature data with raw_data_length samples
    # if D == 5:
    #     raw_data, end_data = read_from_file('dataset/chart.csv', length=R)
    # elif D == 13:
    #     raw_data, end_data = read_from_file('dataset/chart_ma.csv', length=R)

    raw_data, end_data = read_from_file('dataset/day/' + S_STOCK + '.csv', length=R)
    assert raw_data.shape == (R, D) and end_data.shape == (R,)
    actions = np.array([-1, 0, 1])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Make seperate DQN (mainDQN = behaviorDQN, targetDQN)
        mainDQN = DQN(sess, D, S, C, LR, FLAGS.model_name, net_name="main")
        targetDQN = DQN(sess, D, S, C, LR, FLAGS.model_name, net_name="target")

        if S_MODE == "train":
            env_train = Env(num_data=(0, T), raw_data=raw_data, end_data=end_data, actions=actions,
                            input_size=D, seq_size=S, name="train", transaction_cost=TC)
            saver = train(sess, env_train, mainDQN, targetDQN)
            saver = tf.train.Saver()
            test(sess, env_train, mainDQN, saver, df, make_csv=True)
        elif S_MODE == "test":
            env_test = Env(num_data=(T, FLAGS.raw_data_length), raw_data=raw_data, end_data=end_data, actions=actions,
                           input_size=D, seq_size=S, name="train", transaction_cost=TC)
            saver = tf.train.Saver()
            test(sess, env_test, mainDQN, saver, df, make_csv=True)
