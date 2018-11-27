import tensorflow as tf
import numpy as np
import random
from DeepQNet.readData import *
from DeepQNet.models import *

epsilon = 1
epsilon_min = 0.001
actions = [-1, 0, 1]
num_action = len(actions)
epoch = 100
memory_size = 5000
discount = 0.97
learning_rate = 0.005

class DQN:
    def __init__(self, session: tf.Session, model_name: str, input_size: int, output_size: int,
                 learning_rate: float=0.0001, frame_size: int=1, name: str="main") -> None:
        """DeepQNet Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.frame_size = frame_size

        self.net_name = name
        self.learning_rate = learning_rate

        self._build_network(model_name=model_name)

    def _build_network(self, model_name="MLPv1") -> None:
        with tf.variable_scope(self.net_name):
            if self.frame_size > 1:
                X_shape = [None] + [self.input_size] + [self.frame_size]
            else:
                X_shape = [None] + [self.input_size]

            self._X = tf.placeholder(tf.float32, X_shape, name="Input_X")

            models = {
                "MLPv1": MLPv1,
                "ConvNetv1": ConvNetv1,
                "ConvNetv1withAction": ConvNetv1_with_action,
                "ConvNetv2": ConvNetv2,
                "ConvNetv3": ConvNetv3
            }

            model = models[model_name](self._X, self.output_size,
                                       frame_size=self.frame_size, learning_rate=self.learning_rate)
            model.build_network()

            self._Qpred = model.inference
            self._Y = model.Y
            self._loss = model.loss
            self._train = model.optimizer

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """

        if self.frame_size > 1:
            x_shape = [-1] + [self.input_size] + [self.frame_size]
        else:
            x_shape = [-1] + [self.input_size]
        x = np.reshape(state, x_shape)
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self._X : x_stack,
            self._Y : y_stack
        }
        return self.session.run([self._loss, self._train], feed)

#
# class replay_memory:
#     def __init__(self, max_size=memory_size):
#         self.buffer = list()
#         self.max_size = max_size
#
#     def remember(self, state, action, reward, next_state):
#         if len(self.buffer) >= self.max_size:
#             self.buffer.pop(0)      # 메모리 초과시 오래된 걸 지움
#         self.buffer.append([state, action, reward, next_state])
#
#     def generate_batch(self, size, num_action=num_action, actions=actions):
#         buffer_size = len(self.buffer)
#         batch_num = buffer_size // size
#         batch_size = size
#
#         if buffer_size > size:
#             batch_num = 1
#             batch_size = buffer_size
#
#         randomized_buf = random.sample(self.buffer, buffer_size)
#         for i in range(batch_num):
#             yield np.array(randomized_buf[i * batch_size: (i+1) * batch_size], dtype=np.float32)
#

#
# class DeepQNet:
#     def __init__(self, actions=actions, epsilon=epsilon, epsilon_min=epsilon_min,
#                  max_size=memory_size, input_size=100, learning_rate=learning_rate, discount=discount):
#         self.buffer = replay_memory(max_size)
#         self.state = DNN(input_size=input_size, learning_rate=learning_rate, discount=discount)
#         self.actions = actions
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#
#     def learn(self):
