from models import *
import numpy as np


class DQN:
    def __init__(self, session, input_size, seq_size, output_size, learning_rate, model_name, net_name):
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
        self.seq_size = seq_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.net_name = net_name
        self.x = None
        self.model_name = model_name

        self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            if self.model_name == "MLPv1":
                self.x = tf.placeholder(tf.float32, [None, self.input_size], name="Input_x")
            else:
                self.x = tf.placeholder(tf.float32, [None, self.seq_size, self.input_size, 1], name="Input_x")
            model = select_model(self.model_name, self.x, self.output_size, self.learning_rate)
            model.build_network()

            self.Qpred = model.pred  # [batch_size, 3]

            # Mutable type's assignment is passing reference
            self.y = model.y  # self.y is alias of model.y
            # print(id(self.y), id(model.y))  # check the ids are equal
            # self.x = model.x is inefficient as it needs passing the shape of x

            self.loss = model.loss
            self.train = model.train

    def predict(self, state):
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        # In replay_train(), batch is used
        if self.model_name == "MLPv1":
            x = np.reshape(state, [-1, self.input_size])
        else:
            x = np.reshape(state, [-1, self.seq_size, self.input_size, 1])
        return self.session.run(self.Qpred, feed_dict={self.x: x})

    def greedy_action(self, state):
        # state is always [S, D]
        return np.argmax(self.predict(state))

    def update(self, x_stack, y_stack):
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        if self.model_name == "MLPv1":
            x_stack = np.reshape(x_stack, [-1, self.input_size])
        else:
            x_stack = np.reshape(x_stack, [-1, self.seq_size, self.input_size, 1])
        return self.session.run([self.loss, self.train], feed_dict={self.x: x_stack, self.y: y_stack})
