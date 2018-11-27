import tensorflow as tf


def select_model(model_name, x, output_size, learning_rate):
    models = {
        "MLPv1": MLPv1,
        "ConvNetv1": ConvNetv1,
        "ConvNetv2": ConvNetv2
    }

    return models[model_name](x, output_size, learning_rate)


# class MLPv1:
#     def __init__(self, x, num_classes, learning_rate=0.001):
#         self.num_classes = num_classes
#         self.learning_rate = learning_rate
#         self.x = x
#         self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
#
#         self.pred = None
#         self.loss = None
#         self.train = None
#
#     def build_network(self):
#         fc1 = tf.layers.dense(self.x, 128, activation=tf.nn.relu)
#         fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
#         fc3 = tf.layers.dense(fc2, 32, activation=tf.nn.relu)
#         output = tf.layers.dense(fc3, self.num_classes)
#
#         self.pred = output  # [batch_size, output_size]
#         self.loss = tf.losses.mean_squared_error(self.y, self.pred)
#
#         self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class MLPv1:
    def __init__(self, x, num_classes, learning_rate=0.001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.x = x
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        self.pred = None
        self.loss = None
        self.train = None

    def build_network(self):
        nnum_1 = 16
        nnum_2 = 8

        fc1 = tf.layers.dense(self.x, nnum_1, activation=tf.nn.tanh)
        fc2 = tf.layers.dense(fc1, nnum_2, activation=tf.nn.tanh)
        fc3 = tf.layers.dense(fc2, self.num_classes)
        output = fc3

        self.pred = output  # [batch_size, output_size]
        self.loss = tf.losses.mean_squared_error(self.y, self.pred)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


# class ConvNetv1:
#     def __init__(self, x, num_classes, learning_rate=0.001):
#         self.num_classes = num_classes
#         self.learning_rate = learning_rate
#         self.x = x
#         self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
# 
#         self.pred = None
#         self.loss = None
#         self.train = None
# 
#         conv1 = tf.layers.conv2d(self.x, 32, kernel_size=3, padding="same", activation=tf.nn.relu)
#         pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
# 
#         conv2 = tf.layers.conv2d(pool1, 64, kernel_size=3, padding="same", activation=tf.nn.relu)
#         pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
# 
#         conv3 = tf.layers.conv2d(pool2, 128, kernel_size=3, padding="same", activation=tf.nn.relu)
#         pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2)
#         pool3_flat = tf.layers.flatten(pool3)  # tf.reshape(pool3, [-1, 16 * 128])
# 
#         fc4 = tf.layers.dense(pool3_flat, 512)
#         fc5 = tf.layers.dense(fc4, 128)
#         output = tf.layers.dense(fc5, self.num_classes)
# 
#         self.pred = output  # [batch_size, output_size]
#         self.loss = tf.losses.mean_squared_error(self.y, self.pred)
# 
#     def build_network(self):
#         self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class ConvNetv1:
    def __init__(self, x, num_classes, learning_rate=0.001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.x = x
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        self.pred = None
        self.loss = None
        self.train = None

    def build_network(self):
        ksize_1 = 3
        knum_1 = 8
        nnode_2 = 8

        conv1 = tf.layers.conv2d(self.x, knum_1, kernel_size=ksize_1, padding="same", activation=tf.nn.tanh)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
        pool1_flat = tf.layers.flatten(pool1)

        fc2 = tf.layers.dense(pool1_flat, nnode_2, activation=tf.nn.tanh)
        fc3 = tf.layers.dense(fc2, self.num_classes)
        output = fc3

        self.pred = output  # [batch_size, output_size]
        self.loss = tf.losses.mean_squared_error(self.y, self.pred)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

class ConvNetv2:
    def __init__(self, x, num_classes, learning_rate=0.001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.x = x
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        self.pred = None
        self.loss = None
        self.train = None

    def build_network(self):
        # same -> valid
        # pooling: X
        # kernel_size: 3 -> (3, D), (5, D), (7, D) and 3, 5, 7
        # decrease number of parameters (D)
        # kernel: 8, 16, 32
        # Node: 16, 32, 64, 128
        # Conv2d -> Conv1d -> fc -> fc

        S, D = self.x.shape[1], self.x.shape[2]

        ksize_1 = (3, D)
        knum_1 = 8
        ksize_2 = 3
        knum_2 = 16
        nnum_3 = 8

        # New model
        conv1 = tf.layers.conv2d(self.x, knum_1, kernel_size=ksize_1, padding="valid", activation=tf.nn.tanh)
        conv1_reshape = tf.reshape(conv1, [-1, S - ksize_1[0] + 1, knum_1])  # squeeze operation

        conv2 = tf.layers.conv1d(conv1_reshape, knum_2, kernel_size=ksize_2, padding="valid", activation=tf.nn.tanh)
        conv2_flatten = tf.layers.flatten(conv2)

        fc3 = tf.layers.dense(conv2_flatten, nnum_3)
        fc4 = tf.layers.dense(fc3, self.num_classes)
        output = fc4

        self.pred = output  # [batch_size, output_size]
        self.loss = tf.losses.mean_squared_error(self.y, self.pred)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
