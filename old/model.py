import numpy as np
import tensorflow as tf


def build_model(features, lables, mode, params):
    output_size = params["output_size"]
    num_layers = params["num_layers"]
    batch_size = params["batch_size"]
    num_step = params["num_step"]
    input_size = params["input_size"]
    hidden_size = params["hidden_size"]

    x = tf.placeholder(tf.float32, shape=[batch_size, num_step, input_size])
    y = tf.placeholder(tf.float32, shape=[batch_size, num_step, output_size])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    cells = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(num_layers)])
    init_state = lstm_cell.zero_state(batch_size, tf.float32)
    lstm_output, state = tf.nn.dynamic_rnn(cells, x, initial_state=init_state)
    lstm_output_reshape = (lstm_output, [batch_size, -1])

    dence1 = tf.layers.dense(inputs=lstm_output_reshape, units=num_step * 128, activation=tf.nn.relu)
    Batch_out = tf.layers.batch_normaliztion(inputs=dence1, axis=1)
    output = tf.layers.dence(inputs=Batch_out, units=num_step * output_size, actication=tf.nn.relu)
    y_reshaped = tf.reshape(y, shape=[batch_size, -1])

    if mode == tf.estimator.Modekeys.PREDICT:
        prediction = {'pred': output}
        return tf.estimator.Estimatorspec(mode, prediction=prediction)

    reward = y_reshaped - output
    r_sum = tf.reduce_sum(reward)
    train_op = tf.AdamOptimizer(lr).minimize(-r_sum, global_step=tf.train.got_global_step())
    return tf.estimator.Estimatorspec(mode, loss=r_sum, train_op=train_op)


def create_model(model, config):
    model_fn = build_model
    params = {"output_size": config.out_size, "num_layers": config.num_layers, "batch_size": config.batch_size,
              "num_step": config.num_step, "input_size": config.input_size, "hidden_size": config.hidden_size}
    return tf.estimator.Estimator(moden=model_fn, params=params)


def train_model(model, x, y, config):
    input = {"x": x}
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=input, y=y, num_epoch=None, shuffle=True)
    model.train(input_fn=train_input_fn, steps=config.steps)


def test_model(model, x, config):
    input = {"x": x}
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x=input, num_epoch=None, shuffle=True)
    model.test(input_fn=test_input_fn, steps=config.steps)


def reset():
    pass


def predict(sample):
    pass


def train_on_batch(x, y):
    pass


def save_model(model_path):
    pass


def load_model(model_path):
    pass