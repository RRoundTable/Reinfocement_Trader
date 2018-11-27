import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def build_model(features, labels, mode, params):
    output_size = params["output_size"]
    num_layers = params["num_layers"]
    num_step = params["num_step"] if mode == tf.estimator.ModeKeys.TRAIN else 1
    input_size = params["input_size"]
    input_num = params['input_num']
    hidden_size = params["hidden_size"]
    lr = params["learning_rate"]

    x = tf.reshape(features['x'], [-1, num_step, input_size * input_num])
    batch_size = x.shape[0]

    x_norm = tf.layers.batch_normalization(x, axis=1, training=mode == tf.estimator.ModeKeys.TRAIN)
    y = tf.reshape(labels, [batch_size, num_step, 1])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, activation=tf.tanh)
    output_cell = tf.nn.rnn_cell.BasicLSTMCell(output_size, activation=tf.tanh)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)

    cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_layers)] + [output_cell])

    print(x.shape)

    state = cells.zero_state(batch_size, tf.float64)
    lstm_output, state = tf.nn.dynamic_rnn(cell=cells, inputs=x_norm, initial_state=state)
    output = tf.reshape(lstm_output, [batch_size, num_step])

    # dence1 = tf.layers.dense(inputs=lstm_output_reshape, units=num_step * 128, activation=tf.nn.relu)
    # bnorm_out = tf.layers.batch_normalization(inputs=dence1, axis=1)
    # output = tf.layers.dense(inputs=bnorm_out, units=num_step * output_size, actication=tf.nn.relu)
    y_reshaped = tf.reshape(y, shape=[batch_size, -1])
    reward = tf.reduce_sum(output * y_reshaped, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = {'pred': output, 'reward': reward}
        return tf.estimator.EstimatorSpec(mode, predictions=prediction)

    r_sum = tf.reduce_sum(reward)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(-r_sum, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=r_sum, train_op=train_op)


def create_model(config, run_config):
    model_fn = build_model
    params = {"output_size": config.output_size,
              "num_layers": config.num_layers,
              "batch_size": config.batch_size,
              "num_step": config.num_step,
              "input_size": config.input_size,
              'input_num' :config.input_num,
              "hidden_size": config.hidden_size,
              "learning_rate": config.lr
              }
    return tf.estimator.Estimator(model_fn=model_fn, params=params, config=run_config)


def train_model(model, x, y, config):
    inputs = {"x": x}
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=inputs, y=y,
                                                        batch_size=config.batch_size,
                                                        num_epochs=None,
                                                        shuffle=True)
    model.train(input_fn=train_input_fn, steps=config.steps)


def test_model(model, x, y, config):
    inputs = {"x": x}
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x=inputs, y=y,
                                                       batch_size=1,
                                                       num_epochs=None,
                                                       shuffle=False)
    pred = model.predict(input_fn=test_input_fn)

    pred_action = []
    rewards = []

    for i , p in pred:
        pred_action.append(p['pred'])
        rewards.append(p['reward'])

    pred_action = np.asarray(pred_action)
    rewards = np.asarray(rewards)

    return pred_action, rewards

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
