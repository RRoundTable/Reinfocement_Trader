import tensorflow as tf
import os, shutil

import RRL
from old import config
import read_data


def main(unused_args):
    model_dir = 'model/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    else:
        shutil.rmtree('model', ignore_errors=True)
    data_dir = 'data/chart.csv'
    run_config = tf.estimator.RunConfig(model_dir=model_dir)
    conf = config.Config()

    train_idx = 0
    train_window = 3000
    test_window = 250

    raw_data = read_data.read_csv(data_dir)
    train_data = read_data.slide_window(raw_data, train_idx, train_window)
    test_data = read_data.slide_window(raw_data, train_idx + train_window, test_window)

    train_x, train_y = read_data.create_dataset(train_data, conf.input_num, conf.num_step)
    test_x, test_y = read_data.create_dataset(test_data, conf.input_num, 1)

    model = RRL.create_model(conf, run_config)
    RRL.train_model(model, train_x, train_y, conf)
    actions, rewards = RRL.test_model(model, test_x, test_y, conf)

    print(rewards[:,-1])

if __name__ == "__main__":
    tf.app.run(main=main)
