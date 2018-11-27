import numpy as np
import pandas as pd


def read_from_file(filepath, length):
    print("DeepQNet data reader")

    df = pd.read_csv(filepath, index_col=0)
    end_df = df['end']
    return np.array(df[:length], dtype=np.float32), np.array(end_df[:length], dtype=np.float32)


def generate_batch(input_data, input_size=50, batch_size=0):
    index = len(input_data) // input_size * input_size

    data = input_data[:index]
    temp = np.vstack([data[i: -input_size + i] for i in range(input_size)])

    X_data = temp.T
    y_data = np.array(data[input_size :], dtype=np.float32)

    data_len = X_data.shape[0]
    batch_num = data_len // batch_size
    for i in range(batch_num):
        x = X_data[i * batch_size : (i+1) * batch_size]
        y = y_data[i: i + batch_size]
        yield (x, y)


def generate_epoch(raw_data, epoch_num=100, batch_size=50, input_size=20, history=False):
    if history:
        for i in range(epoch_num):
            yield generate_batch_with_history(raw_data, input_size, batch_size)
    else:
        for i in range(epoch_num):
            yield generate_batch(raw_data, input_size, batch_size)
