import numpy as np
#import pymysql
import matplotlib.pyplot as plt
import os

def read_from_file(filepath, length):
    print("DeepQNet data reader")
    with open(filepath, "r") as file:
        data = file.read().split("\n")
    return np.array(data[:length], dtype=np.float32)



def generate_batch_with_history(input_data, input_size=50, batch_num=0):
    # raw_data = np.array(raw_data, dtype=np.float32).reshape([-1])
    current_size = input_size - 5
    index = len(input_data[0])//current_size * current_size

    data = input_data[0][:index]
    temp = np.vstack([data[i: -current_size + i] for i in range(current_size)])
    data_3h = input_data[1][:temp.shape[1]]
    data_5h = input_data[2][:temp.shape[1]]
    data_10h = input_data[3][:temp.shape[1]]
    data_1d = input_data[4][:temp.shape[1]]
    data_3d = input_data[5][:temp.shape[1]]

    X_data = np.vstack([temp, data_3h, data_5h, data_10h, data_1d, data_3d]).T
    y_data = np.array(data[current_size :], dtype=np.float32)

    data_len = X_data.shape[0]
    batch_size = data_len // batch_num
    for i in range(batch_num):
        x = X_data[i * batch_size : (i+1) * batch_size]
        y = y_data[i: i + batch_size]
        yield (x, y)

def generate_batch(input_data, input_size=50, batch_size=0):
    # raw_data = np.array(raw_data, dtype=np.float32).reshape([-1])
    index = len(input_data)//input_size * input_size

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
