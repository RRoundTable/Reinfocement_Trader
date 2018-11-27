import pandas as pd
import numpy as np

file_path = 'data/chart.csv'
train_start_date = '2000-01-01'
test_start_date = ''


def read_csv(file_path):
    df = pd.read_csv(file_path, index_col='date')
    df.drop_duplicates(inplace=True)
    df.to_csv(file_path)
    df['end_diff'] = df['end'].diff().fillna(0)
    return df


def slide_window(raw_df, start_idx, window_size):
    return raw_df.iloc[start_idx:start_idx + window_size, :]


def create_dataset(windowed_df, input_num, num_step):
    window_size, input_size = windowed_df.shape
    input_size -= 1
    xs = []
    ys = []

    for i in range(window_size - (input_num + num_step + 1)):
        input = windowed_df.iloc[i: i + input_num + num_step, :].values
        x_list = []
        for j in range(num_step):
            x_list.append(input[j:j + input_num, :-1])
        x = np.stack(x_list, axis=0)
        y = windowed_df.iloc[i + input_num : i + input_num + num_step, -1].values

        xs.append(x)
        ys.append(y)

    return np.stack(xs, axis=0), np.stack(ys, axis=0)

