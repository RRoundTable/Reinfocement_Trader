import pandas as pd


def load_chart_data(fpath):
    df = pd.read_csv(fpath)
    df['time'] = df['time'].apply(str)
    df = df.set_index("time")
    return df


def preprocess(chart_data):
    df = chart_data
    windows = [5, 20, 60, 120]
    for window in windows:
        df[f'end_{window}'] = df['end'].rolling(window).mean()
        df[f'amount_{window}'] = df['amount'].rolling(window).mean()
    df = df.dropna(axis=0)
    return df


chart_data = load_chart_data("chart.csv")
prep_data = preprocess(chart_data)
prep_data.to_csv("chart_ma.csv")
