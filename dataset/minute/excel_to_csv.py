import pandas as pd

df = pd.read_excel('samsung.xlsx', 'Sheet1')
df['인덱스'] = df['일자'].apply(str) + df['시간'].apply(str)
df = df.drop(columns=['일자', '시간'])
df.columns = ['start', 'high', 'low', 'end', 'amount', 'time']
df = df.set_index('time')
df.to_csv("chart.csv")
