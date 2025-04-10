# %%
from datetime import timedelta

import pandas as pd

# %%
df = pd.read_csv('data/m0/m0_10s.csv')

df = df.rename(columns={'trace_id': 'station',
                        'onset_time': 'time'})
df['timestamp'] = pd.to_datetime(
    df['time'], unit='ns').dt.tz_localize(None)


df['station'] = df['station'].apply(lambda x: x[3:8])

# adjust start and endtime of samples *****************************************
start = df['timestamp'].min() + timedelta(seconds=0.0)  # 0.4
end = df['timestamp'].max() - timedelta(seconds=0.6)  # 0.6

df = df[(df['timestamp'] >= start) & (
    df['timestamp'] <= end)]

# Filter out "overly active" stations *****************************************
# group by station and count number of picks for each station
n_filter = 4
filter = False
groups = df.groupby('station')
counts = []
for name, group in groups:
    counts.append([name, len(group)])
counts.sort(key=lambda x: x[1], reverse=True)
if filter:
    df = df[~df['station'].isin([x[0] for x in counts[:n_filter]])]

# Write to csv ***************************************************************
df[['station', 'time', 'phase']].to_csv(
    'data/m0/m0_10s_formatted_low.csv', index=False)


# %%

cat = pd.read_csv('data/m0/seiscomp_catalog.csv')
start = df['timestamp'].min()
end = df['timestamp'].max()

cat['time'] = pd.to_datetime(
    cat['isotime'], unit='ns').dt.tz_localize(None)

cat = cat[(cat['time'] >= start) & (
    cat['time'] <= end)]

cat = cat.rename(columns={'x': 'e',
                          'y': 'n',
                          'z': 'u'})
cat['u'] = cat['u'] / 1000 - 1485.379791
cat[['e', 'n', 'u', 'time', 'magnitude']
    ].to_csv('data/m0/catalog.csv')
