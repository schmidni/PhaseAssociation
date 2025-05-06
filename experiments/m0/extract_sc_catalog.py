# %%
import pandas as pd

# df = pd.read_csv(
#     '../../data/m0/2024-04-30T040800-T041800_V__Sp0.15_Ss0.30_processed.csv')
df = pd.read_csv(
    '../../data/m0/2024-04-30T040909-T040910_V__Sp0.15_Ss0.30_processed.csv'
)

df['timestamp'] = pd.to_datetime(
    df['time'], unit='ns').dt.tz_localize(None)

cat = pd.read_csv('../../data/m0/sc_fdsn_cat_M0a.csv')
start = df['timestamp'].min()
end = df['timestamp'].max()

cat['time'] = pd.to_datetime(
    cat['time'], unit='ns').dt.tz_localize(None)

cat = cat[(cat['time'] >= start) & (
    cat['time'] <= end)]

# cat = cat.rename(columns={'x': 'e',
#                           'y': 'n',
#                           'z': 'u'})
# cat['u'] = cat['u'] / 1000 - 1485.379791

cat[['longitude', 'latitude', 'depth', 'time', 'mag']
    ].to_csv('../../data/m0/catalog.csv')

# %%
