import numpy as np
import pandas as pd

def get_window_edges(df, window=0.1, sampling_rate=20000):
    df.spike_time = (df.spike_time - df.flash_onset).round(5)

    arr = np.arange(-0.5, .5, 1/sampling_rate).round(5)
    arr = np.column_stack([arr, np.where(np.isin(arr, df['spike_time']), 1, 0)])
    arr[:,1] = arr[:,1].cumsum()
    
    arr = arr[np.where(arr[:,0] >= -0.5+window)] - arr[np.where(arr[:,0] < 0.5-window)]*[0,1]
    pre = arr[np.where(arr[:,0] < 0)]
    post = arr[np.where(arr[:,0] >= window)]

    df['pre_min_edge'] = pre[np.argmin(pre[:,1]),0]
    df['pre_max_edge'] = pre[np.argmax(pre[:,1]),0]
    df['post_min_edge'] = post[np.argmin(post[:,1]),0]
    df['post_max_edge'] = post[np.argmax(post[:,1]),0]
    return df

def get_window_counts(df, window=0.1):
    pre_min, pre_max, post_min, post_max = df.iloc[0,-4:]
    result = pd.Series({
        'baseline': (df.spike_time < 0).sum()/df.flash_onset.iloc[0],
        'pre': df.spike_time.between(pre_max - window, pre_max).sum(),
        'post': df.spike_time.between(post_max - window, post_max).sum(),
    })
    if df.response_type.iloc[0][:3] == 'OFF':
        result.pre -= df.spike_time.between(pre_min - window, pre_min).sum()
        result.post -= df.spike_time.between(post_min - window, post_min).sum()
    return result


spikes = pd.read_csv("dfs/spike_times.csv")
files = pd.read_csv("dfs/files_df.csv")
cells = pd.read_csv("dfs/initial/initial_cells_df.csv")
files = files.merge(cells[['cell', 'response_type']], on='cell')

sweeps = (spikes
    .merge(files[['file_name', 'flash_onset', 'response_type']], on='file_name')
    .groupby('file_name')
    .apply(get_window_edges)
    .groupby(['file_name', 'sweep'])
    .apply(get_window_counts)
    .reset_index()
)