import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def gaussian_convolution(spikes, sweep_length, sigma=500):
    indices = spikes * 20000
    indices = indices.astype(int)
    
    time = np.arange(0, int(sweep_length), 1/20000).round(5)

    arr = np.zeros(int(sweep_length) * 20000)
    for idx in indices:
        arr[idx] += 1
    gauss = gaussian_filter1d(arr, sigma=sigma, mode='constant', cval=arr.mean())
    return time, gauss

def gauss_test(group):
    sweep_length = int(group.sweep_length.max())
    flash_onset = group.flash_onset.max().round(4)
    group.spike_time += (flash_onset - group.flash_onset)

    time, gauss = gaussian_convolution(group.spike_time, sweep_length)

    onset_idx = np.where(time==flash_onset)[0][0]
    window_idx = np.where(time==flash_onset+0.5)[0][0]

    if group.response_type.iloc[0] in ['OFF sustained', 'OFF transient']:
        gauss *= -1

    score = 0
    if gauss[onset_idx:window_idx].max() == gauss.max():
        score += 2
    if gauss[onset_idx:window_idx].max() > gauss.mean() + 2*gauss.std():
        score += 1
    return score

def gauss_test_breakpoints(df, col='gauss_score'):
    df['sig'] = (df[col] == 3).cumsum()
    if df.sig.all():
        low = mid = None
        high = df.intensity.iloc[0]
    elif df.sig.any():
        idx = np.argmax(df.sig>0)
        low = df.intensity.iloc[idx-1]
        high = df.intensity.iloc[idx]
        mid = (high + low)/2
    else:
        low = mid = high = None
    return pd.Series({'gauss_high':high, 'gauss_mid':mid, 'gauss_low':low})

def run_test():
    files_df = pd.read_csv(folder + '/files_df.csv')
    cells_df = pd.read_csv(folder + '/cells_df.csv')
    spike_times = pd.read_csv(folder + '/spike_times.csv')
    intensities_df = pd.read_csv(folder + '/intensities_df.csv')

    return (files_df
        .merge(cells_df[['cell', 'response_type']], on='cell')
        .merge(spike_times, on='file_name')
        .groupby(['cell', 'intensity']).apply(gauss_test)
        .rename('gauss_score')
        .reset_index()
        .merge(intensities_df, on=['cell','intensity'])
    )

def get_key_intensities(intensity_df):
    cells_df = pd.read_csv(folder + '/cells_df.csv')

    return (cells_df
        .merge(intensity_df.groupby('cell').apply(gauss_test_breakpoints), on='cell')
    )

folder = 'dfs_ESM'

"""
Use run_test() and save as a new intensities_df.csv
Then feed that output into get_key_intensities, and get a new cells_df.csv
"""