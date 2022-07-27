from inspect import stack
from process_gauss import gaussian_convolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    files_df = pd.read_csv('dfs/files_df.csv')
    cells_df = pd.read_csv('dfs/cells_df.csv')
    intensities_df = pd.read_csv('dfs/intensities_df.csv')
    spike_times = pd.read_csv('dfs/spike_times.csv')

    return (files_df
        .merge(cells_df[['cell', 'response_type', 'condition']], on='cell')
        .merge(spike_times, on='file_name')
        .merge(intensities_df[['cell','intensity','gauss_score']], on=['cell','intensity'])
    )

def stacked_plots(df, show=True):
    groups = (df.groupby('intensity', as_index=False))
    fig, axes = plt.subplots(groups.ngroups, 1, figsize=[10,8])

    for ax, (intensity, group) in zip(fig.get_axes(), groups):
        sweep_length = int(group.sweep_length.max())
        flash_onset = group.flash_onset.max().round(5)
        group.spike_time += (flash_onset - group.flash_onset)
        time, gauss = gaussian_convolution(group.spike_time, sweep_length)

        score = group.gauss_score.iloc[0] == 3
        colors = {0: 'orange', 1: 'blue'}

        ax.plot(time, gauss, color=colors[score])
        ax.plot([flash_onset, flash_onset], [gauss.min(), gauss.max()], linestyle=':', color=colors[score])
        ax.plot([flash_onset+0.5, flash_onset+0.5], [gauss.min(), gauss.max()], linestyle=':', color=colors[score])
        ax.plot(time[np.argmax(gauss==gauss.max())], gauss.max(), 'bo')

        ax.set_xlim(0,4)
        ax.set_yticks([])
        ax.set_ylabel(f"{intensity}\n{group.groupby(['file_num', 'sweep']).ngroups} sweeps", rotation=0, labelpad=40)

    sns.despine(ax=ax, left=True)
    ax.set_xlabel('s')
    for ax in fig.get_axes()[:-1]:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])

    fig.suptitle(f"{df.cell.iloc[0]}, {df.response_type.iloc[0]}")
    plt.tight_layout()
    if show:
        plt.show()

df = get_data()

if 0:
    for cell in df.cell.unique():
        stacked_plots(df[df.cell==cell], show=False)
        plt.savefig(f'figures/stacked_convolutions/{cell}')
        plt.clf()

if 1:
    stacked_plots(df[df.cell=='H21021901'])