import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()

def single_roc(group):
        x = list(group.pre) + list(group.post)
        y = [0]*(len(x)//2) + [1]*(len(x)//2)
        fpr, tpr, _ = metrics.roc_curve(y, x)
        auc = metrics.auc(fpr, tpr)
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        df.loc[:,'auc'] = auc
        return df

def plot_ROC(cell_df, show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[9,5])

    df = cell_df.groupby('intensity').apply(single_roc).reset_index()
    sns.lineplot(data=df, x='fpr', y='tpr', hue='intensity', estimator=None, ax=ax1)
    sns.scatterplot(data=df, x='intensity', y='auc', ax=ax2)

    ax1.plot([0,1], [0,1], 'k-.')
    ax2.plot([0,df.intensity.max()], [0.6,0.6], 'k-.')
    fig.suptitle(f"{cell_df.cell.iloc[0]}, {cell_df.response_type.iloc[0]}")

    plt.tight_layout()
    if show:
        plt.show()

def get_data():
    files = pd.read_csv('dfs/files_df.csv')
    sweeps = pd.read_csv('dfs/sweeps_df.csv')
    cells = pd.read_csv('dfs/cells_df.csv')

    df = pd.merge(files, sweeps, on=['file_name'])
    df = pd.merge(df, cells[['cell', 'response_type']], on=['cell'])
    return df

if 0:
    df = get_data()
    for cell in df.cell.unique():
        plot_ROC(df[df.cell==cell], show=False)
        plt.savefig(f'figures/ROC/{cell}')
        plt.clf()

if 1:
    df = get_data()
    plot_ROC(df[df.cell=='H21033101'])