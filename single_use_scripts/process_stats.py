import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import metrics

def single_wilcoxon(group):
    differences = group.post - group.pre
    try:
        _, wilcoxon_p = wilcoxon(x=differences, zero_method='pratt', alternative='greater')
    except ValueError:
        wilcoxon_p = 0
    return wilcoxon_p

def single_roc(group):
        x = list(group.pre) + list(group.post)
        y = [0]*(len(x)//2) + [1]*(len(x)//2)
        fpr, tpr, _ = metrics.roc_curve(y, x)
        auc = metrics.auc(fpr, tpr)
        return auc
        return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'auc': [auc]*len(fpr)})

def wilcoxon_breakpoints(df, col='wilcoxon_p'):
    df['sig'] = (df[col] < 0.05).cumsum()
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
    return pd.Series({'wilcoxon_high':high, 'wilcoxon_mid':mid, 'wilcoxon_low':low})

def AUC_breakpoints(df, col='AUC'):
    df['sig'] = (df[col] >= 0.60).cumsum()
    if df.sig.all():
        low = crossing = None
        high = df.intensity.iloc[0]
    elif df.sig.any():
        idx = np.argmax(df.sig>0)
        low = df.intensity.iloc[idx-1]
        high = df.intensity.iloc[idx]

        m = (df.AUC.iloc[idx] - df.AUC.iloc[idx-1]) / (high - low)
        b = df.AUC.iloc[idx] - m * high
        crossing = (0.60 - b)/m

    else:
        low = crossing = high = None
    return pd.Series({'AUC_high':high, 'AUC_crossing':crossing, 'AUC_low':low})

folder = "dfs_ESM"
cells = pd.read_csv(folder + '/initial_cells_df.csv')
files = pd.read_csv(folder + '/files_df.csv')
sweeps = pd.read_csv(folder + '/sweeps_df.csv')
sweeps = sweeps.merge(files[['file_name','cell','intensity']], on='file_name')


intensities = (files
    .groupby(['cell', 'intensity'])['sweep_count'].sum()
    .to_frame()
    .assign(baseline = sweeps.groupby(['cell', 'intensity'])['baseline'].mean())
    .assign(wilcoxon_p = sweeps.groupby(['cell', 'intensity']).apply(single_wilcoxon))
    .assign(AUC = sweeps.groupby(['cell', 'intensity']).apply(single_roc))
    .reset_index()
    #.pipe(lambda df: df[df.sweep_count>=5])
    .sort_values(by=['cell', 'intensity'])
)

cells = (cells
    .merge(intensities.groupby('cell').apply(wilcoxon_breakpoints), on='cell')
    .merge(intensities.groupby('cell').apply(AUC_breakpoints), on='cell')
)
