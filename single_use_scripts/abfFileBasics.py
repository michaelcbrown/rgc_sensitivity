import numpy as np
import pandas as pd
from datetime import datetime
import pyabf
import os


def abfFileBasics(row, path, flash_channel=3):
    """
    This takes the "initial" files_df, where it's just info that I entered by hand at some point,
    And adds in information from the abfs.
    """
    abf = pyabf.ABF(os.path.join(path, row.file_name))
    abf.setSweep(sweepNumber=0, channel=flash_channel)
    idx = np.argmax(abf.sweepY > abf.sweepY[0]+1)
    row['flash_onset'] = abf.sweepX[idx]
    row['start_time'] = abf.abfDateTimeString.split('T')[1]
    row['sweep_length'] = abf.sweepLengthSec
    row['sweep_count'] = abf.sweepCount
    return row

def make_time_relative(df):
    """
    .start_time is a string from the .abf file
    This turns them into datetime objects so they can be subtracted and converted to seconds
    The 1st sweep of the 1st recording for each cell gets set here as t=0
    """
    df.start_time = [datetime.strptime(t, '%H:%M:%S.%f') for t in df.start_time]
    df.start_time = df.groupby('cell')['start_time'].apply(lambda group: group-group.min())
    return [t.total_seconds() for t in df.start_time]


files_df = (
    pd.read_csv('dfs/initial/initial_files_df.csv')
    .apply(abfFileBasics, path='abfs/HB', axis=1)
    .assign(start_time = lambda df: make_time_relative(df))
)