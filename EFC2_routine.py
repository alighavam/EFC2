import os
import numpy as np
import pandas as pd
from config import *

def subject_routine(subject, smoothing_window=30, fs=500):
    """
    This function is used to preprocess the data of a subject
    
    params:
        subject: int, the subject number
        smoothing_window: int, the window size for the moving average filter for forces
        fs: int, the sampling frequency of the force data
    """
    # empty dataframe to store the data:
    df = pd.DataFrame()

    # Load each day's data:
    dirs = [entry.name for entry in os.scandir(DATA_PATH) if entry.is_dir()]
    for dir in dirs:
        # testing or training path:
        sub_dir = os.path.join(DATA_PATH, dir)

        # Loop through the days:
        days_dir = [entry.name for entry in os.scandir(sub_dir) if entry.is_dir()]
        for d in days_dir:

            # Load the .dat file:
            dat_file_name = os.path.join(sub_dir,d, f'efc2_{subject}.dat')
            dat = pd.read_csv(dat_file_name, sep='\t')

            
