import os
import fnmatch
import numpy as np
import pandas as pd
import efc2_routine
from utils.movload import movload
from config import *

def subject_routine(subject: list[int], smoothing_window: int = 30, fs: int = 500):
    """
    This function is used to preprocess the raw of a subject coming from the experimental computers: .dat and .mov

    params:
        subject: list, a list of subject numbers
        smoothing_window: int, the window size in milliseconds for the moving average filter for forces
        fs: int, the sampling frequency of the force data in Hz
    """
    
    for i in subject:
        efc2_routine.subject_routine(i, smoothing_window, fs)


def make_all_dataframe():
    """
    Goes through all the preprocessed data and creates a flat .csv dataframe where each row is one trial. 
    All subject data are concatenated into one dataframe.

    RT, ET, MD, for each trial are calculated and saved in the dataframe.

    This dataframe also includes the incorrect trials. RT, ET, MD values are set to -1 for incorrect trials.
    """

    # Get file names:
    files = sorted([f for f in os.listdir(ANALYSIS_PATH) if fnmatch.fnmatch(f, 'efc2_*.csv')])
    movFiles = sorted([f for f in os.listdir(ANALYSIS_PATH) if fnmatch.fnmatch(f, 'efc2_*_mov.pkl')])

    # Create an empty dataframe:
    df = pd.DataFrame()

    # Loop through the files:
    for i in range(len(files)):
        print(f'Processing {files[i]}')
        # Load the data:
        data = pd.read_csv(os.path.join(ANALYSIS_PATH, files[i]))
        mov = pd.read_pickle(os.path.join(ANALYSIS_PATH, movFiles[i]))

        

        # Add the data to the dataframe:
        df = pd.concat([df, data], ignore_index=True)


    return files, movFiles




    


