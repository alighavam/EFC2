import os
import fnmatch
import numpy as np
import pandas as pd
import efc2_routine
import utils
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


def make_all_dataframe(fs: int = 500, hold_time: float = 600):
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
        # Drop unnamed column:
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        # Load the .mov file:
        mov = pd.read_pickle(os.path.join(ANALYSIS_PATH, movFiles[i]))

        # empty lists to store the RT, ET, MD values:
        RT = []
        ET = []
        MD = []

        # Loop through the trials:
        for j in range(data.shape[0]):
            # if trial was correct
            if data['trial_correct'][j]:
                fGain = [data['fGain1'].iloc[j], data['fGain2'].iloc[j], data['fGain3'].iloc[j], data['fGain4'].iloc[j], data['fGain5'].iloc[j]]
                
                # calculate RT, ET, MD:
                RT.append(utils.measures.get_RT(mov['mov'].iloc[j], data['baselineTopThresh'].iloc[j], fGain, data['forceGain'].iloc[j]))
                ET.append(utils.measures.get_ET(mov['mov'].iloc[j]))
                MD.append(utils.measures.get_MD(mov['mov'].iloc[j], data['baselineTopThresh'].iloc[j], fGain, data['forceGain'].iloc[j], 
                                                fs, hold_time))

            else:
                RT.append(-1)
                ET.append(-1)
                MD.append(-1)

        data.drop(columns=['RT','trialPoint'], inplace=True)

        # Add RT, ET, MD columns:
        data['RT'] = RT
        data['ET'] = ET
        data['MD'] = MD

        # Concatenate the dataframe to df:
        df = pd.concat([df, data], ignore_index=True)
    
    # Save the dataframe:
    df.to_csv(os.path.join(ANALYSIS_PATH, 'efc2_all.csv'), index=False)

    




    

