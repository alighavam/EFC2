import os
import numpy as np
import pandas as pd
from utils.movload import movload
from config import *

def trial_routine(row):
    '''
    This function performs all the necessary preprocessing for a single trial
    '''
    C = pd.DataFrame()

    # add row to the dataframe:
    C = pd.concat([C, row], ignore_index=True)
    C = C.rename(columns={'trialCorr': 'trial_correct'})

    return C
    
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
    df_mov = pd.DataFrame(columns=['day', 'sn', 'BN', 'TN', 'trial_correct', 'mov'])

    # I will add these columns to the dataframe:
    day = []
    is_test = []

    # Load each day's data:
    dirs = [entry.name for entry in os.scandir(DATA_PATH) if entry.is_dir()]
    for dir in dirs:
        # testing or training path:
        sub_dir = os.path.join(DATA_PATH, dir)

        # Loop through the days:
        days_dir = [entry.name for entry in os.scandir(sub_dir) if entry.is_dir()]
        for d in days_dir:
            print(f'=============== {d} ===============')
            # Load the .dat file:
            dat_file_name = os.path.join(sub_dir,d, f'efc2_{subject}.dat')
            dat = pd.read_csv(dat_file_name, sep='\t')

            oldblock = -1
            # loop on trials:
            for i in range(dat.shape[0]):
                if dat['BN'][i] != oldblock:
                    print(f'Processing block {dat["BN"][i]}')
                    # load the .mov file:
                    mov = movload(os.path.join(sub_dir,d,f'efc2_{subject}_{dat["BN"][i]:02d}.mov'))
                    oldblock = dat['BN'][i]
                print(f'Processing trial {dat["TN"][i]}')
                # trial routine:
                C = trial_routine(dat.iloc[[i]])

                # append the trial to the dataframe:
                df = pd.concat([df, C], ignore_index=True)
                
                # building the day and is_test columns for df:
                day.append(int(d[-1]))
                is_test.append(dir == 'testing')

                # add the mov trial in the move dataframe:
                tmp = pd.DataFrame({'day': int(d[-1]), 'sn': subject, 'BN': dat['BN'][i], 'TN': dat['TN'][i], 'trial_correct':dat['trialCorr'][i], 
                                    'mov': [mov[dat['TN'][i]-1]]})
                df_mov = pd.concat([df_mov, tmp],ignore_index=True)

    # add the day and is_test columns to the dataframe:
    df['day'] = day
    df['is_test'] = is_test

    # sort the dataframes by day:
    df = df.sort_values(by='day', kind='mergesort')
    df_mov = df_mov.sort_values(by='day', kind='mergesort')

    # save the data frames:
    df.to_csv(os.path.join(ANALYSIS_PATH, f'efc2_{subject}.csv'), index=False)
    df_mov.to_pickle(os.path.join(ANALYSIS_PATH, f'efc2_{subject}_mov.pkl'))
    
    return df, df_mov
                


