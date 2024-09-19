import os
import re
import fnmatch
import numpy as np
import pandas as pd
import efc2_routine
import utils
from utils.config import *

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
    name_pattern = re.compile(r'efc2_\d+\.csv')
    files = sorted([f for f in os.listdir(ANALYSIS_PATH) if name_pattern.match(f)])
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

        # empty lists to store values:
        RT = []
        ET = []
        MD = []
        trained_flag = []

        # load participant info to add trained or untrained flag to chords:
        participants_info = pd.read_csv(os.path.join(DATA_PATH, 'participants.tsv'), sep='\t', usecols=['Subject number','trained','untrained'])
        participants_info = participants_info.rename(columns={'Subject number':'subNum'})
        trained_chords = participants_info['trained'][participants_info['subNum']==data['subNum'].iloc[0]].values[0].split('.')
        trained_chords = [int(i) for i in trained_chords]
        untrained_chords = participants_info['untrained'][participants_info['subNum']==data['subNum'].iloc[0]].values[0].split('.')
        untrained_chords = [int(i) for i in untrained_chords]
        
        # Loop through the trials:
        for j in range(data.shape[0]):
            # add trained or untrained chord flag:
            if data['chordID'].iloc[j] in trained_chords:
                trained_flag.append(1)
            elif data['chordID'].iloc[j] in untrained_chords:
                trained_flag.append(0)

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
        data['trained'] = trained_flag

        # Concatenate the dataframe to df:
        df = pd.concat([df, data], ignore_index=True)


    # add participant groups:
    participants_info = pd.read_csv(os.path.join(DATA_PATH, 'participants.tsv'), sep='\t', usecols=['Subject number','group'])
    participants_info = participants_info.rename(columns={'Subject number':'subNum'})
    df = pd.merge(df, participants_info, on='subNum', how='left')

    df.rename(columns={'subNum':'sn'},inplace=True)
    df = reorder_dataframe(df)

    # Save the dataframe:
    df.to_csv(os.path.join(ANALYSIS_PATH, 'efc2_all.csv'), index=False)

def reorder_dataframe(df):
    '''
    reorders the dataframe columns to make it more readable
    '''
    
    order = ['day', 'is_test', 'group', 'sn', 'BN', 'TN', 'trial_correct', 'chordID', 'trained', 'RT', 'ET', 'MD', 
             'planTime', 'execMaxTime', 'feedbackTime', 'iti', 
             'fGain1', 'fGain2', 'fGain3', 'fGain4', 'fGain5', 'forceGain', 
             'baselineTopThresh', 'extTopThresh', 'extBotThresh', 'flexTopThresh', 'flexBotThresh']
    
    df = df[order]
    df.reset_index(drop=True, inplace=True)
    
    return df

def should_I_trust_this_data(ana):
    '''
    Well, should you? Run this function, look carefully, and find out!
    '''

    subjects = ana['sn'].unique()
    days = ana['day'].unique()
    BN = ana['BN'].unique()

    red_flags = 0

    for sn in subjects:
        print(f'=========== subject {sn} ===========')
        for day in days:
            print(f'----- {day} -----')
            for bn in BN:
                tn = ana['TN'][(ana['sn'] == sn) & (ana['day'] == day) & (ana['BN'] == bn)]
                if len(tn) != 50:
                    red_flags += 1
                    print(f'*********************************************** ERROR IN NEXT LINE:')
                print(f'sn: {sn}, day: {day}, BN: {bn}, TN: {len(tn)}')
    
    print(f'==== number of red flags: {red_flags} ====')

def collapse_chords(df, repetitions=5):
    '''
    Creates a dataframe for the improvement of measures across days.
    The performance is averaged within trained and untrained chords separately for each participant.
    '''

    ANA = pd.DataFrame()

    # add a repetition column the size of the dataframe:
    df['repetition'] = np.tile(np.arange(1, repetitions+1), (1,int(df.shape[0]/repetitions))).flatten()
    df.replace(-1, np.nan, inplace=True)
    
    # make summary dataframe:
    ANA = df.groupby(['day','sn','trained','repetition'])[['is_test','group','RT','ET','MD']].mean().reset_index()

    ANA['adjusted_repetition'] = (ANA['day']-1)*repetitions + ANA['repetition']

    # save the ANA dataframe:
    ANA.to_csv(os.path.join(ANALYSIS_PATH, 'efc2_collapse_chords.csv'), index=False)

    return ANA


        


    


