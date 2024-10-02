'''
Bunch of helper functions to make my life easier. It's always 
nice to say please when asking for help.
'''

import os
import numpy as np
from collections import defaultdict

def list_to_int(num_list: list[int]) -> int:
    """
    Convert a list of numbers into a single integer.
    
    Args:
    List of numbers to be concatenated.
    
    Returns:
    The concatenated integer.
    """
    # Convert each number to a string and concatenate them
    concatenated_str = ''.join(map(str, num_list))
    
    # Convert the concatenated string back to an integer
    return int(concatenated_str)

def int_to_list(num: int) -> list[int]:
    """
    Convert an integer into a list of numbers.
    
    Args:
    The integer to be converted.
    
    Returns:
    The list of numbers.
    """
    # Convert the integer to a string
    num_str = str(num)
    
    # Convert each character to an integer and store in a list
    return [int(char) for char in num_str]

def get_active_fingers(chord: int) -> list[int]:
    """
    Get the active fingers in a chord.
    
    Args:
    The chord number.
    
    Returns:
    The list of active fingers.
    """
    # Convert the chord number to a list of integers
    chord_list = int_to_list(chord)
    
    # Get the indices of the active fingers
    return [i + 1 for i, val in enumerate(chord_list) if val != 9]

def estimate_transition_matrix(press_matrix, states: list[int]=[1,2,3,4,5]):
    '''
    Estimate the transition proabilities between states. In the case of EFC2, it's the 
    transition probabilities between fingers pressed.

    Args:
        press_matrix: Each row is a trial. Each column is the press index. 
    '''

    num_states = len(states)
    transition_matrix = np.zeros((num_states, num_states))
    transition_counts = defaultdict(lambda: np.zeros(num_states))

    num_trials = press_matrix.shape[0]

    # Create a mapping for the states to ensure we're only working with the specified ones
    state_to_index = {state: i for i, state in enumerate(states)}

    # Loop through trials:
    for trial in range(num_trials):
        sequence = press_matrix[trial]
        # if sequence was nan, skip:
        if np.isnan(sequence).any():
            continue
        # Remove padded zeros:
        sequence = [int(x) for x in sequence if x != 0]
        # Loop through presses:
        for i in range(len(sequence) - 1):
            # Get the current and next press:
            current_finger = sequence[i]
            next_finger = sequence[i + 1]

            # Only count transitions between the specified states
            if current_finger in states and next_finger in states:
                current_idx = state_to_index[current_finger]
                next_idx = state_to_index[next_finger]
                transition_counts[current_idx][next_idx] += 1

    # Normalize the counts to get probabilities
    for finger in range(num_states):
        total_transitions = np.sum(transition_counts[finger])
        if total_transitions > 0:
            transition_matrix[finger] = transition_counts[finger] / total_transitions
    
    return transition_matrix
            
def get_trial_force(mov, fGain, global_gain, baseline_threshold, fs, t_minus=None, t_max=None):
    
    WAIT_EXEC = 3

    # find the beginning of the execution period:
    start_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][0]
    end_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][-1]

    # get the differential forces - five columns:
    force = mov[:, 13:18]
    
    # apply the gains:
    force = force * fGain * global_gain

    # find the first time any finger exits the baseline zone:
    for i in range(start_idx, end_idx+1):
        if np.any(np.abs(force[i, :]) > baseline_threshold):
            RT_idx = i
            break
    
    if t_minus is not None:
        start_idx = RT_idx - int(t_minus/1000*fs)
    if t_max is not None:
        end_idx = start_idx + int(t_max/1000*fs)
    
    # get the differential forces - five columns:
    force = force[start_idx:end_idx, :]
    t = (mov[start_idx:end_idx, 3] - mov[start_idx, 3])/1000 # time in seconds
    if t_minus is not None:
        t = t - t_minus/1000
    
    return t, force