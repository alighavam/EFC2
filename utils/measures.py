import numpy as np

def get_RT(mov: np.ndarray, baseline_threshold: float, fGain: list[float], global_gain: float):
    '''
    RT is defined as the time from the beginning of the execution period (when the targets signal the go-cue)
    to the time that any finger exits the baseline zone (baseline_threshold) for the first time.

    params:
        mov: the mov data for a single trial
        baseline_threshold: the threshold for the baseline zone
        fGain: the gain for force of each finger
        global_gain: the global gain for all finger forces
    '''

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
    
    return mov[RT_idx, 2] - mov[start_idx, 2]

def get_ET(mov: np.ndarray):
    '''
    ET is defined as the time from the the beginning of the execution period (when the targets signal the go-cue)
    to the beginning of the holding period (The first time when the fingers are in the correct position).

    params:
        mov: 2D the mov data for a single trial
    '''

    WAIT_EXEC = 3

    # find the beginning of the execution period:
    start_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][0]
    end_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][-1]

    return mov[end_idx, 2] - mov[start_idx, 2] - 600

def get_MD(mov: np.ndarray, baseline_threshold: float, fGain: list[float], global_gain: float, fs: int, hold_time: float):
    '''
    Mean deviation captures the simultaneity of the forces of the fingers in trial. It is defined as the average of 
    the norm of the deviation of the forces from the ideal trajectory.

    sum from t=1 to T of: norm(F_t - (C' * F_t)/norm(C)^2 . C) / T
    where F_t is the force of the 5 fingers at each time point t and C 
    is the ideal trajectory that you can take to reach the target
    position (Here it is assumed that the ideal trajetory is a straight 
    line from the starting position to the ending position).

    Look Waters-Metenier et al. 2014 for more details.

    params:
        mov: the mov data for a single trial
        baseline_threshold: the threshold for the baseline zone
        fGain: the gain for force of each finger
        global_gain: the global gain for all finger forces
        fs: the sampling frequency of the force data
        hold_time: the time to hold the target in ms
    '''

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

    # select the portion of force from RT to end_idx-hold_time:
    force = force[RT_idx:end_idx-int(hold_time/1000*fs), :]
    
    # calculate the ideal trajectory:
    c = force[-1, :] - force[0, :]

    # calculate mean deviation:
    deviation = []
    for i in range(1,force.shape[0]):
        # force vector:
        tmp_force = force[i, :] - force[0, :]
        # projection:
        projection = np.dot(tmp_force, c) / np.dot(c, c) * c
        # deviation:
        deviation.append(np.linalg.norm(tmp_force - projection))

    return np.mean(deviation)

def get_press_sequence(mov: np.ndarray, baseline_threshold: float, fGain: list[float], global_gain: float, fs: int, hold_time: float):
    """
    Detects the sequence of finger presses based on a threshold.

    params:
        

    returns:
        
    """

    WAIT_EXEC = 3

    # find the beginning of the execution period:
    start_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][0]
    end_idx = np.where(mov[:, 0] == WAIT_EXEC)[0][-1]

    # get the differential forces - five columns:
    force = mov[:, 13:18]
    
    # apply the gains:
    force = force * fGain * global_gain
    # select the portion of force from start of executin phase to end_idx-hold_time:
    force = force[start_idx:end_idx-int(hold_time/1000*fs), :]
    
    # seuqneces of fingers pressed, e.g., 121345:
    seq = []
    # direction of each press, e.g., feffee
    press_dir = []
    # time of each press:
    press_time = []
    
    # Get the number of time points (t) and number of fingers (5)
    t, num_fingers = force.shape
    
    # Initialize a boolean array to track whether each finger was below the threshold
    below_threshold = np.ones(num_fingers, dtype=bool)
    
    # Loop through the time points
    for i in range(t):
        for finger in range(num_fingers):
            # Check if the force exceeds the threshold and if the finger was previously below the threshold
            tmp_force = np.abs(force[i, finger])
            if tmp_force > baseline_threshold and below_threshold[finger]:
                # Add the finger index (1-based) to the press sequence
                seq.append(finger + 1)
                # Add the direction of the press to the press direction sequence:
                press_dir.append('f' if force[i, finger] < 0 else 'e')
                # Add the time of the press to the press time sequence:
                press_time.append(mov[start_idx+i, 2])

                # Mark this finger as having crossed the threshold
                below_threshold[finger] = False
            # Reset the below_threshold state if the force goes below the threshold again
            elif tmp_force <= baseline_threshold:
                below_threshold[finger] = True
    
    return seq, press_dir, press_time
    