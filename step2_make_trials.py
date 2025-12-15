import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


from typing import List
from pathlib import Path
from scipy.io import savemat

from neuroseries.time_series import Tsd
from neuroseries.interval_set import IntervalSet, Range

plt.rcParams["figure.figsize"] = (13, 4)
pix_to_cm = 10000  # Need to update the converting constant here
speed_limit = 15  # Same, needs to be adapted with real pixel 2 cms conversion
trial_times = {}   
    
def plot(df: pd.DataFrame, dst: str = None, filename: str = None):
    colors = np.array(df['colors'])
    df = df.reset_index(drop=False)
    df.dropna().plot.scatter(x='Time (us)', y='position', 
                             figsize=(16,7), alpha=0.5,
                             c=colors, s=1)
    if dst:
        dst = str(dst/filename)
        plt.savefig(dst, format='png', dpi=600)
    plt.show()


def load_data(dir_data):
    # Load BehaveEpochs
    analysis_dir = dir_data/'Analysis'
    session_epochs = pd.read_csv(dir_data/'Epoch_TS.csv', names=[0, 1])
    # Load Positions
    try:
        linear_positions_file = list(analysis_dir.glob('LinearPos_*'))[0]
        print('Using ', linear_positions_file)
    except IndexError:
        print("LinearPos_*.csv not found")
        sys.exit()
    lin_pos = pd.read_csv(linear_positions_file, index_col=['Time (s)'])
    # lin_pos.reset_index().plot.scatter(x='Time (s)', y='position')
    # TODO: Tsd round the times so if t[0] = 4191.1081 it will round it to 4191
    lin_pos = Tsd(t=lin_pos.index.values, d=lin_pos['position'].values, time_units='s')
    position = pd.read_hdf(analysis_dir/'Position.h5', mode='r')
    return session_epochs, lin_pos, position, 


def set_speed(position, pos_time):
    # Defining speed.
    x, y = position['x'].values, position['z'].values
    x, y = pix_to_cm * x, pix_to_cm * y
    x, y = Tsd(t=pos_time.values, d=x), Tsd(t=pos_time.values, d=y)
    dx, dy = np.insert(np.diff(x), 0, 0), np.insert(np.diff(y), 0, 0)
    speed = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    
    # Using matlab script
    eng = matlab.engine.start_matlab()
    speed = np.array(eng.gaussFilt(matlab.double(list(speed)), 25, 0, 0)).flatten()
    eng.quit()

    speed = Tsd(t=pos_time.values, d=speed)
    return speed


def segment_departure(idx_dep_off, idx_dep_on, speed_points, trials, min_position):    
    # When did he start running before or reach the lowest point since last entrance?
    idx_start = idx_dep_off
    departure = trials.loc[min_position:idx_dep_off]
    departure_speed = speed_points[min_position:idx_dep_off]
    
    # Check if the speed is greater than 15
    if any(departure_speed < speed_limit):
        idx_start = np.where(departure_speed < speed_limit)[0][-1]
    else:
        idx_start = departure.index[0]
    ## ########## Comment ########## 
    try:
        min_position = np.argmin(departure['position'])
    except ValueError:
        import pdb; pdb.set_trace()

    min_position = min_position + idx_dep_on
    idx_start = max(idx_start, min_position)
    return idx_start


def make_trials(dir_data: str, events: List, 
                right_threshold: float = 2.95, 
                left_threshold: float = 1.95):
    """ Create trials of the given events using the Position.h5 and Epoch_TS.csv
        files.

    Args:
        dir_data (str): the direction of the folder that contains Analysis 
        folder and the Epoch_TS.csv file.
        
        events (List): List of the event positions.
        
        right_thresholds (float, optional): Value used to discard a trial if 
        it's not overpassed. Defaults to 2.95.
        
        left_threshold (float, optional): Value used to discard a trial if 
        it's not overpassed. Defaults to 1.95.
        
    """    
    session_epochs, lin_pos_tsd, position = load_data(dir_data)
    # pos_var = position.columns
    # pos_data = position.values
    pos_time = position.index
    speed_tsd = set_speed(position, pos_time)  # Create the speed vector

    for ep in events:
    
        # Get the interval for the current ep
        behaviour_epoch = IntervalSet(session_epochs.loc[ep, 0],
                                      session_epochs.loc[ep, 1],
                                      time_units='s')

        # We only want positions of the current ep, so we restrict the Tsd
        lin_pos_restricted = lin_pos_tsd.restrict(behaviour_epoch) 
        speed_restricted = speed_tsd.restrict(behaviour_epoch)
        
        t = lin_pos_restricted.times()
        lin_pos_points = lin_pos_restricted.data()
        speed_points = speed_restricted.data()
        
        # Initialize a helper variables
        valid_trials = []
        trials = pd.DataFrame(lin_pos_restricted)        
        trials.columns = ['position']
        trials.reset_index(inplace=True, drop=False)
        trials['color'] = 'orange'

        # directionn    --> 0: backward,  1: forward
        # arm           --> 0: left,  1: right
        trials['trial'], trials['arm'], trials['direction'] = -2, -2, -2

        idx_dep_on, idx_dep_off = 0, 0
        idx_reward, idx_reward_off = 0, 0
        min_position = 0
        trial_count = 1
        nbBins = len(t)

        # When does the animal exit the departure arm?
        # Where linear positions is greater than 1?
        idx_dep_off = np.where(lin_pos_points > 1)[0][0] - 1
        if idx_dep_off < 1:
            # in case there's some noise at the start of departure
            # find the minimum in departure
            min_position = np.where(lin_pos_points < 1)[0][0]
            trials.loc[:min_position, 'color'] = 'cyan'
            trials.loc[:min_position, 'direction'] = -1
            #  When does the animal exit the departure arm?
            idx_dep_off = min_position + np.where(
                lin_pos_points[min_position:] > 1)[0][0] - 1 
    
        trials.loc[min_position:idx_dep_off, 'color'] = 'pink'
        trials.loc[min_position:idx_dep_off, 'direction'] = 1

        while idx_dep_off < nbBins:
            trial_type = None
            trial_times = {}
            idx_start = segment_departure(idx_dep_off, idx_dep_on, 
                                          speed_points, trials, 
                                          min_position)
            trial_times['fw_dep_start'] = trials.loc[idx_start, 'Time (us)']
            trial_times['fw_dep_end'] = trials.loc[idx_dep_off, 'Time (us)']
            # when did he reenter the departure arm again(pos < 1) at the next trial?
            # we're skipping the first 100 steps, just in case there's some noise
            # note that if it's the last trial, possible that idx_dep_on = nbBins, but it's OK.
            idx_dep_off += 1  # to start where pos >= 1
            idx_dep_on = idx_dep_off + 100            

            try:
                idx_dep_on += np.where(lin_pos_points[idx_dep_on:] < 1)[0][0] - 1
            except IndexError:
                idx_dep_on = trials.index[-1]

            # max_position = round(max(lin_pos_points[idx_dep_off:idx_dep_on]))
            max_position = max(lin_pos_points[idx_dep_off:idx_dep_on])

            arm_color_forward = 'blue' if max_position >= right_threshold else 'red'
            arm_color_backward = 'orange' if max_position >= right_threshold else 'green'
            trials.loc[idx_dep_off:idx_dep_on, 'color'] = arm_color_forward
            trials.loc[idx_dep_off:idx_dep_on, 'direction'] = 1
            trial_times['fw_arm_start'] = trials.loc[idx_dep_off, 'Time (us)']
            
            # Did the animal reach one of the two arm ends?
            idx_reward = idx_dep_off
            
            if max_position >= right_threshold:
                # when did he reach the reward site?
                try:
                    idx_reward += np.where(lin_pos_points[idx_reward:] >= right_threshold)[0][0]
                except IndexError:
                    import pdb; pdb.set_trace()
                    
                trial_type = 1
                # when did he left the rew site?
                idx_reward_off = idx_reward
                idx_reward_off += np.where(lin_pos_points[idx_reward_off:] < right_threshold)[0][0] - 1
                # Reward zone
                trials.loc[idx_reward:idx_reward_off, 'color'] = 'cyan'
                trials.loc[idx_reward:idx_reward_off, 'direction'] = -1
                # Start of the backard
                idx_reward_off += 1
                trials.loc[idx_reward_off:idx_dep_on, 'color'] = arm_color_backward
                trials.loc[idx_reward_off:idx_dep_on, 'direction'] = 0
                
                trial_times['fw_arm_end'] = trials.loc[idx_reward-1, 'Time (us)']                 
                trial_times['bw_arm_start'] = trials.loc[idx_reward_off, 'Time (us)']
                trial_times['bw_arm_end'] = trials.loc[idx_dep_on, 'Time (us)']

            elif (max_position >= left_threshold) & (max_position < 2.15):
                
                # left trial, when did he reach the reward site?
                idx_reward += np.where(lin_pos_points[idx_reward:] >= left_threshold)[0][0] 
                trial_type = 0

                # when did he left the rew site?
                idx_reward_off = idx_reward
                idx_reward_off += np.where(lin_pos_points[idx_reward_off:] < left_threshold)[0][0] - 1
                # Reward zone
                trials.loc[idx_reward:idx_reward_off, 'color'] = 'cyan'
                trials.loc[idx_reward:idx_reward_off, 'direction'] = -1
                # Start of the backward
                idx_reward_off += 1
                trials.loc[idx_reward_off:idx_dep_on, 'color'] = arm_color_backward
                trials.loc[idx_reward_off:idx_dep_on, 'direction'] = 0
                
                trial_times['fw_arm_end'] = trials.loc[idx_reward-1, 'Time (us)']
                trial_times['bw_arm_start'] = trials.loc[idx_reward_off, 'Time (us)']
                trial_times['bw_arm_end'] = trials.loc[idx_dep_on, 'Time (us)']
                
            if trial_type is None:
                # Failed trail (did not reach the ends), look for the next departure
                print(f"Failed trial ({trial_count}), skipped. max_position: {max_position}")
                trials.loc[idx_start:idx_dep_on, 'color'] = "cyan"
                trials.loc[idx_start:idx_dep_on, 'direction'] = -1

                # Locate the next time where animal left departure                
                idx_dep_on += 1
                idx_dep_off = idx_dep_on
                try:
                    idx_dep_off += np.where(lin_pos_points[idx_dep_off:] >= 1)[0][0] -1
                except IndexError:
                    print(f'[Trial {trial_count}] End of the sequence.')
                    break

                # where is the next trial start?
                min_position = np.argmin(lin_pos_points[idx_dep_on:idx_dep_off])
                min_position += idx_dep_on
                # Remove the departure backward of the failed trial
                trials.loc[idx_dep_on:min_position, 'color'] = 'cyan'
                trials.loc[idx_dep_on:min_position, 'direction'] = -1

                # Start of the next trial
                trials.loc[min_position:idx_dep_off, 'color'] = 'pink'
                trials.loc[min_position:idx_dep_off, 'direction'] = 1

            else:
                # when is the next time the animal exit the dep arm?
                idx_dep_on += 1
                idx_dep_off = idx_dep_on
                # Where linear positions is greater than 1?
                try:
                    idx_dep_off += np.where(lin_pos_points[idx_dep_off:] >= 1)[0][0] -1
                except IndexError:
                    print(f'[Trial {trial_count}] End of the sequence.')
                    failed_trial_start = (trials[
                        trials['Time (us)'] ==  trial_times['fw_dep_start']
                        ].index[0])
                    trials.loc[failed_trial_start:, 'color'] = "cyan"
                    trials.loc[failed_trial_start:, 'direction'] = -1
                    break

                trials.loc[idx_dep_on:idx_dep_off, 'color'] = 'pink'
                trials.loc[idx_dep_on:idx_dep_off, 'direction'] = 1

                # when did he reach the end of the departure arm?
                if idx_dep_off < nbBins:
                    min_position = np.argmin(lin_pos_points[idx_dep_on:idx_dep_off]) 
                    min_position += idx_dep_on
                else:
                    min_position = idx_dep_on
                
                trials.loc[idx_dep_on:min_position, 'color'] = 'black'
                trials.loc[idx_dep_on:min_position, 'direction'] = 0
                
                trials.loc[idx_start:min_position, 'arm'] = trial_type
                trials.loc[idx_start:min_position, 'trial'] = trial_count
                trial_times['bw_dep_start'] = trials.loc[idx_dep_on, 'Time (us)']
                trial_times['bw_dep_end'] = trials.loc[min_position, 'Time (us)']
                trial_times['trial'] = trial_count
                trial_times['left0_right1'] = trial_type
                valid_trials.append(trial_times)
                
            trial_count += 1

        # plot(trials, dst=BASE_DIR/f"Analysis", filename=f'trials_{ep}.png')
        df_trials = pd.DataFrame.from_records(valid_trials)
        new_order = df_trials.columns.tolist()
        new_order.insert(0, new_order.pop(-2))
        df_trials = df_trials[new_order]
        plot_trials(trials, df_trials, dst=BASE_DIR/f"Analysis", filename=f'trials_{ep}.png')
        df_trials.to_csv(BASE_DIR/f"Analysis/trials_{ep}.csv", index=False)


def plot_trials(df: pd.DataFrame, df2: pd.DataFrame, dst: str = None, filename: str = None):
    df = df.set_index('Time (us)', drop=False)
    times = df2.iloc[:, 1:-1].values.flatten()
    positions = df.loc[times, 'position']
    
    df.plot.scatter(x='Time (us)', y='position', figsize=(16,7), alpha=0.5, c='color', s=1)
    plt.scatter(x=times, y=positions, c='magenta', s=30, alpha=0.4, marker='X')
    
    if dst:
        dst = str(dst/filename)
        plt.savefig(dst, format='png', dpi=600)
    plt.show()

if __name__=='__main__':
    # The base dir must have the Analysis folder and Epoch_TS.csv file 
    BASE_DIR = Path('/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/SandyReplayAnalysis/Data/A2908/A2908-190611')
    base_name = 'A2908-190611'
    make_trials(dir_data=BASE_DIR, events=[1,3], right_threshold=2.93, left_threshold=1.93) 
                

