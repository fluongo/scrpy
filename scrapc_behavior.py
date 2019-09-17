import pandas as pd
import numpy as np

def load_csv(filename):
    '''
    Loads in data from filename
    '''
    df = pd.read_csv(filename, sep=';', engine='python', skiprows=6)
    # Convert to times to datetime
    df['PC-TIME'] = pd.to_datetime(df['PC-TIME'])
    return df

def parse_trials(sub_trials, trial_types, win=50, using_correction = False):
    '''
    Parse the trials of the full dataframe into a sub dataframe of the
    trials, outcomes, and the stimulus, as well as compute some moving averages

    win : Window to use for computing the moving average of performance
    Right now this only works for a pretty simple 2AFC
    '''

    # Now make a new dataframe of just n_trials which contains 3 columns
    # timestamp, outcome, stimulus

    # Check for inconsistent trial info
    if len(trial_types) != len(sub_trials): 
        trial_stimulus = list(np.zeros(len(sub_trials)))
    else:
        trial_stimulus = list(trial_types['MSG'])

    dfTrials = pd.DataFrame(data={'timestamp': list(sub_trials['PC-TIME']),
                                  'stimulus': trial_stimulus,
                                  'outcome': list(sub_trials['MSG'])})

    # Convert to numbers for outcome to compute percentages
    dfTrials.at[dfTrials['outcome'] == 'Reward', 'outcome'] = 1
    dfTrials.at[dfTrials['outcome'] == 'Punish', 'outcome'] = 0

    dfTrials['average_running'] = 0
    dfTrials['average_cumulative'] = 0
    dfTrials['percent_correct_left'] = 0
    dfTrials['percent_correct_right'] = 0
    n_stimuli = len(trial_types['MSG'].unique())
    print(n_stimuli)
    left_stim_str = [str(i) for i in range(1, int(n_stimuli/2)+1)]
    right_stim_str = [str(i)for i in range(int(n_stimuli/2)+1, n_stimuli+1)]

    # # Remove the correction trials
    # if using_correction:
    #     for i in range(len(dfTrials)):

    total_trials = len(dfTrials)
    # Fill in relevant stats
    for i in range(total_trials):
        dfTrials.loc[i, 'average_cumulative'] = sum(dfTrials.iloc[:i]['outcome']) / (i + 1)
        if i < win:
            dfTrials.loc[i, 'average_running'] = sum(dfTrials.iloc[:i]['outcome']) / (i + 1)
        else:
            dfTrials.loc[i, 'average_running'] = sum(dfTrials.iloc[i - win:i]['outcome']) / win
            # Computes the side bias e.g. what percentage of trials were left correct trials
            # Should stay as close to 50% as possible or else suggests bias
            total_left = sum((dfTrials.iloc[i - win:i]['outcome'] == 1) & (dfTrials.iloc[i - win:i]['stimulus'].isin(left_stim_str)))
            total_right = sum((dfTrials.iloc[i - win:i]['outcome'] == 1) & (dfTrials.iloc[i - win:i]['stimulus'].isin(right_stim_str)))

            dfTrials.loc[i, 'percent_correct_left'] = total_left / sum(dfTrials.iloc[i - win:i]['outcome'])
            dfTrials.loc[i, 'percent_correct_right'] = total_right / sum(dfTrials.iloc[i - win:i]['outcome'])

    return dfTrials