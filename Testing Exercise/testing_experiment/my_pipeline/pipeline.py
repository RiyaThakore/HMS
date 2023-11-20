## codeblock 2

import numpy as np

def pipeline(filepath_data, filepath_parameters):
    """
    Fake neuroscience analysis pipeline.
    This code counts the number of spikes fired by neurons.
    The input data are timeseries traces of the voltage of individual neurons.
    The code applies some signal processing techniques to clean the data,
     then extracts and counts the number of peaks, which are assumed to be spikes.
    The output is a dataframe with the total number of spikes summed across all neurons
     during the trial periods for different experimental conditions.

    Required libraries:
        - pandas
        - numpy
        - scipy
        - matplotlib

    Args:
        filepath_data (str):
            Filepath to CSV file.
            CSV file should have one row of headers.
            Each column should correspond to a timeseries trace and the first row should be the name of that trace.
            The data should contain the 3 critical columns that correspond to temporal epochs within the experiment:
                - 'trial_on'
                - 'reward_on'
                - 'light_on'
            All other columns are expected to be the voltage traces of individual neurons.
        filepath_parameters (str):
            Filepath to JSON file that can be read into python as a python dictionary object.
            The parameters are expected to contain the following elements with defined 'key' names:
                - 'sample_rate' (float): frequency at which data samples were collected.
                - 'threshold' (float): voltage above which the voltage must reach to be considered a valid spike.
    """
    import pandas as pd
    import json
    import logging

    # Create empty dataframe to store final output
    result = pd.DataFrame()

    # Now try reading both the CSV and JSON input files from filepath_data and
    # filepath_parameters respectively. If any error comes up, simply log it
    # and return empty result.
    try:
      data = pd.read_csv(filepath_data)
    except Exception as ex:
      logging.exception('Error in reading filepath_data CSV file. Please check the stacktrace below for details.')
      return result

    try:
      with open(filepath_parameters, 'r') as f:
        parameters = json.load(f)
    except Exception as ex:
      logging.exception('Error in reading filepath_parameters JSON file. Please check the stacktrace below for details.')
      return result

    # Validate the data read above. If not as expected, return empty result.
    status, optional_error = validate_input(data, parameters)
    if not status:
      logging.exception('Input data validation failed due to: ' + optional_error)
      return result

    keys_trial = ['trial_on', 'reward_on', 'light_on']
    t, r, l = (data[key] for key in keys_trial)
    keys_neurons = [key for key in data.keys() if key not in keys_trial]
    # Use sample_rate and threshold value from extracted 'parameters'.
    st = [count_spikes(data[key], parameters['sample_rate'], parameters['threshold']) for key in keys_neurons] # 'spike_times'
    # st = [count_spikes(data[key]) for key in keys_neurons] # 'spike_times'
    st_cat = np.concatenate(st)

    bool_to_idx = lambda x: np.where(x)[0]    # Means only return 'True' indexes
    idx_conditions = {
        't': bool_to_idx(t), ## trial_on
        'r': bool_to_idx(r), ## reward_on
        'l': bool_to_idx(l), ## light_on
        'tr': bool_to_idx(t * r),
        'tl': bool_to_idx(t * l),
        'rl': bool_to_idx(r * l),
        'trl': bool_to_idx(t * r * l),
    }

    ns_conditions = {key: np.isin(st_cat, val).sum() for key, val in idx_conditions.items()}
    return ns_conditions


def count_spikes(trace, sample_rate=10000, threshold=10):
    import scipy

    ## smooth the trace
    w = sample_rate / 500 # roughly the n_samples of a spike
    w = int(w + np.remainder(w, 2)) # make odd
    trace_smooth = scipy.signal.savgol_filter(
        x=trace,
        window_length=w,
        polyorder=2,
    )

    ## find peaks (spike times)
    peaks, _ = scipy.signal.find_peaks(
        x=trace_smooth,
        height=threshold,
        distance=(2/1000 * sample_rate),
    )

    return peaks


def validate_input(data, parameters):
  # Input data must have:
  #   1. 3 critical column names: 'trial', 'reward' and 'light' tags
  #   2. above 3 columns are temporal epochs that should have a boolean value
  #   3. there is atleast 1 neuron voltage trace observation per row
  #   4. all neuron voltage trace columns should have numerical values
  for required_column in ['trial_on', 'reward_on', 'light_on']:
    if required_column not in data.keys():
      return False, 'Critical temporal epoch column: ' + required_column + ' not in input data'
    # Refered: https://docs.python.org/2/library/functions.html#isinstance
    for val in data[required_column]:
      if not isinstance(val, bool):
        return False, 'Critical temporal epoch column: ' + required_column + ' value type is not bool in input data'

  for column in data.keys():
    # For all other columns, it should be numerical (int/float) value
    if column not in ['trial_on', 'reward_on', 'light_on']:
      if np.isnan(data[column]).any():
        return False, 'Found invalue NaN value for column: ' + column + ' in input data'

  # Input parameters must have:
  #   1. Both 'sample_rate' and 'threshold' keys
  #   2. Value corresponding to each should be of 'float' type
  for required_column in ['sample_rate', 'threshold']:
    if required_column not in parameters:
      return False, 'Column ' + required_column + ' missing in input parameters'
    if not isinstance(parameters[required_column], float):
      return False, 'Column ' + required_column + ' value type is not float in input parameters'

  # If all validation checks pass, return true
  return True, ''

