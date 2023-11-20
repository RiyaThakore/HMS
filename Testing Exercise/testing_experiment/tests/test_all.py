## Tests

from pathlib import Path
import tempfile
import warnings
import pytest
import numpy as np
import pandas as pd
import json

from my_pipeline import pipeline

# Based on the reference https://github.com/RichieHakim/ROICaT/blob/main/tests/test_*.py
# we could ideally add more tests for individual packages,
# environments (python version, etc.) and more but is intentionally kept
# out-of-scope for now.

dir_parent = Path(__file__).parent
test_filepath_data = str('..' / dir_parent / "fake_data.csv")
test_filepath_parameters = str('..' / dir_parent / "fake_params.json")

# Part 1

def test_importing_packages():
    """
    Runs pytest on the core packages.
    """
    corePackages = [
      'pandas',
      'json',
      'logging',
      'scipy',
      'numpy'
    ]
    for pkg in corePackages:
      try:
          exec(f'import {pkg}')
      except ModuleNotFoundError:
          warnings.warn(f'RH Warning: {pkg} not found. Skipping tests.')


def test_invalid_input_files():
  # For all invalid inputs, pipeline() should output empty result and not fail.

  assert pipeline.pipeline('', test_filepath_parameters).empty, 'Invalid filepath_data input resulted in non-empty output'
  assert pipeline.pipeline(test_filepath_data, '').empty, 'Invalid filepath_parameters input resulted in non-empty output'
  assert pipeline.pipeline('', '').empty, 'Invalid filepath inputs resulted in non-empty output'

  empty_file = str((Path(tempfile.gettempdir()) / 'empty_file.txt').resolve().absolute())
  open(empty_file, 'w').close() # Create dummy empty file
  assert pipeline.pipeline(empty_file, test_filepath_parameters).empty, 'Empty filepath_data input resulted in non-empty output'
  assert pipeline.pipeline(test_filepath_data, empty_file).empty, 'Empty filepath_parameters input resulted in non-empty output'


# Referred https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html for 'drop' function.
def test_invalid_data_in_filepath_data():
  # For all invalid inputs, pipeline() should output empty result and not fail.

  invalid_filepath_data = str((Path(tempfile.gettempdir()) / 'invalid_filepath_data.csv').resolve().absolute())
  valid_data = pd.read_csv(test_filepath_data)

  # Ensure all 3 critical columns are part of the filepath_data file.
  invalid_data = valid_data.drop('trial_on', axis=1)
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: trial_on column missing in filepath_data file. Non-empty output is not expected.'

  invalid_data = valid_data.drop('reward_on', axis=1)
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: trial_on column missing in filepath_data file. Non-empty output is not expected.'

  invalid_data = valid_data.drop('light_on', axis=1)
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: trial_on column missing in filepath_data file. Non-empty output is not expected.'

  # These 3 columns should have bool values
  invalid_data = valid_data
  invalid_data.at[5, 'trail_on'] = np.NaN  # change to NaN for testing
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: trial_on column contains non-boolean value in filepath_data file. Non-empty output is not expected.'

  # Ensure there is atleast 1 neuron voltage trace observation per row.
  invalid_data = valid_data.drop(['neuron_1', 'neuron_2', 'neuron_3', 'neuron_4', 'neuron_5', 'neuron_6', 'neuron_7'], axis=1)
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: neuron voltage trace columns NOT found in filepath_data file. Non-empty output is not expected.'

  # Neuron voltage trace columns should be numeric
  invalid_data = valid_data
  invalid_data.at[15, 'neuron_5'] = np.NaN  # change to NaN for testing
  invalid_data.to_csv(invalid_filepath_data, index=False)
  assert pipeline.pipeline(invalid_filepath_data, test_filepath_parameters).empty, 'Invalid data: neuron voltage trace columns contains NaN value in filepath_data file. Non-empty output is not expected.'


def test_invalid_data_in_filepath_parameters():
  # For all invalid inputs, pipeline() should output empty result and not fail.

  invalid_filepath_params = str((Path(tempfile.gettempdir()) / 'invalid_filepath_params.csv').resolve().absolute())
  with open(test_filepath_parameters, 'r') as f:
        valid_parameters = json.load(f)

  # Verify both keys 'sample_rate' and 'threshold' are present.
  invalid_params = valid_parameters
  del invalid_params['sample_rate']
  with open(invalid_filepath_params, 'w') as f:
        json.dump(invalid_params, f)
  assert pipeline.pipeline(test_filepath_data, invalid_filepath_params).empty, 'Invalid data: sample_rate key missing in filepath_params file. Non-empty output is not expected.'

  invalid_params = valid_parameters
  del invalid_params['threshold']
  with open(invalid_filepath_params, 'w') as f:
        json.dump(invalid_params, f)
  assert pipeline.pipeline(test_filepath_data, invalid_filepath_params).empty, 'Invalid data: threshold key missing in filepath_params file. Non-empty output is not expected.'

  # Verify that value corresponding to both keys are float type.
  invalid_params = valid_parameters
  invalid_params['sample_rate'] = np.NaN
  with open(invalid_filepath_params, 'w') as f:
        json.dump(invalid_params, f)
  assert pipeline.pipeline(test_filepath_data, invalid_filepath_params).empty, 'Invalid data: sample_rate key has NaN value in filepath_params file. Non-empty output is not expected.'


# Part 2: Accuracy testing

def test_all_neuron_voltage_trace_above_threshold_scenario():
  # make dummy filepath_data and write to the file
  data = {}
  data['trial_on'] = [True, False, False, False, True, True]
  data['reward_on'] = [False, True, False, True, True, True]
  data['light_on'] = [False, False, True, True, False, True]
  data['neuron_1'] = [0, 1, 0, 1, 1, 0]
  data['neuron_2'] = [-1, 1, 1, 0, 0, -1]
  data['neuron_3'] = [1, 0, -1, 1, -1, -1]
  data_df = pd.DataFrame(data)
  testing_data_file = str((Path(tempfile.gettempdir()) / 'test_data_file.txt').resolve().absolute())
  data_df.to_csv(testing_data_file, index=False)

  # make dummy filepath_data and write to the file
  parameters = {
      'sample_rate': float(1500),
      'threshold': float(-100)
  }
  testing_params_file = str((Path(tempfile.gettempdir()) / 'test_params_file.txt').resolve().absolute())
  with open(testing_params_file, 'w') as f:
      json.dump(parameters, f)

  # expected output for above input is easy to compute
  expected_result = {
    't': 0, 'r': 3, 'l': 2, 'tr': 0, 'tl': 0, 'rl': 2, 'trl': 0
  }

  actual_result = pipeline.pipeline(testing_data_file, testing_params_file)
  assert actual_result == expected_result, 'Test failed. Expected: ' + str(expected_result) + ', Actual: ' + str(actual_result)


def test_all_neuron_voltage_trace_below_threshold_scenario():
  # make dummy filepath_data and write to the file
  data = {}
  data['trial_on'] = [True, False, False, False, True, True]
  data['reward_on'] = [False, True, False, True, True, True]
  data['light_on'] = [False, False, True, True, False, True]
  data['neuron_1'] = [0, 1, 0, 1, 1, 0]
  data['neuron_2'] = [-1, 1, 1, 0, 0, -1]
  data['neuron_3'] = [1, 0, -1, 1, -1, -1]
  data_df = pd.DataFrame(data)
  testing_data_file = str((Path(tempfile.gettempdir()) / 'test_data_file.txt').resolve().absolute())
  data_df.to_csv(testing_data_file, index=False)

  # make dummy filepath_data and write to the file
  parameters = {
      'sample_rate': float(1500),
      'threshold': float(100)
  }
  testing_params_file = str((Path(tempfile.gettempdir()) / 'test_params_file.txt').resolve().absolute())
  with open(testing_params_file, 'w') as f:
      json.dump(parameters, f)

  # expected output for above input is easy to compute
  expected_result = {
    't': 0, 'r': 0, 'l': 0, 'tr': 0, 'tl': 0, 'rl': 0, 'trl': 0
  }

  actual_result = pipeline.pipeline(testing_data_file, testing_params_file)
  assert actual_result == expected_result, 'Test failed. Expected: ' + str(expected_result) + ', Actual: ' + str(actual_result)


def test_common_neuron_voltage_trace_threshold_scenario():
  # make dummy filepath_data and write to the file
  data = {}
  data['trial_on'] = [True, True, True, True, True, True]
  data['reward_on'] = [True, True, True, True, True, True]
  data['light_on'] = [True, True, True, True, True, True]
  data['neuron_1'] = [0, 1, 0, 1, 1, 0]
  data['neuron_2'] = [-1, 1, 1, 0, 0, -1]
  data['neuron_3'] = [1, 0, -1, 1, -1, -1]
  data_df = pd.DataFrame(data)
  testing_data_file = str((Path(tempfile.gettempdir()) / 'test_data_file.txt').resolve().absolute())
  data_df.to_csv(testing_data_file, index=False)

  # make dummy filepath_data and write to the file
  parameters = {
      'sample_rate': float(1500),
      'threshold': float(0)
  }
  testing_params_file = str((Path(tempfile.gettempdir()) / 'test_params_file.txt').resolve().absolute())
  with open(testing_params_file, 'w') as f:
      json.dump(parameters, f)

  # expected output for above input is easy to compute
  expected_result = {
    't': 3, 'r': 3, 'l': 3, 'tr': 3, 'tl': 3, 'rl': 3, 'trl': 3
  }

  actual_result = pipeline.pipeline(testing_data_file, testing_params_file)
  assert actual_result == expected_result, 'Test failed. Expected: ' + str(expected_result) + ', Actual: ' + str(actual_result)


# def run_all_tests():
#   test_importing_packages()
#   test_invalid_input_files()
#   test_invalid_data_in_filepath_data()
#   test_invalid_data_in_filepath_parameters()
#   test_all_neuron_voltage_trace_above_threshold_scenario()
#   test_all_neuron_voltage_trace_below_threshold_scenario()
#   test_common_neuron_voltage_trace_threshold_scenario()
