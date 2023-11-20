## codeblock 1

def make_fake_parameters(filepath_save=r'~/Desktop/parameters.json'):
    """
    Make a fake parameters file.

    Args:
        filepath_save (str):
            Filepath to save the parameters file.
    """
    import json
    import numpy as np

    parameters = {
        'sample_rate': float(2000),
        'threshold': float(np.random.randint(0, 100))
    }

    with open(filepath_save, 'w') as f:
        json.dump(parameters, f)

def make_fake_data_file(filepath_save=r'~/Desktop/data.csv'):
    """
    Make a fake data file.

    Args:
        filepath_save (str):
            Filepath to save the data file.
    """
    import pandas as pd
    import numpy as np

    N = 100000

    data = {}

    data['trial_on'] = ((np.arange(N) % 1000) > 500).astype(np.bool_)
    data['reward_on'] = ((np.arange(N) % 1000) > 750).astype(np.bool_)
    data['light_on'] = ((np.arange(N) % 2000) > 900).astype(np.bool_)

    voltages = {f"neuron_{ii + 1}": make_fake_voltage_trace(N) for ii in range(7)}

    ## Make dataframe with named columns
    data.update(voltages)
    data_df = pd.DataFrame(data)

    ## Save to CSV
    data_df.to_csv(filepath_save, index=False)

def make_fake_voltage_trace(n):
    """
    Poisson spike train with some noise.
    Baseline voltage: -70 mV
    Spike voltage: +80 mV
    Spike width: 2 ms
    Noise: Gaussian with mean 0 and std 5 mV
    """
    import numpy as np
    import scipy.signal

    baseline_voltage = -70
    spike_voltage = 100
    threshold_voltage = -45
    refractory_period = 40 # ms
    spike_width = 1 # ms (std)
    noise_mean = 0
    variance = 8
    noise_std = 10

    sample_rate = 10000 # Hz

    ## make a randomly varying trace
    voltage_trace = np.random.normal(0, variance, n) + baseline_voltage
    spike_bool = voltage_trace > threshold_voltage
    ## remove spikes within the refractory period
    spike_times = np.where(spike_bool)[0].astype(np.float32)
    spike_times[np.where(np.diff(spike_times, prepend=0) < refractory_period)[0]] = np.nan
    spike_times = spike_times[~np.isnan(spike_times)]
    ## make a new boolean trace with the refractory spikes removed
    spike_bool = np.isin(np.arange(n), spike_times)
    ## convolve boolean with gaussian to approximate the timecourse of a spike
    spike_kernel = scipy.signal.windows.gaussian(spike_width/1000 * sample_rate * 5, spike_width/1000 * sample_rate) * spike_voltage
    spike_kernel = spike_kernel / np.max(spike_kernel)
    voltage_trace += (np.convolve(spike_bool, spike_kernel, mode='same') * spike_voltage) + np.random.normal(noise_mean, noise_std, n)

    return voltage_trace


# Write fake data to filepath_data and filepath_params files for testing

from pathlib import Path

dir_parent = Path(__file__).parent

test_filepath_data = str(dir_parent / "data.csv")
test_filepath_parameters = str(dir_parent / "params.json")

make_fake_data_file(test_filepath_data)
make_fake_parameters(test_filepath_parameters)
