import pandas as pd
from typing import *
import numpy as np

# Controls which calibration parameters are actively updated.
UPDATEABLE_PARAMETER_NAMES = [
    'pi_amplitude', 
    'pi_half_amplitude', 
    'IF', 
    'readout_amplitude',
    'readout_duration',
    'readout_frequency',
    # 'use_opt_readout',
    # 'readout_fidelity', 
    'readout_angle', 
    'readout_threshold'
]

# Account for slight difference in naming convention. 
# See calibration_nodes.py/update_calibration_configuration docstring for details
SEARCH_PARAMETER_KEY_CORRESPONDENCE = {
    'pi_amplitude': 'pi_amplitude', 
    'pi_half_amplitude': 'pi_half_amplitude', 
    'IF': 'IF', 
    'readout_amplitude': 'amplitude',
    'readout_duration': 'readout_length',
    'readout_frequency': 'IF',
    'readout_fidelity': 'readout_fidelity',
    'use_opt_readout': 'use_opt_readout',
    'readout_angle': 'rotation_angle',
    'readout_threshold': 'ge_threshold',
}


def pull_latest_calibrated_values(
    qubits: List[str],
    search_parameter_names: List[str],
    n_latest: int = 1
):
    assert isinstance(qubits, list), 'qubits must be a list of qubit names'

    DATAFRAME_FILE = "./calibration_database.pkl"
    df = pd.read_pickle(DATAFRAME_FILE)

    df_found = df.loc[
            (df['calibration_parameter_name'].isin(search_parameter_names)) &
            (df['qubit_name'].isin(qubits)) &
            (df.calibration_success == True)
        ].drop_duplicates(subset = ['qubit_name', 'calibration_parameter_name'], keep='last')
    
    return df_found

def thresholding(data_array, threshold):
    return np.array(data_array) > threshold

def wilson_score_interval(avg_thresholded_data, n, z=1.96):

    intervals = np.zeros((2,len(avg_thresholded_data)))
    for i_p, p in enumerate(avg_thresholded_data):
        intervals[0][i_p] = np.max([0, 
                                    ( 2*n*p + z**2 - (z * np.sqrt( z**2 - 1/n + 4*n*p*(1-p) + (4*p-2) ) + 1 ) ) / (2 *(n+z**2))
                                    ])
        intervals[1][i_p] = np.min([1,
                                    ( 2*n*p + z**2 + (z * np.sqrt( z**2 - 1/n + 4*n*p*(1-p) - (4*p-2) ) + 1 ) ) / (2 *(n+z**2))
                                    ])
    return intervals

def errorbars_from_intervals(avg_thresholded_data, intervals):
    """Assumes intervals contain lower and upper CIs, shaped (2,data_length)"""
    errorbars = np.zeros((2,len(avg_thresholded_data)))
    for i in range(2):
        errorbars[i] = np.abs(np.array(avg_thresholded_data) - np.array(intervals[i]))
    return errorbars