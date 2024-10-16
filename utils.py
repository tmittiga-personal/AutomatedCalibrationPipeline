import pandas as pd
from typing import *

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