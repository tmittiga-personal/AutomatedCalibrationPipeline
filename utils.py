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

    for i_df, param_name in enumerate(search_parameter_names):
        
        dfq = df.loc[
            (df.calibration_parameter_name == param_name) &
            (df.qubit_name in qubits) &
            (df.calibration_success == True)
        ].iloc[-1*n_latest:]
        
        if i_df == 0:
            df_found = pd.DataFrame(dfq)
        else:
            df_found = pd.concat([df_found, dfq], ignore_index=True)
    return df_found