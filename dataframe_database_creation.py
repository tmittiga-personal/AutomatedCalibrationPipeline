############################
### Create New DataFrame ###
############################
"""
NOTE: THIS WILL OVERWRITE ANY FILE SHARING THE FILENAME
A simple script to create a dataframe pkl file that serves as a database for the pipeline.
It populates the database with the appropriate columns, and initializes the parameters in the 
parameter_names list with NaN, [], {}, etc.
"""

import pandas as pd
from datetime import datetime
from multiplexed_configuration import *

DATAFRAME_FILE = "./calibration_database_new.pkl"
ALL_QUBITS = list(QUBIT_CONSTANTS.keys())
parameter_names = [
    'pi_amplitude', 
    'pi_half_amplitude', 
    'IF', 
    'readout_amplitude',
    'readout_duration',
    'readout_frequency',
    'readout_weights',
]
pl = len(parameter_names)
ql = len(ALL_QUBITS)

df_qubits_list = ALL_QUBITS*pl
df_parameters_list = parameter_names*ql

now_time = datetime.now()
data = {
    'timestamp': [now_time]*ql*pl,
    'calibration_parameter_name': df_parameters_list,
    'qubit_name': df_qubits_list,
    'calibration_value': [np.nan]*ql*pl,
    'calibration_success': [True]*ql*pl,
    'experiment_data_location': ['']*ql*pl,
    'miscellaneous': [{}]*ql*pl,
}

df = pd.DataFrame(data)
dfq = df.loc[df['qubit_name'] == 'q6_xy']
dfq.style

df.to_pickle(DATAFRAME_FILE)