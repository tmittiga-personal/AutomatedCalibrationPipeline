from multiplexed_configuration import *
import pandas as pd

qubit_resonator_correspondence = {qu: res for qu, res in zip(QUBIT_CONSTANTS.keys(), RR_CONSTANTS.keys())}

def pull_latest_calibrated_values(
    qubit,
    search_parameter_names,
    n_latest = 1
):

    DATAFRAME_FILE = "./calibration_database.pkl"
    df = pd.read_pickle(DATAFRAME_FILE)

    for i_df, param_name in enumerate(search_parameter_names):
        
        dfq = df.loc[
            (df.calibration_parameter_name == param_name) &
            (df.qubit_name == qubit) &
            (df.calibration_success == True)
        ].iloc[-1*n_latest:]
        
        if i_df == 0:
            df_found = pd.DataFrame(dfq)
        else:
            df_found = pd.concat([df_found, dfq], ignore_index=True)
    return df_found