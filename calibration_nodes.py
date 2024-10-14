import pandas as pd
import numpy as np
from typing import *
from pandas._typing import DataFrame
import json
from datetime import datetime, timedelta

from IQ_blobs_comparison import IQ_blobs_comparison
from qubit_power_error_amplification_class import Power_error_amplification
from ramsey_w_virtual_rotation import Ramsey_w_virtual_rotation, DEFAULT_TAUS
from readout_amplitude_binary_search import readout_amplitude_binary_search
from readout_duration_optimization import readout_duration_optimization
from readout_frequency_optimization import readout_frequency_optimization
from readout_weights_optimization import readout_weights_optimization

from multiplexed_configuration import *
from utils import *

DATAFRAME_FILE = "./calibration_database.pkl"
MAX_ATTEMPTS = 3
AMPLITUDE_CHANGE_THRESHOLD = 0.10  # 10% deviation tolerated
Q_FREQUENCY_CHANGE_THRESHOLD = 4000  # 1.2 kHz tolerated
RR_FREQUENCY_CHANGE_THRESHOLD = 30_000  # 10 kHz tolerated
READOUT_AMPLITUDE_CHANGE_THRESHOLD = 0.10  # 10% deviation tolerated
READOUT_DURATION_CHANGE_THRESHOLD = 0.05  # 5% deviation tolerated
IQ_THRESHOLD = 90  # 90% Readout fidelity tolerated

# Controls which calibration parameters are actively updated.
UPDATEABLE_PARAMETER_NAMES = [
    'pi_amplitude', 
    'pi_half_amplitude', 
    'IF', 
    'readout_amplitude',
    'readout_duration',
    'readout_frequency',
    # 'use_opt_readout',
]
# Account for slight difference in naming convention. See update_calibration_configuration docstring for details
SEARCH_PARAMETER_KEY_CORRESPONDENCE = {
    'pi_amplitude': 'pi_amplitude', 
    'pi_half_amplitude': 'pi_half_amplitude', 
    'IF': 'IF', 
    'readout_amplitude': 'amplitude',
    'readout_duration': 'readout_length',
    'readout_frequency': 'IF',
    'readout_fidelity': 'readout_fidelity',
    'use_opt_readout': 'use_opt_readout',
}

class Node: 
    """
    The base class for all calibrations performed in the pipeline
    
    The automated calibration pipeline designed and run in Run_Calibration.py relies on "nodes" that each have similar
    funationality (see that script for more details on the pipeline). The shared functionality of the nodes is defined
    in this class, including
    1) Shared database: The "shared_dataframe" variable loads the database only once, to reduce potential slow downs,
    then shared that database as a dataframe with all nodes in the pipeline script.
    2) Timing: refresh_time controls how long to wait between successful recalibration attempts. expiration_time
    sets the duration for which a calibrated parameter is still considered calibrated (unused currently). retry_time
    sets the period to wait before trying to recalibrate after a failed calibration attempt (unused currently).
    3) Freshness: Currently unused. Once implemented, freshness is a status to indicate to Users which nodes 
    successfully calibrated their respective parameters recently enough for the calibration to still be considered
    valid. This would be particularly useful when trying to visualize the current status of the calibrations.

    Methods: see details in the method docstrings
    4) calibrate: runs the calibration
    5) success_condition: determines success or failure of calibration attempt
    6) calibration_measurement: calls the experiment that performs the calibration measurement
    7) save_to_database: saves the latest calibration values and metadata to the database
    8) pull_latest_calibrated_values: pulls the latest calibrated value and metadata for a parameter from the database
    9) update_calibration_configuration: Because we are still using the configuration.py files Quantum Machines wrote
    we have a janky way to update the configuration files. This should be replaced with pulling only from the database

    """
    # Class variable shared by all instances, loaded from a file.
    # This way, we only have to load the dataframe once
    shared_dataframe: DataFrame = pd.read_pickle(DATAFRAME_FILE)

    def __init__(
        self,
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        update_calibration_config = True,
    ):
        """
        Initialize a node instance by defining key parameters.

        :param calibration_parameter_name: The name of the parameter pulled from and written to the database. 
        TODO: PRIORITY accept a list of parameter names, so more than one parameter can be calibrated by a single 
        measurement.
        :param qubits_to_calibrate: List of qubit names that is iterated through for each calibration
        :param refresh time: The time (seconds) to wait after a successful calibration, before attempting recalibration
        :param expiration time: Unused. The time (seconds) after which a successful calibration is considered invalid.
        :param retry_time: Unused. The time (seconds) after a failed calibration to attempt recalibration
        :param fresh: A status indicator. If True, the latest calibration is still considered valid.
        :param update_calibration_config: If True, successful calibrations of this parameter will overwrite the value
        used by the configuration.py and multiplex_configuration.py files. If False, the node runs as usual and saves 
        to the database, but the Quantum Machines configuration files will not be updated (useful for monitoring).
        """
        self.calibration_parameter_name = calibration_parameter_name
        self.refresh_time = refresh_time
        self.expiration_time = expiration_time
        self.retry_time = retry_time
        self.fresh = fresh
        self.qubits_to_calibrate = qubits_to_calibrate
        self.calibration_value = np.nan
        self.experiment_data_location = ''
        self.miscellaneous = {}
        self.current_qubit = ''
        self.calibration_success = False
        # Use class variable
        self.loaded_database = Node.shared_dataframe
        self.exception_log = []
        self.update_calibration_config = update_calibration_config


    def calibrate(
        self,
        initialize = False,
    ):
        """
        Perform the calibration.

        Loops over qubits to calibrate. Pulls the latest database entry to determine if it is time to calibrate each
        qubit. Makes at most MAX_ATTEMPTS number of attempts in a row to calibrate each qubit before giving up.
        Can be used for the first run of a node, when there are no database entries for that calibration, by setting
        initialize = True.
        
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because 1) the while loop terminates when it becomes True and 2) if there is an error in post-processing, it
        may make more sense to rerun the calibration or avoid saving a false positive. 

        TODO: Move all calls of self.save_to_database and self.success_condition to this function to enforce the order
        of method calls. Eg. save_to_database must preceed update_calibration_configuration

        :param initialize: Intended for starting a new calibration node. Set to True to ignore the database entries
        and just run a measurement and store its result in the database. Of course, this can be useful in other
        scenarios as well. TODO: Initialize does not skip the success_condition. This is an easy fix by promoting 
        initialize to an instance variable.
        """
        for qubit in self.qubits_to_calibrate:
            # Reset values for new qubit
            self.current_qubit = qubit
            self.exception_log = []
            self.miscellaneous = {}
            self.calibration_success = False  # If last qubit succeeded, we need this reset

            if not initialize:
                # Pull latest time of successful calibration
                latest_successful_entry = self.loaded_database.loc[
                    (self.loaded_database.calibration_parameter_name == self.calibration_parameter_name) &
                    (self.loaded_database.qubit_name == self.current_qubit) &
                    (self.loaded_database.calibration_success == True)
                ].iloc[-1]
                latest_time = latest_successful_entry['timestamp']
                refresh_time = timedelta(seconds=self.refresh_time)
                now_time = datetime.now()
            else:
                # Specific values unimportant
                latest_time = 100
                refresh_time = 0
                now_time = 0

            i_attempt = 0
            
            # If it's time to refresh this qubit, then run the measurement
            # If initialize = True, ignore timing and run.
            if (latest_time + refresh_time < now_time) or initialize:
                while i_attempt < MAX_ATTEMPTS and not self.calibration_success:
                    # If the last attempt (possibly by another qubit) overwrote these values
                    # we need to reset them
                    self.calibration_value = self.calibration_value = np.nan
                    self.experiment_data_location = ''

                    try:
                        self.calibration_measurement()
                    except Exception as e:
                        #TODO: Store entire traceback, not just the exception message
                        print(f'Failed Attempt. Try again. Exception: {e}')
                        self.exception_log.append(e)
                    i_attempt +=1

                if i_attempt >= MAX_ATTEMPTS:
                    # Save whatever we have to the database
                    print(f'Failed after {MAX_ATTEMPTS} attempts. Saving Exception log')
                    self.miscellaneous.update({'Exception Log': self.exception_log})
                    self.save_to_database()
                #TODO We should save the exception log regardless of success or failure
        self.update_calibration_configuration()
        return
    
        
    def success_condition(
            self, 
            calibration_value: float,
            threshold: float,
            percent_change_bool: bool = True
    ):
        """
        A Basic check for success by seeing if the new value is within a threshold of the old value. The threshold can
        be an absolute or a fractional value. Overwrite this method in the child class if fractional and absolute 
        comparison to the old value is not the correct method (eg. comparison to an minimum value).
        
        Regardless of the method used, the result must be stored in self.calibration_success to be used by the rest of
        the Node class structure.

        :param calibration_value: value obtained from the latest measurement.
        :param threshold: the value within which the new value must be relative to the old value.
        :param percent_change_bool: Set True to see if new value is within a certain percentage of the old value. False
        for new value within an absolute value of the old value.
        """
        self.calibration_success = False
        previous_calibration_value = self.loaded_database.loc[
            (self.loaded_database.calibration_parameter_name == self.calibration_parameter_name) &
            (self.loaded_database.qubit_name == self.current_qubit) &
            (self.loaded_database.calibration_success == True)
        ].iloc[-1]['calibration_value']

        if percent_change_bool:
            change = np.abs(calibration_value - previous_calibration_value)/previous_calibration_value
        else:
            change = np.abs(calibration_value - previous_calibration_value)


        if change < threshold:
            self.calibration_success = True
        #TODO: upgrade from print statement to logging
        print(f'calibration success: {self.calibration_success}')
    

    def calibration_measurement(self):
        """
        The calibrate method loops around this method, so it must be written for a since qubit calibration.
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method.
        See calibrate method's docstring for details.
        """
        raise Exception('calibration_measurement method must be defined by Node child class for a single qubit.')


    def save_to_database(self):
        """
        Save the latest results to the database file.

        dataframe columns: 
        timestamp, calibration_parameter_name, qubit_name, calibration_value, calibration_success, 
        experiment data location, miscellaneous
        """
        
        new_row = [
            datetime.now(), 
            self.calibration_parameter_name, 
            self.current_qubit, 
            self.calibration_value,
            self.calibration_success,
            self.experiment_data_location,  # Path/To/Files/
            self.miscellaneous,
        ]
        self.loaded_database.loc[len(self.loaded_database.index)] = new_row
        self.loaded_database.to_pickle(DATAFRAME_FILE)

    
    def pull_latest_calibrated_values(
        self,
        qubit: str,
        search_parameter_names: List[str],
        n_latest: int = 1
    ) -> DataFrame:
        """
        Assuming the database is loaded as a DataFrame, pull the n_latest calibration parameter database entries.
        This method is written more generally, so it can be used outside of the calibration pipeline run, if desired.

        :param qubit: qubit name of interest
        :param search_parameter_names: List of string of calibration_parameter_name to search for in the database.
        :param n_latest: Sets the maximum number of most-recent entries to return.
        :return: A Dataframe containing all found rows meeting the criteria.
        """

        df = self.loaded_database

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


    def update_calibration_configuration(self):
        """
        To make the pipeline structure compatible with the Quantum Machines codebase as it is, this function writes to
        a json file containing a dictionary of the calibration parameters. That dictionary is called by the 
        multiplexed_configuration.py and configuration.py files to overwrite parameters with the latest calibration
        values.
        Ideally, all of the Quantum Machines scripts would pull from the database directly. However, this code was 
        initially written to try to minimize the changes needed to the Quantum Machines code, leading to the current
        convoluted structure.
        TODO: write configuration files to pull from the database directly.
        NOTE: Quantum Machines uses some names for parameters that this author finds to be too vague. So I have used
        slightly different naming in the database vs the QM's configuration files. Because of this, the 
        SEARCH_PARAMETER_KEY_CORRESPONDENCE dictionary translates between the two naming conventions.
        """
        if self.update_calibration_config:

            with open('calibration_data_dict.json', 'r') as json_file:
                cal_dict = json.load(json_file)

            for qubit in self.qubits_to_calibrate:
                try:
                    database = self.pull_latest_calibrated_values(
                        qubit=qubit,
                        search_parameter_names=UPDATEABLE_PARAMETER_NAMES,
                    )
                except Exception as e:
                    print(e)
                # Overwrite with new calbirated values
                for param in UPDATEABLE_PARAMETER_NAMES:
                    if 'readout' in param:
                        cal_dict['RR_CONSTANTS'][qubit_resonator_correspondence[qubit]][\
                            SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                            database.loc[database.calibration_parameter_name == param][\
                            'calibration_value'].values[0]
                    else:
                        cal_dict['QUBIT_CONSTANTS'][qubit][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                            database.loc[database.calibration_parameter_name == param]['calibration_value'].values[0]

            with open('calibration_data_dict.json', 'w') as json_file:
                json.dump(cal_dict, fp=json_file)
    

#######################
#### Child Classes ####
#######################

class Qubit_Amplitude_Node(Node):
    """
    Calibrate the amplitude of qubit drive pulses.

    Works for both pi and pi-half pulses.

    TODO: Why didn't *arg work for pulling all of the parent class arguments? Debug this so we don't repeat the same
    arguments from the parent class explicitly.
    """
    def __init__(
        self, 
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        pulse_parameter_name: str = 'pi_', 
        nb_pulse_step: int = 2,
        a_min: float = 0.75,
        a_max: float = 1.25,
    ):
        """
        Initialize parameters for this class. The parent class arguments are the same.

        :param pulse_parameter_name: determines the type of pulse, whose amplitude is being calibrated. eg pi_ or 
        pi_half_
        :param nb_pulse_step: This technique sweeps over an integer multiple number of nb_pulse_step number of pulses.
        The measurement assumes all of the pulses amount to a total area being an integer multiple of 2pi. So if 
        calibrating pi pulses, nb_pulse_step = 2. If calibrating pi-half, it's 4, etc.
        :param a_min: Sets lower bound of the amplitude sweep to a fraction of the original amplitude.
        :param a_max: Sets upper bound of the amplitude sweep to a fraction of the original amplitude.
        """
        #TODO: Don't repeat parent class arguments explicitly
        super().__init__(
            calibration_parameter_name,
            qubits_to_calibrate,
            refresh_time,
            expiration_time,
            retry_time,
            fresh,
        )
        self.pulse_parameter_name = pulse_parameter_name
        self.nb_pulse_step = nb_pulse_step
        self.a_min = a_min
        self.a_max = a_max


    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        TODO: promote all success_condition and save_to_database calls to parent class
        """

        pea = Power_error_amplification(
            qubit = self.current_qubit,
            parameter_name = self.pulse_parameter_name,
        )
        fit_dict, data_folder = pea.power_rabi_pulse(
            a_min = self.a_min,
            a_max = self.a_max,
            nb_pulse_step = self.nb_pulse_step,
        )
        self.experiment_data_location = data_folder
        self.miscellaneous.update({'fit_dict': fit_dict})
        self.calibration_value = fit_dict['fit_values']['center']*fit_dict['scaled_original_amplitude']
        self.success_condition(self.calibration_value, AMPLITUDE_CHANGE_THRESHOLD)
        self.save_to_database()
        return fit_dict




class Qubit_Frequency_Node(Node):
    """
    Calibrate the frequency of qubit drive pulses.

    TODO: Why didn't *arg work for pulling all of the parent class arguments? Debug this so we don't repeat the same
    arguments from the parent class explicitly.
    """
    def __init__(
        self,  
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        n_avg: int = 1000,
        detuning: float = 2* u.MHz,  # in Hz
        taus: np.typing.NDArray = DEFAULT_TAUS,
    ):
        super().__init__(
            calibration_parameter_name,
            qubits_to_calibrate,
            refresh_time,
            expiration_time,
            retry_time,
            fresh,
        )
        self.n_avg = n_avg
        self.detuning = detuning
        self.taus = taus


    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        rvr = Ramsey_w_virtual_rotation(
            qubit = self.current_qubit,
            n_avg = self.n_avg,
            detuning = self.detuning,
            taus = self.taus,
        )
        fit_dict, data_folder = rvr.ramsey_w_virtual_rotation()
        self.experiment_data_location = data_folder
        self.miscellaneous.update({'fit_dict': fit_dict})
        self.calibration_value = QUBIT_CONSTANTS[self.current_qubit]["IF"] + fit_dict['qubit_detuning']
        self.success_condition(self.calibration_value, Q_FREQUENCY_CHANGE_THRESHOLD, False)
        self.save_to_database()
        return fit_dict



class Resonator_Amplitude_Node(Node):
    def __init__(
        self,  
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        n_avg: int = 10000,
    ):
        super().__init__(
            calibration_parameter_name,
            qubits_to_calibrate,
            refresh_time,
            expiration_time,
            retry_time,
            fresh,
        )
        self.n_avg = n_avg


    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        optimal_amplitude, data_folder = readout_amplitude_binary_search(
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        self.experiment_data_location = data_folder
        self.calibration_value = optimal_amplitude
        self.success_condition(
            calibration_value=self.calibration_value, 
            threshold=AMPLITUDE_CHANGE_THRESHOLD, 
            percent_change_bool=False
        )
        self.save_to_database()




class Resonator_Duration_Node(Node):
    def __init__(
        self,  
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        n_avg: int = 10000,
    ):
        super().__init__(
            calibration_parameter_name,
            qubits_to_calibrate,
            refresh_time,
            expiration_time,
            retry_time,
            fresh,
        )
        self.n_avg = n_avg


    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        opt_readout_length, data_folder = readout_duration_optimization(
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        self.experiment_data_location = data_folder
        self.calibration_value = opt_readout_length
        self.success_condition(self.calibration_value, READOUT_DURATION_CHANGE_THRESHOLD)
        self.save_to_database()




class Readout_Frequency_Node(Node):

    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        fit_dict, data_folder = readout_frequency_optimization(
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        self.experiment_data_location = data_folder
        self.miscellaneous.update({'fit_dict': fit_dict})
        self.calibration_value = fit_dict['fit_values']['center'] + RR_CONSTANTS[qubit_resonator_correspondence[self.current_qubit]]['IF']
        self.success_condition(self.calibration_value, RR_FREQUENCY_CHANGE_THRESHOLD, False)
        self.save_to_database()
        return fit_dict
    



class Readout_Weights_Node(Node):

    def success_condition(self):
        self.calibration_success = True
        print(f'calibration success: {self.calibration_success}')


    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        weights_dict, data_folder = readout_weights_optimization(            
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        self.experiment_data_location = data_folder
        self.miscellaneous.update({'weights_dict': weights_dict})
        self.success_condition()
        self.save_to_database()
        return weights_dict
    



class IQ_Blobs_Node(Node):

    def success_condition(self, fidelity: float):
        self.calibration_success = True if fidelity > IQ_THRESHOLD else False
        print(f'calibration success: {self.calibration_success}')
        

    def calibration_measurement(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """

        iq_blobs_dict = IQ_blobs_comparison(            
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        
        if iq_blobs_dict['optimized']['fidelity'] > iq_blobs_dict['rotated']['fidelity']:
            data_folder = iq_blobs_dict['optimized']['data_folder']
            fidelity = iq_blobs_dict['optimized']['fidelity']
            use_opt_readout = True
        else:
            data_folder = iq_blobs_dict['rotated']['data_folder']
            fidelity = iq_blobs_dict['rotated']['fidelity']
            use_opt_readout = False

        self.experiment_data_location = data_folder
        self.miscellaneous.update({'iq_blobs_dict': iq_blobs_dict})
        self.calibration_value = use_opt_readout
        self.success_condition(fidelity)
        self.save_to_database()

