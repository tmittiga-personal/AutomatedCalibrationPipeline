import pandas as pd
import numpy as np
from typing import *
from datetime import datetime, timedelta

from qubit_power_error_amplification_class import Power_error_amplification
from ramsey_w_virtual_rotation import Ramsey_w_virtual_rotation, DEFAULT_TAUS
from readout_amplitude_optimization import readout_amplitude_optimization
from readout_duration_optimization import readout_duration_optimization
from readout_frequency_optimization import readout_frequency_optimization
from readout_weights_optimization import readout_weights_optimization

from multiplexed_configuration import *
from utils import *

DATAFRAME_FILE = "./calibration_database.pkl"
MAX_ATTEMPTS = 3
AMPLITUDE_CHANGE_THRESHOLD = 0.10  # 10% deviation tolerated
Q_FREQUENCY_CHANGE_THRESHOLD = 1200  # 1.2 kHz tolerated
RR_FREQUENCY_CHANGE_THRESHOLD = 10_000  # 10 kHz tolerated
READOUT_AMPLITUDE_CHANGE_THRESHOLD = 0.10  # 10% deviation tolerated
READOUT_DURATION_CHANGE_THRESHOLD = 0.05  # 5% deviation tolerated

# Controls which calibration parameters are actively updated.
SEARCH_PARAMETER_NAMES = [
    'pi_amplitude', 
    'pi_half_amplitude', 
    'IF', 
    # 'readout_amplitude',
    'readout_duration',
    'readout_frequency',
    'readout_weights',
]
# Account for slight difference in naming convention
SEARCH_PARAMETER_KEY_CORRESPONDENCE = {
    'pi_amplitude': 'pi_amplitude', 
    'pi_half_amplitude': 'pi_half_amplitude', 
    'IF': 'IF', 
    'readout_amplitude': 'amplitude',
    'readout_duration': 'readout_length',
    'readout_frequency': 'IF',
}

class Node: 
    # Class variable shared by all instances, loaded from a file.
    # This way, we only have to load the dataframe once
    shared_dataframe = pd.read_pickle(DATAFRAME_FILE)

    def __init__(
        self,
        calibration_parameter_name: str,
        qubits_to_calibrate: List[str],
        refresh_time: float,
        expiration_time: float,
        retry_time: float,
        fresh: bool = False,
        update_calibration_config = False,
    ):
        """
        Base class for calibration nodes. More explanation needed.
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
        self.loaded_database = Node.shared_dataframe
        self.exception_log = []
        self.update_calibration_config = update_calibration_config


    def calibrate(self):
        """
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """
        for qubit in self.qubits_to_calibrate:
            # Reset values for new qubit
            self.current_qubit = qubit
            self.exception_log = []
            self.miscellaneous = {}
            self.calibration_success = False  # If last qubit succeeded, we need this reset

            latest_successful_entry = self.loaded_database.loc[
                (self.loaded_database.calibration_parameter_name == self.calibration_parameter_name) &
                (self.loaded_database.qubit_name == self.current_qubit) &
                (self.loaded_database.calibration_success == True)
            ].iloc[-1]
            latest_time = latest_successful_entry['timestamp']
            refresh_time = timedelta(seconds=self.refresh_time)
            now_time = datetime.now()

            i_attempt = 0
            
            # If it's time to refresh this qubit, then run the measurement
            if latest_time + refresh_time < now_time:
                while i_attempt < MAX_ATTEMPTS and not self.calibration_success:
                    # If the last attempt (possibly by another qubit) overwrote these values
                    # we need to reset them
                    self.calibration_value = self.calibration_value = np.nan
                    self.experiment_data_location = ''

                    try:
                        self.calibration_measurement()
                    except Exception as e:
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
    

    def calibration_measurement(self):
        """
        calibrate method loops around this method, so it must be written for a since qubit calbiration.
        NOTE: self.calibration_success MUST be the last value set in the calbiration_measurement method
        because the while loop terminates when it becomes True.
        """
        raise Exception('Calibration method must be defined by Node child class')


    def save_to_database(self):
        # dataframe columns: timestamp, calibration_parameter_name, qubit_name, calibration_value, calibration_success, experiment data location, miscellaneous
        new_row = [
            datetime.now(), 
            self.calibration_parameter_name, 
            self.current_qubit, 
            self.calibration_value,
            self.calibration_success,
            self.experiment_data_location,
            self.miscellaneous,
        ]
        self.loaded_database.loc[len(self.loaded_database.index)] = new_row
        self.loaded_database.to_pickle(DATAFRAME_FILE)


    def update_calibration_configuration(self):
        if self.update_calibration_config:

            with open('calibration_data_dict.json', 'r') as json_file:
                cal_dict = json.load(json_file)

            for qubit in QUBIT_CONSTANTS.keys():
                try:
                    database = pull_latest_calibrated_values(
                        qubit=qubit,
                        SEARCH_PARAMETER_NAMES=SEARCH_PARAMETER_NAMES,
                    )
                except Exception as e:
                    print(e)
                # Overwrite with new calbirated values
                for param in SEARCH_PARAMETER_NAMES:
                    if param in ['readout_weights']:
                        continue
                    # print(database.loc[database.calibration_parameter_name == param]['calibration_value'].values[0])
                    if 'readout' in param:
                        cal_dict['RR_CONSTANTS'][qubit_resonator_correspondence[qubit]][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = database.loc[database.calibration_parameter_name == param]['calibration_value'].values[0]
                    else:
                        cal_dict['QUBIT_CONSTANTS'][qubit][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = database.loc[database.calibration_parameter_name == param]['calibration_value'].values[0]

            with open('calibration_data_dict.json', 'w') as json_file:
                json.dump(cal_dict, fp=json_file)
    



class Qubit_Amplitude_Node(Node):
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
        self.calibration_value = fit_dict['fit_values']['center']*fit_dict['original_amplitude']
        self.success_condition(self.calibration_value, AMPLITUDE_CHANGE_THRESHOLD)
        self.save_to_database()
        return fit_dict




class Qubit_Frequency_Node(Node):
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

        optimal_amplitude, data_folder = readout_amplitude_optimization(
            qubit = self.current_qubit,
            resonator = qubit_resonator_correspondence[self.current_qubit]
        )
        self.experiment_data_location = data_folder
        self.calibration_value = optimal_amplitude
        self.success_condition(self.calibration_value, 1)
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
        self.calibration_value = fit_dict['fit_values']['center']
        self.success_condition(self.calibration_value, RR_FREQUENCY_CHANGE_THRESHOLD, False)
        self.save_to_database()
        return fit_dict
    



class Readout_Weights_Node(Node):

    def success_condition(self):
        self.calibration_success = True


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