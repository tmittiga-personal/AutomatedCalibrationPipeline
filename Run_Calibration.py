from calibration_nodes import *
import time

ALL_QUBITS = QUBIT_CONSTANTS.keys()
CALIBRATION_QUBITS = ["q1_xy","q3_xy","q5_xy"] #"q3_xy",
CALIBRATION_TIME_WINDOW = [datetime.strptime("17:00", "%H:%M").time(), datetime.strptime("17:00", "%H:%M").time()]

def is_valid_time():
    now_time = datetime.now().time()
    return (now_time >= CALIBRATION_TIME_WINDOW[0]) or (now_time < CALIBRATION_TIME_WINDOW[1]) 

####################
### DEFINE NODES ###
####################

pi_amplitude_node = Qubit_Amplitude_Node(
    calibration_parameter_name = 'pi_amplitude',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

pi_half_amplitude_node = Qubit_Amplitude_Node(
    calibration_parameter_name = 'pi_half_amplitude',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
    # Amplitude-Node-specific arguments
    pulse_parameter_name = 'pi_half_', 
    nb_pulse_step = 4,
)

qubit_frequency_node = Qubit_Frequency_Node(
    calibration_parameter_name = 'IF',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

resonator_frequency_node = Readout_Frequency_Node(    
    calibration_parameter_name = 'readout_frequency',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

resonator_amplitude_node = Resonator_Amplitude_Node(    
    calibration_parameter_name = 'readout_amplitude',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

resonator_duration_node = Resonator_Duration_Node(    
    calibration_parameter_name = 'readout_duration',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

resonator_weights_node = Readout_Weights_Node(    
    calibration_parameter_name = 'readout_weights',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

iq_blobs_node = IQ_Blobs_Node(
    calibration_parameter_name = 'use_opt_readout',
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

###########
### RUN ###
###########

if __name__ == "__main__":

    while True:
        while is_valid_time():
            pi_amplitude_node.calibrate()

            pi_half_amplitude_node.calibrate()

            qubit_frequency_node.calibrate()

            resonator_frequency_node.calibrate()

            resonator_duration_node.calibrate()

            resonator_weights_node.calibrate()

            iq_blobs_node.calibrate()

            time.sleep(60)  # probe if calibration is needed every 60 seconds
        
