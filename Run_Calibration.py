"""
Define and run the calibration pipeline.
First, call a Node child class instance and set its timings.
Then, loop over each node's calibrate method. In this loop, you can define logic to determine relationships between 
nodes. TODO: write parent-child logic into the Node class to determine which nodes require another node to run first.
TODO: Use asyncio to permit running this script in the background and interleaving with User experiments.
"""

from calibration_nodes import *
import time

ALL_QUBITS = QUBIT_CONSTANTS.keys()
CALIBRATION_QUBITS = ["q1_xy", "q3_xy", "q5_xy"] #"q3_xy", "q1_xy", 
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
    calibration_parameter_name = ['readout_amplitude','readout_fidelity', 'readout_angle', 'readout_threshold'],
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
    calibration_parameter_name = ['use_opt_readout', 'readout_fidelity', 'readout_angle', 'readout_threshold'],
    qubits_to_calibrate = CALIBRATION_QUBITS,
    refresh_time = 3600*3,
    expiration_time = 3600*24,
    retry_time = 60*5,
)

###########
### RUN ###
###########

# Set the order of nodes
# You can also implement any dependency logic here.
initialize_bool = True

if __name__ == "__main__":

    while True:
        while is_valid_time():
            
            qubit_frequency_node.calibrate(initialize=initialize_bool)

            pi_amplitude_node.calibrate(initialize=initialize_bool)

            pi_half_amplitude_node.calibrate(initialize=initialize_bool)

            resonator_frequency_node.calibrate(initialize=initialize_bool)

            resonator_duration_node.calibrate(initialize=initialize_bool)

            resonator_amplitude_node.calibrate(initialize=initialize_bool)

            # resonator_weights_node.calibrate()

            # iq_blobs_node.calibrate()

            time.sleep(5*60)  # probe if a calibration is needed every 60 seconds
        
