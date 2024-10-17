from pathlib import Path
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from set_octave import OctaveUnit, octave_declaration
from utils import *

ALL_QUBIT_NAMES = ["q1_xy", "q2_xy", "q3_xy", "q4_xy", "q5_xy", "q6_xy"]
CALIBRATION_QUBITS = ["q1_xy","q3_xy","q5_xy"]
# True to overwrite values with automated calbiration values
# False to use values as written in this file
use_calibrated_values = True  
if use_calibrated_values:
    calibration_dataframe = pull_latest_calibrated_values(
        qubits=ALL_QUBIT_NAMES,
        search_parameter_names=UPDATEABLE_PARAMETER_NAMES,
    )


#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "192.168.88.250"  # Write the QM router IP address
octave_ip = "192.168.88.251"

cluster_name = "Cluster_1"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

# Path to save data
save_dir = Path().absolute() / "QM" / "INSTALLATION" / "data"

############################
# Set octave configuration #
############################

# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", octave_ip, port=80, con="con1")
# octave_2 = OctaveUnit("octave2", qop_ip, port=11051, con="con1")

# If the control PC or local network is connected to the internal network of the QM router (port 2 onwards)
# or directly to the Octave (without QM the router), use the local octave IP and port 80.
# octave_ip = "192.168.88.X"
# octave_1 = OctaveUnit("octave1", octave_ip, port=80, con="con1")

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
# OPX configuration #
#####################
# CW pulse parameter
const_len = 1000
const_amp = 125 * u.mV

########
# TWPA #
########

twpa_status = True

# MARK: QUBITS
#############################################
#                  Qubits                   #
#############################################
qubit_rotation_keys = ["x180", "x90", "minus_x90", "y180", "y90", "minus_y90"]
qubit_LO = 3.3 * u.GHz
# Constants for Pi Pulse
PI_LENGTH = 100
PI_SIGMA = PI_LENGTH / 5

qubit_octave_gain = 6
amplitude_scaling = 0.5

MULTIPLEX_DRIVE_CONSTANTS = {
    "drive1": {
        "QUBITS": ["q1_xy", "q2_xy", "q3_xy", "q4_xy", "q5_xy", "q6_xy"],
        "LO": qubit_LO,
        "con": "con1",
        "octave": "octave1",
        "RF_port": 2,
        "delay": 0,
    }
}

# Constants for each qubit (replace example values with actual values)
QUBIT_CONSTANTS = {
    ALL_QUBIT_NAMES[0]: {
        "pi_amplitude": 0.3017, 
        "pi_half_amplitude": 0.3017/2,
        "pi_len": 160,
        "pi_half_len": 160,
        "pi_sigma": 160/5,
        "anharmonicity": -200 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": -341403900,
    },
    ALL_QUBIT_NAMES[1]: {
        "pi_amplitude": 0.0414,
        "pi_half_amplitude": 0.0414/2,
        "pi_len": PI_LENGTH,
        "pi_half_len": PI_LENGTH,
        "pi_sigma": PI_SIGMA, 
        "anharmonicity": -180 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": -88 * u.MHz,
    },
    ALL_QUBIT_NAMES[2]: {
        "pi_amplitude": 0.289,
        "pi_half_amplitude": 0.289/2,
        "pi_len": PI_LENGTH,
        "pi_half_len": PI_LENGTH,
        "pi_sigma": PI_SIGMA, 
        "anharmonicity": -190 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": 31.14558 * u.MHz + 0.01 * u.MHz,
    },
    ALL_QUBIT_NAMES[3]: {
        "pi_amplitude": 0.0944/3,
        "pi_half_amplitude": 0.0944/2,
        "pi_len": PI_LENGTH,
        "pi_half_len": PI_LENGTH,
        "pi_sigma": PI_SIGMA, 
        "anharmonicity": -185 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": -31 * u.MHz,
    },
    ALL_QUBIT_NAMES[4]: {
        "pi_amplitude": 0.2273,
        "pi_half_amplitude": 0.2273/2,
        "pi_len": PI_LENGTH,
        "pi_half_len": PI_LENGTH,
        "pi_sigma": PI_SIGMA, 
        "anharmonicity": -150 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": 141.9014 * u.MHz,
    },
    ALL_QUBIT_NAMES[5]: {
        "pi_amplitude": 0.1797,
        "pi_half_amplitude": 0.1797/2,
        "pi_len": PI_LENGTH,
        "pi_half_len": PI_LENGTH,
        "pi_sigma": PI_SIGMA, 
        "anharmonicity": -150 * u.MHz,
        "drag_coefficient": 0.0,
        "ac_stark_shift": 0.0 * u.MHz,
        "IF": 322.30683 * u.MHz -78*u.kHz,
    },
}
if use_calibrated_values:
    for qubit_key, constants in zip(CALIBRATION_QUBITS, QUBIT_CONSTANTS.values()):
        for param in UPDATEABLE_PARAMETER_NAMES:
            if SEARCH_PARAMETER_KEY_CORRESPONDENCE[param] in ["pi_amplitude", "pi_half_amplitude"]:
                QUBIT_CONSTANTS[qubit_key][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                    calibration_dataframe.loc[
                        (calibration_dataframe.qubit_name == qubit_key) &
                        (calibration_dataframe.calibration_parameter_name == param)
                    ]['calibration_value'].values[0]*amplitude_scaling
            elif SEARCH_PARAMETER_KEY_CORRESPONDENCE[param] == 'IF':
                # Due to overlap in names for qubit and resonator, we need to be specific
                QUBIT_CONSTANTS[qubit_key][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                    calibration_dataframe.loc[
                        (calibration_dataframe.qubit_name == qubit_key) &
                        (calibration_dataframe.calibration_parameter_name == 'IF')
                    ]['calibration_value'].values[0]
            elif SEARCH_PARAMETER_KEY_CORRESPONDENCE[param] in constants.keys():
                QUBIT_CONSTANTS[qubit_key][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                    calibration_dataframe.loc[
                        (calibration_dataframe.qubit_name == qubit_key) &
                        (calibration_dataframe.calibration_parameter_name == param)
                    ]['calibration_value'].values[0]
    print('QUBIT_CONSTANTS pulled from calibration_data_dict.json')

# Relaxation time
qubit_T1 = int(200 * u.us)
thermalization_time = 5 * qubit_T1

# Saturation_pulse
saturation_len = 50 * u.us
saturation_amp = 0.45

def generate_waveforms(rotation_keys):
    """ Generate all necessary waveforms for a set of rotation types across all qubits. """
    
    if not isinstance(rotation_keys, list):
        raise ValueError("rotation_keys must be a list")

    waveforms = {}

    for qubit_key, constants in QUBIT_CONSTANTS.items():
        amp = constants["pi_amplitude"]
        ph_amp = constants["pi_half_amplitude"]
        pi_len = constants["pi_len"]
        pi_sigma = constants["pi_sigma"]
        drag_coef = constants["drag_coefficient"]
        ac_stark_shift = constants["ac_stark_shift"]
        anharmonicity = constants["anharmonicity"]

        for rotation_key in rotation_keys:
            if rotation_key in ["x180", "y180"]:
                wf_amp = amp
            elif rotation_key in ["x90", "y90"]:
                wf_amp = ph_amp
            elif rotation_key in ["minus_x90", "minus_y90"]:
                wf_amp = -ph_amp
            else:
                continue

            wf, der_wf = np.array(drag_gaussian_pulse_waveforms(wf_amp, pi_len, pi_sigma, drag_coef, anharmonicity, ac_stark_shift))

            if rotation_key.startswith("x") or rotation_key == "minus_x90":
                I_wf = wf
                Q_wf = der_wf
            else:  # y rotations
                I_wf = (-1) * der_wf
                Q_wf = wf

            waveforms[f"{qubit_key}_{rotation_key}_I"] = I_wf
            waveforms[f"{qubit_key}_{rotation_key}_Q"] = Q_wf

    return waveforms

waveforms = generate_waveforms(qubit_rotation_keys)

# MARK: RESONATORS
#############################################
#                Resonators                 #
#############################################
readout_len = 3000  # Only used by sin and cosine integration weights for non-resonator-specific readout.
depletion_time = 10 * u.us
resonator_octave_gain = 0


RL_CONSTANTS = {
    "rl1": {
        "LO": 7.0 * u.GHz,
        "RESONATORS": ["q1_rr", "q2_rr", "q3_rr", "q4_rr", "q5_rr", "q6_rr"],
        "TOF": 24 + 272,
        "rl_con": "con1",
        "delay": 0,
        "rl_octave": "octave1",
        "rf_input": 1,
        "rf_output": 1
    },   
}

RR_CONSTANTS = {
    "q1_rr": {
        "amplitude": 0.005,
        "readout_length": 8608,
        "midcircuit_amplitude": 0.2511, 
        "mc_readout_length": 3000,
        "IF": 92174364.0,    
        "rotation_angle": ((254.9) / 180) * np.pi,
        "ge_threshold": 1.1816289,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
    "q2_rr": {
        "amplitude": 0.002, 
        "readout_length": 3000,
        "midcircuit_amplitude": 0.1585, 
        "mc_readout_length": 3000,
        "IF": 116.18 * u.MHz,    
        "rotation_angle": (0.0 / 180) * np.pi,
        "ge_threshold": 0.0,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
    "q3_rr": {
        "amplitude": 0.005,
        "readout_length": 5120,
        "midcircuit_amplitude": 0.2238, 
        "mc_readout_length": 3000,
        "IF": 156.8392 * u.MHz,     
        "rotation_angle": (247.2 / 180) * np.pi,
        "ge_threshold": 1.1815551,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
    "q4_rr": {
        "amplitude": 0.002, 
        "readout_length": 3000,
        "midcircuit_amplitude": 0.0355, 
        "mc_readout_length": 3000,
        "IF": 177.2 * u.MHz,    
        "rotation_angle": (0.0 / 180) * np.pi,
        "ge_threshold": 0.0,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
    "q5_rr": {
        "amplitude": 0.0055, 
        "readout_length": 3000,
        "midcircuit_amplitude": 0.3548, 
        "mc_readout_length": 3000,
        "IF": 211.052 * u.MHz,    
        "rotation_angle": ((285.4 + 241-3.5) / 180) * np.pi,
        "ge_threshold": 1.182e-0 -4.749e-04,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
    "q6_rr": {
        "amplitude": 0.006, 
        "readout_length": 3000,
        "midcircuit_amplitude": 0.3548, 
        "mc_readout_length": 3000,
        "IF": 231 * u.MHz,    
        "rotation_angle": ((285.4 + 241-3.5) / 180) * np.pi,
        "ge_threshold": 1.182e-0 -4.749e-04,
        "midcircuit_rotation_angle": (0.0 / 180) * np.pi,
        "midcircuit_ge_threshold": 0.0,
        "use_opt_readout": False,
    },
}

qubit_resonator_correspondence = {qu: res for qu, res in zip(QUBIT_CONSTANTS.keys(), RR_CONSTANTS.keys())}

if use_calibrated_values:
    for qubit_key, constants in zip(CALIBRATION_QUBITS, RR_CONSTANTS.values()):
        for param in UPDATEABLE_PARAMETER_NAMES:
            if SEARCH_PARAMETER_KEY_CORRESPONDENCE[param] == 'IF':
                # Due to overlap in names for qubit and resonator, we need to be specific
                RR_CONSTANTS[qubit_resonator_correspondence[qubit_key]][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                    calibration_dataframe.loc[
                        (calibration_dataframe.qubit_name == qubit_key) &
                        (calibration_dataframe.calibration_parameter_name == 'readout_frequency')
                    ]['calibration_value'].values[0]
            elif SEARCH_PARAMETER_KEY_CORRESPONDENCE[param] in constants.keys():
                try:
                    RR_CONSTANTS[qubit_resonator_correspondence[qubit_key]][SEARCH_PARAMETER_KEY_CORRESPONDENCE[param]] = \
                        calibration_dataframe.loc[
                            (calibration_dataframe.qubit_name == qubit_key) &
                            (calibration_dataframe.calibration_parameter_name == param)
                        ]['calibration_value'].values[0]
                except Exception as e:
                    print(e)
    print('RR_CONSTANTS pulled from calibration_data_dict.json')

weights_dict = {}
for i_q, key in enumerate(RR_CONSTANTS.keys()):
    weights = np.load(f"optimal_weights_qubit{i_q+1}.npz")
    weights_dict[key] = {
        'real': [(x, weights["division_length"] * 4) for x in weights["weights_real"]],
        'imag': [(x, weights["division_length"] * 4) for x in weights["weights_imag"]],
        'minus_real': [(x, weights["division_length"] * 4) for x in weights["weights_minus_real"]],
        'minus_imag': [(x, weights["division_length"] * 4) for x in weights["weights_minus_imag"]],
    }

# MARK: CONFIGURATION
#############################################
#                  Config                   #
#############################################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # I resonator
                2: {"offset": 0.0},  # Q resonator
                3: {"offset": 0.0},  # I qubit
                4: {"offset": 0.0},  # Q qubit
            },
            "digital_outputs": {
                1: {},
                3: {},
                5: {},
                7: {},
                9: {},
            },
            "analog_inputs": {
                1: {"offset": 0.08779529467773438, "gain_db": 20},  # I from down-conversion
                2: {"offset": 0.10877938305664063, "gain_db": 20},  # Q from down-conversion
            },
        },
    },
    "elements": {

        **{qubit_key: {
            "RF_inputs": {"port": (MULTIPLEX_DRIVE_CONSTANTS['drive1']["octave"], MULTIPLEX_DRIVE_CONSTANTS['drive1']["RF_port"])},
            "intermediate_frequency": QUBIT_CONSTANTS[qubit_key]["IF"],  # in Hz
            "operations": {
                "cw": "const_pulse",
                "zero": "zero_pulse",
                "saturation": "saturation_pulse",
                "x180": f"x180_pulse_{qubit_key}",
                "x90": f"x90_pulse_{qubit_key}",
                "-x90": f"-x90_pulse_{qubit_key}",
                "y90": f"y90_pulse_{qubit_key}",
                "y180": f"y180_pulse_{qubit_key}",
                "-y90": f"-y90_pulse_{qubit_key}",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 3),
                    "delay": 57,
                    "buffer": 18,
                },
            },
        } for qubit_key in MULTIPLEX_DRIVE_CONSTANTS["drive1"]["QUBITS"]},

        # readout line 1
        **{rr: {
            "RF_inputs": {"port": (RL_CONSTANTS["rl1"]["rl_octave"], RL_CONSTANTS["rl1"]["rf_input"])},
            "RF_outputs": {"port": (RL_CONSTANTS["rl1"]["rl_octave"], RL_CONSTANTS["rl1"]["rf_output"])},
            "intermediate_frequency": RR_CONSTANTS[rr]["IF"], 
			'time_of_flight': RL_CONSTANTS["rl1"]["TOF"],
            'smearing': 0,
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_"+rr,
                "midcircuit_readout": "midcircuit_readout_pulse_"+rr,
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 1),
                    "delay": 57,
                    "buffer": 18,
                },
            },
        } for rr in RL_CONSTANTS["rl1"]["RESONATORS"]},

    },
    "octaves": {
        "octave1": {
            "RF_outputs": {
                RL_CONSTANTS["rl1"]["rf_input"]: {
                    "LO_frequency": RL_CONSTANTS["rl1"]["LO"],
                    "LO_source": "internal",
                    "output_mode": "triggered",
                    "gain": resonator_octave_gain,
                },
                MULTIPLEX_DRIVE_CONSTANTS['drive1']["RF_port"]: {
                    "LO_frequency": MULTIPLEX_DRIVE_CONSTANTS['drive1']["LO"],
                    "LO_source": "internal",
                    "output_mode": "triggered",
                    "gain": qubit_octave_gain,
                },
            },
            "RF_inputs": {
                RL_CONSTANTS["rl1"]["rf_output"]: {
                    "LO_frequency": RL_CONSTANTS["rl1"]["LO"],
                    "LO_source": "internal",
                },
            },
            "connectivity": "con1",
        }
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "zero_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "zero_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
        **{f"x90_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"x90_I_wf_{key}",
                    "Q": f"x90_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{f"x180_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"x180_I_wf_{key}",
                    "Q": f"x180_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{f"-x90_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"minus_x90_I_wf_{key}",
                    "Q": f"minus_x90_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{f"y90_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"y90_I_wf_{key}",
                    "Q": f"y90_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{f"y180_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"y180_I_wf_{key}",
                    "Q": f"y180_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{f"-y90_pulse_{key}":
            {
                "operation": "control",
                "length": QUBIT_CONSTANTS[key]["pi_len"],
                "waveforms": {
                    "I": f"minus_y90_I_wf_{key}",
                    "Q": f"minus_y90_Q_wf_{key}"
                },
                "digital_marker": "ON",
            }
            for key in QUBIT_CONSTANTS.keys()
        },
        **{
            f"readout_pulse_{key}": {
                "operation": "measurement",
                "length": RR_CONSTANTS[key]["readout_length"],
                "waveforms": {
                    "I": f"readout_wf_{key}",
                    "Q": "zero_wf"
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                    "rotated_cos": f"rotated_cosine_weights_{key}",
                    "rotated_sin": f"rotated_sine_weights_{key}",
                    "rotated_minus_sin": f"rotated_minus_sine_weights_{key}",
                    "opt_cos": f"opt_cosine_weights_{key}",
                    "opt_sin": f"opt_sine_weights_{key}",
                    "opt_minus_sin": f"opt_minus_sine_weights_{key}",
                },
                "digital_marker": "ON",
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"midcircuit_readout_pulse_{key}": {
                "operation": "measurement",
                "length": RR_CONSTANTS[key]["mc_readout_length"],
                "waveforms": {
                    "I": f"midcircuit_readout_wf_{key}",
                    "Q": "zero_wf"
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                    "rotated_cos": f"midcircuit_rotated_cosine_weights_{key}",
                    "rotated_sin": f"midcircuit_rotated_sine_weights_{key}",
                    "rotated_minus_sin": f"midcircuit_rotated_minus_sine_weights_{key}",
                    "opt_cos": f"opt_cosine_weights_{key}",
                    "opt_sin": f"opt_sine_weights_{key}",
                    "opt_minus_sin": f"opt_minus_sine_weights_{key}",
                },
                "digital_marker": "ON",
            } for key in RR_CONSTANTS.keys()
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "saturation_drive_wf": {"type": "constant", "sample": saturation_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        **{f"x90_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_x90_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"x90_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_x90_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"x180_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_x180_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"x180_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_x180_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"minus_x90_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_minus_x90_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"minus_x90_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_minus_x90_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"y90_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_y90_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"y90_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_y90_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"y180_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_y180_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"y180_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_y180_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"minus_y90_I_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_minus_y90_I"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"minus_y90_Q_wf_{key}": {"type": "arbitrary", "samples": waveforms[key+"_minus_y90_Q"].tolist()} for key in QUBIT_CONSTANTS.keys()},
        **{f"readout_wf_{key}": {"type": "constant", "sample": RR_CONSTANTS[key]["amplitude"]} for key in RR_CONSTANTS.keys()},
        **{f"midcircuit_readout_wf_{key}": {"type": "constant", "sample": RR_CONSTANTS[key]["midcircuit_amplitude"]} for key in RR_CONSTANTS.keys()},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
        **{
            f"rotated_cosine_weights_{key}": {
                "cosine": [(np.cos(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]],
                "sine": [(np.sin(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]]
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"rotated_sine_weights_{key}": {
                "cosine": [(-np.sin(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]],
                "sine": [(np.cos(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]]
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"rotated_minus_sine_weights_{key}": {
                "cosine": [(np.sin(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]],
                "sine": [(-np.cos(RR_CONSTANTS[key]["rotation_angle"])), RR_CONSTANTS[key]["readout_length"]]
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"opt_cosine_weights_{key}": {
                "cosine": weights_dict[key]['real'],
                "sine": weights_dict[key]['minus_imag'],
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"opt_sine_weights_{key}": {
                "cosine": weights_dict[key]['imag'],
                "sine": weights_dict[key]['real'],
            } for key in RR_CONSTANTS.keys()
        },
        **{
            f"opt_minus_sine_weights_{key}": {
                "cosine": weights_dict[key]['minus_imag'],
                "sine": weights_dict[key]['minus_real'],
            } for key in RR_CONSTANTS.keys()
        },
    },
}