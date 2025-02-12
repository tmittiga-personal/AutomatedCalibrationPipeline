import pandas as pd
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qualang_tools.analysis.discriminator import _false_detections

# The time it takes for Quantum Machines OPX to calculate a dynamically updated variable is ~100ns
# If it is calculating a wait time, it is smart enough to subtract the calculation delay from the total
# wait time. MEanwhile, precompiled zero_pulses don't cause a delay... 
# So we want to switch from precompiled zero pulses to the wait command after this threshold time.
QM_DYNAMICAL_UPDATE_TIME_THRESHOLD = 1000//4

DATAFRAME_FILE = "./calibration_database_2025.pkl"
BACKUP_DATAFRAME_FILE = "./calibration_database_2025_BACKUP.pkl"

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
    all_attempts: bool = False,
):
    assert isinstance(qubits, list), 'qubits must be a list of qubit names'

    df = pd.read_pickle(DATAFRAME_FILE)

    if all_attempts:
        df_found = df.loc[
            (df['calibration_parameter_name'].isin(search_parameter_names)) &
            (df['qubit_name'].isin(qubits))
        ].drop_duplicates(subset = ['qubit_name', 'calibration_parameter_name'], keep='last')
    else:
        df_found = df.loc[
            (df['calibration_parameter_name'].isin(search_parameter_names)) &
            (df['qubit_name'].isin(qubits)) &
            (df.calibration_success == True)
        ].drop_duplicates(subset = ['qubit_name', 'calibration_parameter_name'], keep='last')
    
    return df_found


def pull_latest_n_calibrated_values(
    qubits: List[str],
    search_parameter_names: List[str],
    n_latest: int = 1,
    all_attempts: bool = False,
):
    assert isinstance(qubits, list), 'qubits must be a list of qubit names'

    df = pd.read_pickle(DATAFRAME_FILE)

    if all_attempts:
        df_found = df.loc[
            (df['calibration_parameter_name'].isin(search_parameter_names)) &
            (df['qubit_name'].isin(qubits))
        ].iloc[-1*n_latest:]
    else:
        df_found = df.loc[
            (df['calibration_parameter_name'].isin(search_parameter_names)) &
            (df['qubit_name'].isin(qubits)) &
            (df.calibration_success == True)
        ].iloc[-1*n_latest:]
    
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


def two_state_discriminator_plot(Ig, Qg, Ie, Qe, b_print=True):
    """
    Given two blobs in the IQ plane representing two states, finds the optimal threshold to discriminate between them
    and calculates the fidelity. Also returns the angle in which the data needs to be rotated in order to have all the
    information in the `I` (`X`) axis.

    .. note::
        This function assumes that there are only two blobs in the IQ plane representing two states (ground and excited)
        Unexpected output will be returned in other cases.


    :param float Ig: A vector containing the `I` quadrature of data points in the ground state
    :param float Qg: A vector containing the `Q` quadrature of data points in the ground state
    :param float Ie: A vector containing the `I` quadrature of data points in the excited state
    :param float Qe: A vector containing the `Q` quadrature of data points in the excited state
    :param bool b_print: When true (default), prints the results to the console.
    :param bool b_plot: When true (default), plots the results in a new figure.
    :returns: A tuple of (angle, threshold, fidelity, gg, ge, eg, ee).
        angle - The angle (in radians) in which the IQ plane has to be rotated in order to have all the information in
            the `I` axis.
        threshold - The threshold in the rotated `I` axis. The excited state will be when the `I` is larger (>) than
            the threshold.
        fidelity - The fidelity for discriminating the states.
        gg - The matrix element indicating a state prepared in the ground state and measured in the ground state.
        ge - The matrix element indicating a state prepared in the ground state and measured in the excited state.
        eg - The matrix element indicating a state prepared in the excited state and measured in the ground state.
        ee - The matrix element indicating a state prepared in the excited state and measured in the excited state.
    """
    # Condition to have the Q equal for both states:
    angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
    C = np.cos(angle)
    S = np.sin(angle)
    # Condition for having e > Ig
    if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
        angle += np.pi
        C = np.cos(angle)
        S = np.sin(angle)

    Ig_rotated = Ig * C - Qg * S
    Qg_rotated = Ig * S + Qg * C

    Ie_rotated = Ie * C - Qe * S
    Qe_rotated = Ie * S + Qe * C

    fit = minimize(
        _false_detections,
        0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),
        (Ig_rotated, Ie_rotated),
        method="Nelder-Mead",
    )
    threshold = fit.x[0]

    gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
    ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
    eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
    ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)

    fidelity = 100 * (gg + ee) / 2

    if b_print:
        print(
            f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f} | {ge:.3f} |
        ----------------
        | {eg:.3f} | {ee:.3f} |
        -----------------
        IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
        Threshold: {threshold:.3e}
        Fidelity: {fidelity:.1f}%
        """
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=2)
    ax1.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=2)
    ax1.axis("equal")
    ax1.legend(["Ground", "Excited"])
    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.set_title("Original Data")

    ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label="Ground", markersize=2)
    ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label="Excited", markersize=2)
    ax2.axis("equal")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.set_title("Rotated Data")

    ax3.hist(Ig_rotated, bins=50, alpha=0.75, label="Ground")
    ax3.hist(Ie_rotated, bins=50, alpha=0.75, label="Excited")
    ax3.axvline(x=threshold, color="k", ls="--", alpha=0.5)
    text_props = dict(
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax3.transAxes,
    )
    ax3.text(0.7, 0.9, f"{threshold:.3e}", text_props)
    ax3.set_xlabel("I")
    ax3.set_title("1D Histogram")

    ax4.imshow(np.array([[gg, ge], [eg, ee]]))
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(labels=["|g>", "|e>"])
    ax4.set_yticklabels(labels=["|g>", "|e>"])
    ax4.set_ylabel("Prepared")
    ax4.set_xlabel("Measured")
    ax4.text(0, 0, f"{100 * gg:.1f}%", ha="center", va="center", color="k")
    ax4.text(1, 0, f"{100 * ge:.1f}%", ha="center", va="center", color="w")
    ax4.text(0, 1, f"{100 * eg:.1f}%", ha="center", va="center", color="w")
    ax4.text(1, 1, f"{100 * ee:.1f}%", ha="center", va="center", color="k")
    ax4.set_title("Fidelities")
    fig.tight_layout()

    return angle, threshold, fidelity, gg, ge, eg, ee, fig


def flattop_gaussian_risefall_waveforms(
    amplitude, risefalllength, flatlength, sigma, subtracted=True, sampling_rate=1e9, **kwargs
):
    """

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float sigma: The gaussian standard deviation.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
    :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
    :param bool subtracted: If true, returns a subtracted Gaussian, such that the first and last points will be at 0
        volts. This reduces high-frequency components due to the initial and final points offset. Default is true.
    :param float sampling_rate: The sampling rate used to describe the waveform, in samples/s. Default is 1G samples/s.
    :return: Returns a tuple of two lists. The first list is the 'I' waveform (real part) and the second is the
        'Q' waveform (imaginary part)
    """
    
    t = np.arange(2*risefalllength, step=1e9 / sampling_rate)  # An array of size pulse length in ns
    center = (2*risefalllength - 1e9 / sampling_rate) / 2
    gauss_wave = amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2))  # The gaussian function
    gauss_wave = gauss_wave - gauss_wave[-1]  # subtracted gaussian
    z = gauss_wave + 1j * 0
    
    I_wf = z.real.tolist()  # The `I` component is the real part of the waveform
    max_I = np.max(I_wf)  # extract any aliased maximum

    # Insert flat period
    rise_I = I_wf[:int(np.floor(len(I_wf)/2))]
    fall_I = I_wf[int(np.floor(len(I_wf)/2)):]
    t = np.arange(flatlength, step=1e9 / sampling_rate)
    flat_I = np.zeros(len(t)) + max_I

    I_wf = np.concatenate((rise_I, flat_I, fall_I))

    # The `Q` component is the imaginary part of the waveform, which by definition is 0 here
    Q_wf = np.zeros(len(I_wf)) 
    return I_wf, Q_wf