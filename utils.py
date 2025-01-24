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
