"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Having found the pi pulse amplitude (power_rabi).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the configuration.

Next steps before going to the next node:
    - Update the qubit pulse amplitude (x180_amp) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
from scipy import optimize

MINIMUM_SLOPE_SCALE = 0.1  # Factor of original slope in peak_turnover function that determines cut-off condition

data_handler = DataHandler(root_data_folder="./")

def power_pi_pulse(
    simulate = False
):
    ###################
    # The QUA program #
    ###################
    n_avg = 25  # The number of averages
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    a_min = 0.5
    a_max = 1.5
    n_a = 100
    amplitudes = np.linspace(a_min, a_max, n_a)
    # Number of applied Rabi pulses sweep
    max_nb_of_pulses = 30  # Maximum number of qubit pulses
    nb_of_pulses = np.arange(0, max_nb_of_pulses, 2)  # Always play an odd/even number of pulses to end up in the same state

    power_rabi_error_amplification_data = {
        "n_avg": n_avg,
        "resonator_LO": resonator_LO,
        "readout_amp": readout_amp,
        "qubit_LO": qubit_LO,
        "qubit_IF":qubit_IF,
        "x180_amp": x180_amp,
        "x180_Len": x180_len,
        "amplitudes": amplitudes,
        "qubit_octave_gain": qubit_octave_gain,
        "resonator_octave_gain": resonator_octave_gain,
    }

    with program() as power_rabi_err:
        n = declare(int)  # QUA variable for the averaging loop
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        n_rabi = declare(int)  # QUA variable for the number of qubit pulses
        n2 = declare(int)  # QUA variable for counting the qubit pulses
        I = declare(fixed)  # QUA variable for the measured 'I' quadrature
        Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
        I_st = declare_stream()  # Stream for the 'I' quadrature
        Q_st = declare_stream()  # Stream for the 'Q' quadrature
        n_st = declare_stream()  # Stream for the averaging iteration 'n'

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            with for_(*from_array(n_rabi, nb_of_pulses)):  # QUA for_ loop for sweeping the number of pulses
                with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude
                    # Loop for error amplification (perform many qubit pulses with varying amplitudes)
                    with for_(n2, 0, n2 < n_rabi, n2 + 1):
                        play("x180" * amp(a), "qubit")
                    # Align the two elements to measure after playing the qubit pulses.
                    align("qubit", "resonator")
                    # Measure the state of the resonator
                    # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                    )
                    # Wait for the qubit to decay to the ground state
                    wait(thermalization_time * u.ns, "resonator")
                    # Save the 'I' & 'Q' quadratures to their respective streams
                    save(I, I_st)
                    save(Q, Q_st)
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
            I_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).average().save("I")
            Q_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).average().save("Q")
            n_st.save("iteration")

    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    ###########################
    # Run or Simulate Program #
    ###########################

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, power_rabi_err, simulation_config)
        job.get_simulated_samples().con1.plot()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(power_rabi_err)
        # Get results from QUA program
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            I, Q, iteration = results.fetch_all()
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Plot results
            plt.suptitle("Power Rabi with error amplification")
            plt.subplot(221)
            plt.cla()
            plt.pcolor(amplitudes * x180_amp, nb_of_pulses, I)
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.ylabel("# of Rabi pulses")
            plt.title("I quadrature [V]")
            plt.subplot(222)
            plt.cla()
            plt.pcolor(amplitudes * x180_amp, nb_of_pulses, Q)
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.title("Q quadrature [V]")
            plt.subplot(212)
            plt.cla()
            plt.plot(amplitudes * x180_amp, np.sum(Q, axis=0))
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.ylabel("Sum along the # of Rabi pulses")
            plt.pause(1)
            plt.tight_layout()
        power_rabi_error_amplification_data["I"] = I
        power_rabi_error_amplification_data["Q"] = Q
        power_rabi_error_amplification_data["figure"] = fig
        plt.close()

        fit_dict, fig_fit = power_analysis(amplitudes, I, Q)
        power_rabi_error_amplification_data["fit_dict"] = fit_dict
        power_rabi_error_amplification_data["fit_figure"] = fig_fit

        data_handler.save_data(power_rabi_error_amplification_data, name="power_pi_error_amplification")

        return fit_dict


def power_analysis(amplitudes, I, Q):
    """
    Analyze the data from a power rabi sweep measurement.
    Assumes the data swept both the pulse amplitude and the number of pulses.
    Assumes the IQ Blob has been calibrated, so one of I or Q is high contrast data.

    Averages the I or Q data along the power-sweep axis to form 1D data. Finds the datapoint closest to the
    peak location, then creates a crop window around the peak before fitting.

    :param amplitudes: the swept amplitude array
    :param I: I data
    :param Q: Q data

    :return: dictionary of fit parameters and uncertainties, figure of the fit
    """
    # Pick I or Q based on contrast
    I_lineout = np.mean(I, axis=0)
    Q_lineout = np.mean(Q, axis=0)

    if np.abs(I_lineout.max() - I_lineout.min()) > np.abs(Q_lineout.max() - Q_lineout.min()):
        lineout = I_lineout
        is_negative_amplitude = True
    else:
        lineout = Q_lineout
        is_negative_amplitude = False

    fig_fit, fit_params, std_vec = peak_fit(
        x = amplitudes,
        y = lineout,
        x_label = 'Pulse Amplitude [V]',
        y_label = 'Population',
        is_negative_amplitude=is_negative_amplitude,
    )

    # Turn the fit params and uncertainties into a dict
    fit_dict = {
        'fit_data_name': 'I' if is_negative_amplitude else 'Q',
        'amplitudes': amplitudes,
        'fit_data': lineout,
        'fit_values': {
            'peak_amplitude': fit_params[0],
            'center': fit_params[1],
            'width': fit_params[2],
            'background': fit_params[3],
        },
        'fit_uncertainties':{
            'peak_amplitude': std_vec[0],
            'center': std_vec[1],
            'width': std_vec[2],
            'background': std_vec[3],
        },
    }


    return fit_dict, fig_fit


def peak_fit(
    x,
    y,
    x_label = '',
    y_label = '',
    is_negative_amplitude=False, 
    fitname='gaussian',
    amp=x180_amp,
):
    """
    Fits a gaussian or lorentzian to 1D data cropped to a window.

    :param x: the x-axis array
    :param y: the y-axis data
    :param x_window_start: minimum x-axis value of the cropped data
    :param x_window_end: max x-axis value of the cropped data
    :param x_label: str for labeling the x axis
    :param y_label: str for labeling the y axis
    :param is_negative_amplitude: set True to indicate the peak has a negative amplitude
    :param fitname: choose the peak type from guassian or lorentzian

    :return: figure of the fit, fit_params array, and fit parameter uncertainty array
    """
    assert fitname in ['gaussian', 'lorentzian'], f"{fitname=} not in  ['gaussian', 'lorentzian']"

    # Grab data just within the window
    window_indices = power_peak_turnover(
        x,
        y,
        is_negative_amplitude=is_negative_amplitude,
        roll_number = 3,
    )
    y = np.array(y[window_indices[0]:window_indices[1]])
    x = np.array(x[window_indices[0]:window_indices[1]])

    if fitname == 'gaussian':
        fitfunc = lambda x, *p, :  p[0]*np.exp(-( ( (x-p[1])/p[2] )**2 )/2) + p[3]
    else:
        fitfunc = lambda x, *p, :  p[0]/( (1+ ((x-p[1])/p[2])**2)  ) + p[3]

    ymax, ymin = y.max(), y.min()
    if is_negative_amplitude:
        s = np.flatnonzero(y < (ymax + ymin)/2) # Find data points above middle y value. First and last points used to estimate width
        # Seed paramters: amplitude, center, width, background
        p0 = [-1*(ymax-ymin), x[np.argmin(y)], np.abs(x[s[0]]- x[s[-1]]), ymax]
    else:
        s = np.flatnonzero(y > (ymax + ymin)/2) # Find data points above middle y value. First and last points used to estimate width
        # Seed paramters: amplitude, center, width, background
        p0 = [(ymax-ymin), x[np.argmax(y)], np.abs(x[s[0]]- x[s[-1]]), ymin]


    # Fit
    fit_params, covar = optimize.curve_fit(fitfunc, x,y, p0=p0) #, sigma=wd9e, bounds=([109,-1, 0, 0.5 ],[110,0, 1, 1.5 ]))
    std_vec = np.sqrt(np.diag(covar))

    fig_peak = plt.figure()
    plt.errorbar(x*amp,y, color=(0, 0, 1), ls='none', #yerr=wd9e,
        fmt='.', elinewidth=1, capsize=None, label='Data')
    xvals = np.linspace(x.min(), x.max(), 500)
    # fit_y = fitfunc(xvals,*p0)  # plot seed fit func for debugging
    fit_y = fitfunc(xvals,*fit_params)
    plt.plot(xvals*amp, fit_y , "r-", label='Fit', linewidth=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(f'Location Fit: {round(fit_params[1]*amp,5)} +- {round(std_vec[1]*amp,5)}')
    return fig_peak, fit_params, std_vec


def power_peak_turnover(    
    x,
    y,
    is_negative_amplitude=False,
    roll_number = 3,
):
    """
    Given data associated with a single peak, find array indices that can crop the data to a reasonable
    region of interest (ROI) centered on the peak.
    This function finds the indices by:
        1) Find the datapoint closest to the peak
        2) Take a number of datapoints (roll_number) stepping away from the peak in the negative direction
        3) Fit the ("rolling average") slope of those datapoints.
        4) Continue stepping away by 1 index, until the rolling average slope either:
            a) changes polarity
            b) becomes significantly smaller in modulus than the original slope
        5) note the index when one of the above conditions is met. This is the lower index of the ROI
        6) Repeat All above steps for the positive direction for the upper index of the ROI.
    
    :param x: the sweep array
    :param y: the data
    :param is_negative_amplitude: True when the peak has a negative amplitude

    :return: an array with two elements, [lower index, upper index]
    """
    i_peak = np.argmin(y) if is_negative_amplitude else np.argmax(y)
        
    fitfunc = lambda x, *p, :  p[0]*x

    low_high_indices = [0,0]
    for i_low_high in range(2):
        found = False
        i_step = 0

        while not found:

            if i_low_high == 0:
                first_index = i_peak-(i_step+roll_number)
                second_index = i_peak-i_step
            else:
                first_index = i_peak+i_step
                second_index = i_peak+i_step+roll_number

            x_crop = x[first_index:second_index]
            y_crop = y[first_index:second_index]

            p0 = [(y_crop[-1]-y_crop[0])/(x_crop[-1]-x_crop[0])]
            fit_params, _ = optimize.curve_fit(fitfunc, x_crop-x_crop[0], y_crop-y_crop[0], p0=p0)
            new_slope = fit_params[0]

            if i_step == 0:
                i_step += 1
                original_slope = new_slope
                old_slope_polarity_positive = new_slope > 0
                continue
            
            # If the rolling average slope polarity changes, the peak has ended
            # If the polarities are the same, but the new slope is 10x flatter, the peak has ended
            if (
                old_slope_polarity_positive != (new_slope > 0)
                or
                (old_slope_polarity_positive == (new_slope > 0)) and (np.abs(new_slope) < np.abs(MINIMUM_SLOPE_SCALE*original_slope))
            ):
                found = True
                neg_pos = -1 if i_low_high == 0 else 1
                low_high_indices[i_low_high] = int(np.ceil(neg_pos*(i_step+roll_number/2) + i_peak))

            i_step += 1
        
    return low_high_indices
        


if __name__ == "__main__":
    power_pi_pulse()