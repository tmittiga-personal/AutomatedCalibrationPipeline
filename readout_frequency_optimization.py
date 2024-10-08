"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from multiplexed_configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
from scipy import optimize


def peak_fit(
    x,
    y,
    fitname='gaussian',
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

    if fitname == 'gaussian':
        fitfunc = lambda x, *p, :  p[0]*np.exp(-( ( (x-p[1])/p[2] )**2 )/2) + p[3]
    else:
        fitfunc = lambda x, *p, :  p[0]/( (1+ ((x-p[1])/p[2])**2)  ) + p[3]

    ymax, ymin = y.max(), y.min()
    s = np.flatnonzero(y > (ymax + ymin)/2) # Find data points above middle y value. First and last points used to estimate width
    # Seed paramters: amplitude, center, width, background
    p0 = [(ymax-ymin), x[np.argmax(y)], np.abs(x[s[0]]- x[s[-1]]), ymin]


    # Fit
    fit_params, covar = optimize.curve_fit(fitfunc, x,y, p0=p0) #, sigma=wd9e, bounds=([109,-1, 0, 0.5 ],[110,0, 1, 1.5 ]))
    std_vec = np.sqrt(np.diag(covar))

    fig_peak = plt.figure()
    plt.errorbar(x,y, color=(0, 0, 1), ls='none', #yerr=wd9e,
        fmt='.', elinewidth=1, capsize=None, label='Data')
    xvals = np.linspace(x.min(), x.max(), 500)
    # fit_y = fitfunc(xvals,*p0)  # plot seed fit func for debugging
    fit_y = fitfunc(xvals,*fit_params)
    plt.plot(xvals, fit_y , "r-", label='Fit', linewidth=5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("SNR")
    plt.legend()
    plt.title(f'Location Fit: {round(fit_params[1],5)} +- {round(std_vec[1],5)}')
    return fig_peak, fit_params, std_vec


def readout_frequency_optimization(
    qubit,
    resonator,
):
    
    data_handler = DataHandler(root_data_folder="./")

    ###################
    # The QUA program #
    ###################
    n_avg = 4000  # The number of averages
    # The frequency sweep parameters
    span = 1 * u.MHz
    df = 60 * u.kHz
    dfs = np.arange(-span, +span + 0.1, df)

    ro_freq_opt_data = {
        "n_runs": n_avg,
        "dfs": dfs,
        "Twpa_status": twpa_status,
        "resonator_LO": RL_CONSTANTS["rl1"]["LO"],
        "readout_amp": RR_CONSTANTS[resonator]["amplitude"],
        "resonator_IF": RR_CONSTANTS[resonator]['IF'],
        "qubit_LO": MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
        "qubit_IF": QUBIT_CONSTANTS[qubit]["IF"],
        "qubit_octave_gain": qubit_octave_gain,
        "resonator_octave_gain": resonator_octave_gain,
    }

    with program() as ro_freq_opt:
        n = declare(int)  # QUA variable for the averaging loop
        df = declare(int)  # QUA variable for the readout frequency
        I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
        Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
        Ig_st = declare_stream()
        Qg_st = declare_stream()
        I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
        Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
        Ie_st = declare_stream()
        Qe_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(df, dfs)):
                # Update the frequency of the digital oscillator linked to the resonator element
                update_frequency(resonator, df + RR_CONSTANTS[resonator]['IF'])
                # Measure the state of the resonator
                if RR_CONSTANTS[resonator]["use_opt_readout"]:
                    measure(
                        "readout",
                        resonator,
                        None,
                        dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I_g),
                        dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q_g),
                    )
                else:
                    measure(
                        "readout",
                        resonator,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
                    )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, resonator)
                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                save(I_g, Ig_st)
                save(Q_g, Qg_st)

                align()  # global align
                # Play the x180 gate to put the qubit in the excited state
                play("x180", qubit)
                # Align the two elements to measure after playing the qubit pulse.
                align(qubit, resonator)
                # Measure the state of the resonator
                if RR_CONSTANTS[resonator]["use_opt_readout"]:
                    measure(
                        "readout",
                        resonator,
                        None,
                        dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I_e),
                        dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q_e),
                    )
                else:
                    measure(
                        "readout",
                        resonator,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
                    )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, resonator)
                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                save(I_e, Ie_st)
                save(Q_e, Qe_st)
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            n_st.save("iteration")
            # mean values
            Ig_st.buffer(len(dfs)).average().save("Ig_avg")
            Qg_st.buffer(len(dfs)).average().save("Qg_avg")
            Ie_st.buffer(len(dfs)).average().save("Ie_avg")
            Qe_st.buffer(len(dfs)).average().save("Qe_avg")
            # variances to get the SNR
            (
                ((Ig_st.buffer(len(dfs)) * Ig_st.buffer(len(dfs))).average())
                - (Ig_st.buffer(len(dfs)).average() * Ig_st.buffer(len(dfs)).average())
            ).save("Ig_var")
            (
                ((Qg_st.buffer(len(dfs)) * Qg_st.buffer(len(dfs))).average())
                - (Qg_st.buffer(len(dfs)).average() * Qg_st.buffer(len(dfs)).average())
            ).save("Qg_var")
            (
                ((Ie_st.buffer(len(dfs)) * Ie_st.buffer(len(dfs))).average())
                - (Ie_st.buffer(len(dfs)).average() * Ie_st.buffer(len(dfs)).average())
            ).save("Ie_var")
            (
                ((Qe_st.buffer(len(dfs)) * Qe_st.buffer(len(dfs))).average())
                - (Qe_st.buffer(len(dfs)).average() * Qe_st.buffer(len(dfs)).average())
            ).save("Qe_var")

    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    ###########################
    # Run or Simulate Program #
    ###########################
    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, ro_freq_opt, simulation_config)
        job.get_simulated_samples().con1.plot()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(ro_freq_opt)  # execute QUA program
        # Get results from QUA program
        results = fetching_tool(
            job,
            data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
            mode="live",
        )
        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Derive the SNR
            Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
            var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
            SNR = ((np.abs(Z)) ** 2) / (2 * var)
            # Plot results
            plt.cla()
            plt.plot(dfs / u.MHz, SNR, ".-")
            plt.title(f"Readout frequency optimization around {RR_CONSTANTS[resonator]['IF'] / u.MHz} MHz")
            plt.xlabel("Readout frequency detuning [MHz]")
            plt.ylabel("SNR")
            plt.grid("on")
            plt.pause(0.1)
        
        ro_freq_opt_data["figure"] = fig
        plt.close()
        
        fig_peak, fit_params, std_vec = peak_fit(
            x = dfs,
            y = SNR,
        )

        fit_dict = {
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
        ro_freq_opt_data["Ig_avg"] = Ig_avg
        ro_freq_opt_data["Qg_avg"] = Qg_avg
        ro_freq_opt_data["Ie_avg"] = Ie_avg
        ro_freq_opt_data["Qe_avg"] = Qe_avg

        ro_freq_opt_data["Ig_var"] = Ig_var
        ro_freq_opt_data["Qg_var"] = Qg_var
        ro_freq_opt_data["Ie_var"] = Ie_var
        ro_freq_opt_data["Qe_var"] = Qe_var

        ro_freq_opt_data["fit_figure"] = fig_peak

        data_folder = data_handler.save_data(ro_freq_opt_data, name=f"{resonator}_ro_freq_opt")

        plt.close()
        return fit_dict, data_folder