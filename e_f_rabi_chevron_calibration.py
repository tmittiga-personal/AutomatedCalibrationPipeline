"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubit T1 (qubit_T1) in the configuration.
"""

from qm.qua import *
from qm.qua.lib import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from create_multiplexed_configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
import itertools
from copy import deepcopy
from scipy import optimize
from utils import *

def ef_rabi_chevron(
    current_qubit: str,
    a_min: float = 0,
    a_max: float = 2.1,
    n_a: int = 31,
    f_min: float = -25,  # MHz
    f_max: float = -8,
    f_step: float = 0.5,
    n_avg: int = 30,
):
    mc = create_multiplexed_configuration()

    data_handler = DataHandler(root_data_folder="./")

    ###################
    # The QUA program #
    ###################

    assert current_qubit[-2:] == 'ef', f'Current qubit must be ef qubit, not {current_qubit}'
    ge_qubit = current_qubit.replace('ef','xy')
    ef_qubit = current_qubit
    qubit_IF = mc.QUBIT_CONSTANTS[ef_qubit]['IF']
    x180_amp = mc.QUBIT_CONSTANTS[ef_qubit]['pi_amplitude']
    fs = np.arange(f_min, f_max, f_step)*mc.u.MHz
    fs = np.round(fs)

    amplitudes = np.linspace(a_min, a_max, n_a)

    # Automatically select resonators associated with the probed qubits
    qubit_resonator_correspondence = {qu: res for qu, res in zip(mc.QUBIT_CONSTANTS.keys(), mc.RR_CONSTANTS.keys())}
    resonators = [qubit_resonator_correspondence[key] for key in [ge_qubit]]
    resonator = resonators[0]

    amplitude_rabi_chevron_data = {
        "n_avg": n_avg,
        "resonators": resonators,
        "RR_CONSTANTS": mc.RR_CONSTANTS,
        "RL_CONSTANTS": mc.RL_CONSTANTS,
        "QUBIT_CONSTANTS": mc.QUBIT_CONSTANTS,
        "MULTIPLEX_DRIVE_CONSTANTS": mc.MULTIPLEX_DRIVE_CONSTANTS,
        "ge_qubit": ge_qubit,
        "qubit_octave_gain": mc.qubit_octave_gain,
        "frequencies": fs,
        'ef_qubit': ef_qubit,
        'ef_qubit_IF': qubit_IF,
        'ef_amplitude': x180_amp,
        "amplitude_scaling": mc.amplitude_scaling,
    }

    # generate config pulses to match time sweep
    config_copy = deepcopy(mc.config)  # copy the config for safety

    with program() as rabi_amp_freq:
        n = declare(int)  # QUA variable for the averaging loop
        f = declare(int)  # QUA variable for the qubit frequency
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        I_cases = declare(fixed)  # QUA variable for the measured 'I' quadrature in each case
        Q_cases = declare(fixed)  # QUA variable for the measured 'Q' quadrature in each case
        I_st = declare_stream()  # Stream for the 'I' quadrature in each case
        Q_st = declare_stream()  # Stream for the 'I' quadrature in each case
        n_st = declare_stream()  # Stream for the averaging iteration 'n'

        with for_(n, 0, n < n_avg, n + 1):  
            with for_(*from_array(f, fs)):  # QUA for_ loop for sweeping the frequency
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency(ef_qubit, f + round(qubit_IF))
                with for_(*from_array(a, amplitudes)):
                    # assign(final_phase, Cast.mul_fixed_by_int( f_artificial * 1e-9, 4 * t))

                    # Strict_timing ensures that the sequence will be played without gaps
                    with strict_timing_():           
                        align()
                        # Initialize to |e>
                        play("x180", ge_qubit)                    

                        # Attempt to tranfer to |f>
                        wait(10, ge_qubit)

                        align(ge_qubit, ef_qubit)
                        play("x180" * amp(a), ef_qubit)
                        align() #ge_qubit, ef_qubit)
                        
                        # return to |g>
                        # wait(10, ge_qubit)
                        # play("x180", ge_qubit)

                        # align(ge_qubit, resonator)
                        measure(
                            "readout",
                            resonator,
                            None,
                            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                        )
                        # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                        # assign(I_thresholded, Util.cond(I_cases>thresholds[0], 1, 0))
                        # save(I_cases, I_cases_st[i_q])
                        # save(Q_cases, Q_cases_st[i_q])
                        save(I_cases, I_st)
                        save(Q_cases, Q_st)
                        # Wait for the qubit to decay to the ground state
                        wait(mc.thermalization_time * mc.u.ns, resonator)
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

        with stream_processing():
            # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
            # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
            # get_equivalent_log_array() is used to get the exact values used in the QUA program.
            
            I_st.buffer(n_a).buffer(len(fs)).average().save("I")
            Q_st.buffer(n_a).buffer(len(fs)).average().save("Q")
            n_st.save("iteration")

    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=mc.qop_ip, port=mc.qop_port, cluster_name=mc.cluster_name, octave=mc.octave_config)

    ###########################
    # Run or Simulate Program #
    ###########################
    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(mc.config, rabi_amp_freq, simulation_config)
        # get DAC and digital samples
        samples = job.get_simulated_samples()
        # get the waveform report object and plot
        waveform_report = job.get_simulated_waveform_report()
        waveform_report.create_plot(samples, plot=True, save_path="./Simulated_Waveforms/")

    else:
        # Open the quantum machine
        # qm = qmm.open_qm(config)
        qm = qmm.open_qm(config_copy)
        qm.queue.clear()
        job = qm.execute(rabi_amp_freq)
        results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
        print(job.execution_report())
        # Live plotting
        fig1 = plt.figure()
        while results.is_processing():
            # Fetch results
            I, Q, iteration = results.fetch_all()
            # Convert results into Volts
            S = mc.u.demod2volts(I + 1j * Q, mc.RR_CONSTANTS[resonator]["readout_length"])
            R = np.abs(S)  # Amplitude
            phase = np.angle(S)  # Phase
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Plot results
            plt.subplot(211)
            plt.suptitle(f"Rabi chevron with LO={mc.qubit_LO / mc.u.GHz}GHz and IF={qubit_IF / mc.u.MHz}MHz")
            plt.cla()
            plt.title(r"$R=\sqrt{I^2 + Q^2}$")
            plt.pcolor(amplitudes * x180_amp, fs / mc.u.MHz, R)
            plt.ylabel("Frequency detuning [MHz]")
            plt.xlabel("Pulse amplitude [V]")
            plt.subplot(212)
            plt.cla()
            plt.title("Phase")
            plt.pcolor(amplitudes * x180_amp, fs / mc.u.MHz, np.unwrap(phase))
            plt.ylabel("Frequency detuning [MHz]")
            plt.xlabel("Pulse amplitude [V]")
            plt.tight_layout()
            plt.pause(1)

        plt.close()
        amplitude_rabi_chevron_data["I"] = I
        amplitude_rabi_chevron_data["Q"] = Q
        amplitude_rabi_chevron_data["R"] = R
        amplitude_rabi_chevron_data["phase"] = phase
        amplitude_rabi_chevron_data["figure"] = fig1

        fit_dict, pi_half_amp, fig2 = fit_lineouts(
            amplitudes, 
            fs,
            amplitude_rabi_chevron_data,
        )
        fit_dict['scaled_original_amplitude'] = x180_amp/mc.amplitude_scaling,
        fit_dict['new_IF'] = qubit_IF + fit_dict['frequency']['fit_values']['center']
        amplitude_rabi_chevron_data["fit_figure"] = fig2

        data_folder = data_handler.save_data(
            amplitude_rabi_chevron_data, 
            name=f"{current_qubit}_ef_power_freq_chevron"
        )
        qmm.clear_all_job_results()
        qmm.close_all_quantum_machines()
        
        return fit_dict, pi_half_amp, data_folder


def fit_lineouts(
    amplitude_sweep,
    frequency_sweep,
    amplitude_rabi_chevron_data,
):
    R = amplitude_rabi_chevron_data["R"]
    # phase = amplitude_rabi_chevron_data["phase"]
    freq_lineout = np.mean(R, axis=1)
    amp_lineout = np.mean(R, axis=0)

    fitfunc = lambda x, *p, :  p[0]*np.exp(-( ( (x-p[1])/p[2] )**2 )/2) + p[3]

    # amplitude fit
    x = amplitude_sweep
    xmin, xmax = np.min(x), np.max(x)
    y = amp_lineout
    ymin, ymax = np.min(y), np.max(y)
    ydiff = ymax - ymin
    s = np.flatnonzero(y > ydiff/2 + ymin) # Find data points above middle y value. First and last points used to estimate width
    # Seed paramters: amplitude, center, width, background
    p0 = [
        ydiff, 
        x[np.argmax(y)], 
        np.abs(x[s[0]]- x[s[-1]])/np.sqrt(8*np.log(2)), 
        ymin,
    ]
    bounds = ([0, xmin, 0, ymin-10*ydiff],
                [2*ydiff, xmax, 10*(xmax-xmin), ymax+10*ydiff])
    amp_fit_params, covar = optimize.curve_fit(fitfunc, x, y, p0=p0, bounds=bounds)
    amp_std_vec = np.sqrt(np.diag(covar))

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(amplitude_sweep, amp_lineout)
    xvals = np.linspace(xmin, xmax, 100)
    fit_y = fitfunc(xvals,*amp_fit_params)
    plt.plot(xvals, fit_y)
    plt.xlabel('Drive Amplitude')

    # frequency fit
    x = frequency_sweep
    xmin, xmax = np.min(x), np.max(x)
    y = freq_lineout
    ymin, ymax = np.min(y), np.max(y)
    ydiff = ymax - ymin
    s = np.flatnonzero(y > ydiff/2 + ymin) # Find data points above middle y value. First and last points used to estimate width
    # Seed paramters: amplitude, center, width, background
    p0 = [
        ydiff, 
        x[np.argmax(y)], 
        np.abs(x[s[0]]- x[s[-1]])/np.sqrt(8*np.log(2)), 
        ymin,
    ]
    bounds = ([0, xmin, 0, ymin-10*ydiff],
                [2*ydiff, xmax, 10*(xmax-xmin), ymax+10*ydiff])
    freq_fit_params, covar = optimize.curve_fit(fitfunc, x, y, p0=p0, bounds=bounds)
    freq_std_vec = np.sqrt(np.diag(covar))

    plt.subplot(212)
    plt.plot(frequency_sweep, freq_lineout)
    xvals = np.linspace(xmin, xmax, 100)
    fit_y = fitfunc(xvals,*freq_fit_params)
    plt.plot(xvals, fit_y)
    plt.xlabel('Drive Frequency')
    plt.tight_layout()
    plt.close()

    fit_dict = {
        'frequency': {
            'fit_values': {
                'peak_amplitude': freq_fit_params[0],
                'center': freq_fit_params[1],
                'width': freq_fit_params[2],
                'background': freq_fit_params[3],
            },
            'fit_uncertainties':{
                'peak_amplitude': freq_std_vec[0],
                'center': freq_std_vec[1],
                'width': freq_std_vec[2],
                'background': freq_std_vec[3],
            },
        },
        'amplitude': {
            'fit_values': {
                'peak_amplitude': amp_fit_params[0],
                'center': amp_fit_params[1],
                'width': amp_fit_params[2],
                'background': amp_fit_params[3],
            },
            'fit_uncertainties':{
                'peak_amplitude': amp_std_vec[0],
                'center': amp_std_vec[1],
                'width': amp_std_vec[2],
                'background': amp_std_vec[3],
            },
        },

    }

    # Solve for pi_half amplitude. set gaussian equal to half amplitude
    pi_half_amp = amp_fit_params[1] - amp_fit_params[2]*np.sqrt(2*np.log(2))

    return fit_dict, pi_half_amp, fig




if __name__ == "__main__":
    # while True:
    #     try:
    ef_rabi_chevron('q1_ef')
        # except Exception as e:
        #     print(e)
        #     continue