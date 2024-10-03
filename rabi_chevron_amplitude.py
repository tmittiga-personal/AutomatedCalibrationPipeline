"""
        RABI CHEVRON (AMPLITUDE VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse amplitudes.
By analyzing the results, one can determine the qubit and estimate the x180 pulse amplitude for a specified duration.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse duration (labeled as "x180_len").
Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "qubit_IF", in the configuration.
    - Modify the qubit pulse amplitude setting, labeled as "x180_amp", in the configuration.
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

data_handler = DataHandler(root_data_folder="./")

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
# The frequency sweep parameters
span = 0.5 * u.MHz
df = 10 * u.kHz
dfs = np.arange(-span, span, df)  # The frequency vector
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
a_min = 0.75
a_max = 1.25
n_a = 26
amplitudes = np.linspace(a_min, a_max, n_a)

amplitude_rabi_chevron_data = {
    "n_avg": n_avg,
    "dfs": dfs,
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

with program() as rabi_amp_freq:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the qubit frequency
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the digital oscillator linked to the qubit element
            update_frequency("qubit", f + qubit_IF)
            with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the pulse amplitude pre-factor
                # Adjust the qubit pulse amplitude
                play("x180" * amp(a), "qubit")
                # Align the two elements to measure after playing the qubit pulse.
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
        I_st.buffer(n_a).buffer(len(dfs)).average().save("I")
        Q_st.buffer(n_a).buffer(len(dfs)).average().save("Q")
        n_st.save("iteration")


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
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_amp_freq)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.subplot(211)
        plt.suptitle(f"Rabi chevron with LO={qubit_LO / u.GHz}GHz and IF={qubit_IF / u.MHz}MHz")
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$")
        plt.pcolor(amplitudes * x180_amp, dfs / u.MHz, R)
        plt.ylabel("Frequency detuning [MHz]")
        plt.xlabel("Pulse amplitude [V]")
        plt.subplot(212)
        plt.cla()
        plt.title("Phase")
        plt.pcolor(amplitudes * x180_amp, dfs / u.MHz, np.unwrap(phase))
        plt.ylabel("Frequency detuning [MHz]")
        plt.xlabel("Pulse amplitude [V]")
        plt.tight_layout()
        plt.pause(1)

    amplitude_rabi_chevron_data["I"] = I
    amplitude_rabi_chevron_data["Q"] = Q
    amplitude_rabi_chevron_data["R"] = R
    amplitude_rabi_chevron_data["phase"] = phase
    amplitude_rabi_chevron_data["figure"] = fig

    data_handler.save_data(amplitude_rabi_chevron_data, name="amplitude_rabi_chevron")

    plt.show()