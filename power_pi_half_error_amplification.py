"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x90, square_pi, or similar) 'N' times and
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
    - Update the qubit pulse amplitude (x90_amp) in the configuration.
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
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
a_min = 0.0
a_max = 1.5
n_a = 150
amplitudes = np.linspace(a_min, a_max, n_a)
# Number of applied Rabi pulses sweep
max_nb_of_pulses = 30  # Maximum number of qubit pulses
nb_of_pulses = np.arange(0, max_nb_of_pulses, 4)  # Always play an 4x pi/2 pulses to end up in the same state

power_rabi_error_amplification_data = {
    "n_avg": n_avg,
    "resonator_LO": resonator_LO,
    "readout_amp": readout_amp,
    "qubit_LO": qubit_LO,
    "qubit_IF":qubit_IF,
    "x90_amp": x90_amp,
    "x90_Len": x90_len,
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
                    play("x90" * amp(a), "qubit")
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
simulate = False

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
        plt.suptitle("Power pi/2 with error amplification")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amplitudes * x90_amp, nb_of_pulses, I)
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("# of pi/2 pulses")
        plt.title("I quadrature [V]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amplitudes * x90_amp, nb_of_pulses, Q)
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.title("Q quadrature [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot(amplitudes * x90_amp, np.sum(Q, axis=0))
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("Sum along the # of pi/2 pulses")
        plt.pause(1)
        plt.tight_layout()
    print(f"Optimal x90_amp = {amplitudes[np.argmax(np.sum(Q, axis=0))] * x90_amp:.4f} V")
    power_rabi_error_amplification_data["I"] = I
    power_rabi_error_amplification_data["Q"] = Q
    power_rabi_error_amplification_data["figure"] = fig

    data_handler.save_data(power_rabi_error_amplification_data, name="power_pi_half_error_amplification")