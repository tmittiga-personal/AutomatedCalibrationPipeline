"""
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubit frequency (qubit_IF) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from create_multiplexed_configuration import *
from utils import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
from typing import *

# Assumes 2 MHz detuning. 4 points per oscillation
DEFAULT_TAUS = np.arange(4, 40124//4 , 124//4)

class Ramsey_w_virtual_rotation:
    def __init__(
        self, 
        qubit: str,
        n_avg: int = 500,
        detuning: float = 2e6,  # in Hz
        taus: np.typing.NDArray = DEFAULT_TAUS,
    ):
        self.mc = create_multiplexed_configuration()
        self.qubit = qubit
        self.resonator = self.mc.qubit_resonator_correspondence[self.qubit]
        self.n_avg = n_avg
        self.detuning = detuning
        self.taus = taus

    def ramsey_w_virtual_rotation(
        self,
        simulate: bool = False,
    ):
        """

        Docstring needed
        # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
        
        # Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
        

        """
        #TODO: replace all variables with self.variable
        n_avg = self.n_avg
        detuning = self.detuning
        taus = self.taus

        data_handler = DataHandler(root_data_folder="./")

        ###################
        # The QUA program #
        ###################
        
        
        ramsey_w_virtual_rotation_data = {
            "n_avg": n_avg,
            "resonator_LO": self.mc.RL_CONSTANTS["rl1"]["LO"],
            "readout_amp": self.mc.RR_CONSTANTS[self.mc.qubit_resonator_correspondence[self.qubit]]["amplitude"],
            "qubit_LO": self.mc.MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
            "qubit_IF": self.mc.QUBIT_CONSTANTS[self.qubit]["IF"],
            "qubit_octave_gain": self.mc.qubit_octave_gain,
            "resonator_octave_gain": self.mc.resonator_octave_gain,
        }

        with program() as ramsey:
            n = declare(int)  # QUA variable for the averaging loop
            tau = declare(int)  # QUA variable for the idle time
            phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
            I = declare(fixed)  # QUA variable for the measured 'I' quadrature
            Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
            state = declare(bool)  # QUA variable for the qubit state
            I_st = declare_stream()  # Stream for the 'I' quadrature
            Q_st = declare_stream()  # Stream for the 'Q' quadrature
            state_st = declare_stream()  # Stream for the qubit state
            n_st = declare_stream()  # Stream for the averaging iteration 'n'

            with for_(n, 0, n < n_avg, n + 1):
                with for_(*from_array(tau, taus)): #for_each_(tau, taus): #for_(*from_array(tau, taus)):
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * tau))
                    # Strict_timing ensures that the sequence will be played without gaps
                    with strict_timing_():
                        play("cw", self.resonator)
                        align(self.qubit, self.resonator)
                        # 1st x90 gate
                        play("x180", self.qubit)
                        # Wait a varying idle time
                        wait(tau, self.qubit)
                    # Align the two elements to measure after playing the qubit pulse.
                    align(self.qubit, self.resonator)
                    # Measure the state of the resonator
                    if self.mc.RR_CONSTANTS[self.resonator]["use_opt_readout"]:
                        measure(
                            "readout",
                            self.resonator,
                            None,
                            dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I),
                            dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q),
                        )
                    else:
                        measure(
                            "readout",
                            self.resonator,
                            None,
                            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                        )
                    
                    # Wait for the qubit to decay to the ground state
                    wait(self.mc.thermalization_time * self.mc.u.ns, self.resonator)
                    # State discrimination
                    assign(state, I > self.mc.RR_CONSTANTS[self.resonator]['ge_threshold'])
                    # Save the 'I', 'Q' and 'state' to their respective streams
                    save(I, I_st)
                    save(Q, Q_st)
                    save(state, state_st)
                    # Reset the frame of the qubit in order not to accumulate rotations
                    reset_frame(self.qubit)
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                I_st.buffer(len(taus)).average().save("I")
                Q_st.buffer(len(taus)).average().save("Q")
                state_st.boolean_to_int().buffer(len(taus)).average().save("state")
                n_st.save("iteration")

        #####################################
        #  Open Communication with the QOP  #
        #####################################
        qmm = QuantumMachinesManager(
            host=self.mc.qop_ip, 
            port=self.mc.qop_port, 
            cluster_name=self.mc.cluster_name, 
            octave=self.mc.octave_config
        )
        ###########################
        # Run or Simulate Program #
        ###########################

        if simulate:
            # Simulates the QUA program for the specified duration
            simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
            job = qmm.simulate(self.mc.config, ramsey, simulation_config)
            job.get_simulated_samples().con1.plot()
        else:
            # Open the quantum machine
            qm = qmm.open_qm(self.mc.config)
            # Send the QUA program to the OPX, which compiles and executes it
            job = qm.execute(ramsey)
            # Get results from QUA program
            results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
            # Live plotting
            fig = plt.figure()
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
            while results.is_processing():
                # Fetch results
                I, Q, state, iteration = results.fetch_all()
                # Convert the results into Volts
                I = self.mc.u.demod2volts(I, self.mc.RR_CONSTANTS[self.resonator]["readout_length"])
                Q = self.mc.u.demod2volts(Q, self.mc.RR_CONSTANTS[self.resonator]["readout_length"])
                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.get_start_time())
                # Plot results
                plt.suptitle(f"Ramsey with frame rotation (detuning={detuning / self.mc.u.MHz} MHz)")
                plt.subplot(311)
                plt.cla()
                plt.plot(4 * taus, I, ".")
                plt.ylabel("I quadrature [V]")
                plt.subplot(312)
                plt.cla()
                plt.plot(4 * taus, Q, ".")
                plt.ylabel("Q quadrature [V]")
                plt.subplot(313)
                plt.cla()
                plt.plot(4 * taus, state, ".")
                plt.ylim((0, 1))
                plt.xlabel("Idle time [ns]")
                plt.ylabel("State")
                plt.pause(0.1)
                plt.tight_layout()
            ramsey_w_virtual_rotation_data["figure"] = fig
            plt.close()

            # Fit the results to extract the qubit frequency and T2*
            fit_dict = {}
            from qualang_tools.plot.fitting import Fit

            fit = Fit()
            fig2 = plt.figure()
            
            # Pick higher contrast data for fit
            contrast_I = np.abs(np.max(I) - np.min(I))
            contrast_Q = np.abs(np.max(Q) - np.min(Q))
            if contrast_I > contrast_Q:
                fit_data = I
                y_label = "I quadrature [V]"
            else:
                fit_data = Q
                y_label = "Q quadrature [V]"

            ramsey_fit = fit.T1(4 * taus, state, plot=True)
            qubit_T2 = np.abs(ramsey_fit["T1"][0])
            plt.xlabel("Idle time [ns]")
            plt.ylabel(y_label)
            print(f"T1 = {qubit_T2:.0f} ns")
            plt.legend((f"T1 = {qubit_T2:.0f} ns"))
            plt.title("T1")
            fit_dict = {
                'fit_values': {
                    'amp': ramsey_fit['amp'][0],
                    'qubit_T1': qubit_T2,
                    'final_offset': ramsey_fit['final_offset'][0],
                },
                'fit_uncertainties': {
                    'amp': ramsey_fit['amp'][1],
                    'qubit_T2': ramsey_fit['T1'][1],
                    'final_offset': ramsey_fit['final_offset'][1],
                },

            }
            ramsey_w_virtual_rotation_data["qubit_T1"] = qubit_T2
            ramsey_w_virtual_rotation_data["I"] = I
            ramsey_w_virtual_rotation_data["Q"] = Q
            ramsey_w_virtual_rotation_data["state"] = state
            ramsey_w_virtual_rotation_data["figure2"] = fig2
            data_folder = data_handler.save_data(
                ramsey_w_virtual_rotation_data, 
                name=f"{self.qubit}_T1_vs_quasiparticle"
            )
            plt.close()
            return fit_dict, data_folder


if __name__ == "__main__":
    mr = Ramsey_w_virtual_rotation(
        'q3_xy',
        taus = np.arange(4, 501240//4 , (5240)//2)
    )
    # while True:
        # try:
    # for i in range(2):
    mr.ramsey_w_virtual_rotation()
        # except Exception as e:
        #     print(e)
        #     continue