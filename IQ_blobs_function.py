"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the configuration.
    - Update the g -> e threshold (ge_threshold) in the configuration.
"""

from qm.qua import *
from qm import SimulationConfig
from qm import QuantumMachinesManager
from multiplexed_configuration import *
from utils import *
from qualang_tools.analysis.discriminator import two_state_discriminator_plot
from qualang_tools.results.data_handler import DataHandler
import matplotlib.pyplot as plt

def measure_IQ_blobs(
    qubit,
    resonator,
    n_runs = 10_000,
):
    data_handler = DataHandler(root_data_folder="./")

    ###################
    # The QUA program #
    ###################

    IQ_blobs_data = {
        "n_runs": n_runs,
        "resonator_LO": RL_CONSTANTS["rl1"]["LO"],
        "readout_amp": RR_CONSTANTS[qubit_resonator_correspondence[qubit]]["amplitude"],
        "qubit_LO": MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
        "qubit_IF": QUBIT_CONSTANTS[qubit]["IF"],
    }

    with program() as IQ_blobs:
        n = declare(int)
        I_g = declare(fixed)
        Q_g = declare(fixed)
        I_g_st = declare_stream()
        Q_g_st = declare_stream()
        I_e = declare(fixed)
        Q_e = declare(fixed)
        I_e_st = declare_stream()
        Q_e_st = declare_stream()

        with for_(n, 0, n < n_runs, n + 1):

            if RR_CONSTANTS[resonator]["use_opt_readout"]:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I),
                    dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q),
                )
            else:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
            # Wait for the qubit to decay to the ground state in the case of measurement induced transitions
            wait(thermalization_time * u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)

            align()  # global align
            # Play the x180 gate to put the qubit in the excited state

            play("x180", qubit)
            # Align the two elements to measure after playing the qubit pulse.
            align("qubit", resonator)
            # Measure the state of the resonator
            if RR_CONSTANTS[resonator]["use_opt_readout"]:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I),
                    dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q),
                )
            else:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )

            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)

        with stream_processing():
            # Save all streamed points for plotting the IQ blobs
            I_g_st.save_all("I_g")
            Q_g_st.save_all("Q_g")
            I_e_st.save_all("I_e")
            Q_e_st.save_all("Q_e")

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
        job = qmm.simulate(config, IQ_blobs, simulation_config)
        job.get_simulated_samples().con1.plot()

    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(IQ_blobs)
        # Creates a result handle to fetch data from the OPX
        res_handles = job.result_handles
        # Waits (blocks the Python console) until all results have been acquired
        res_handles.wait_for_all_values()
        # Fetch the 'I' & 'Q' points for the qubit in the ground and excited states
        Ig = res_handles.get("I_g").fetch_all()["value"]
        Qg = res_handles.get("Q_g").fetch_all()["value"]
        Ie = res_handles.get("I_e").fetch_all()["value"]
        Qe = res_handles.get("Q_e").fetch_all()["value"]
        # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
        # for state discrimination and derive the fidelity matrix
        angle, threshold, fidelity, gg, ge, eg, ee, fig = two_state_discriminator_plot(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)
        results_dict = {
            "fidelity": fidelity,
            "angle": angle,
            "threshold": threshold,
        }
        #########################################
        # The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
        # the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
        # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
        # Once we do this, we can perform active reset using:
        #########################################
        
        IQ_blobs_data["I_g"] = Ig
        IQ_blobs_data["Q_g"] = Qg
        IQ_blobs_data["I_e"] = Ie
        IQ_blobs_data["Q_e"] = Qe

        IQ_blobs_data["figure"] = fig

        data_folder = data_handler.save_data(IQ_blobs_data, name=f"{qubit}_IQ_blobs")

        plt.close()
        return results_dict, data_folder