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
from create_multiplexed_configuration import *
from utils import *
from qualang_tools.results.data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from copy import deepcopy

OPTIMIZED_READOUT = False
READOUT_TYPE = 'rotated'
SEARCH_FIDELITY_THRESHOLD = 95
MAX_ITERATIONS = 10

def readout_amplitude_binary_search(
    qubit,
    resonator,
    n_runs = 10_000,
):
    mc = create_multiplexed_configuration()
    initial_amplitude = mc.RR_CONSTANTS[resonator]["amplitude"]

    ###################
    # The QUA program #
    ###################

    IQ_blobs_data = {
        "n_runs": n_runs,
        "resonator_LO": mc.RL_CONSTANTS["rl1"]["LO"],
        "readout_amp": mc.RR_CONSTANTS[resonator]["amplitude"],
        "qubit_LO": mc.MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
        "qubit_IF": mc.QUBIT_CONSTANTS[qubit]["IF"],
    }

    results_dict = {
        'optimized': {},
        'rotated': {},
    }

    data_handler = DataHandler(root_data_folder="./")
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
            if resonator[-2:] == 're':
                # If this is a e-f qubit
                # State prep into e
                play("x180", qubit.replace('ef','xy'))
                align()

            if OPTIMIZED_READOUT:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I_g),
                    dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q_g),
                )
            else:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
                )
            # Wait for the qubit to decay to the ground state in the case of measurement induced transitions
            wait(mc.thermalization_time * mc.u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)

            align()  # global align
            # Play the x180 gate to put the qubit in the excited state

            if resonator[-2:] == 're':
                # If this is a e-f qubit
                # State prep into e
                play("x180", qubit.replace('ef','xy'))
                align()
            play("x180", qubit)
            # Align the two elements to measure after playing the qubit pulse.
            align(qubit, resonator)
            # Measure the state of the resonator
            if OPTIMIZED_READOUT:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I_e),
                    dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q_e),
                )
            else:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
                )

            # Wait for the qubit to decay to the ground state
            wait(mc.thermalization_time * mc.u.ns, resonator)
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
    qmm = QuantumMachinesManager(
        host=mc.qop_ip, 
        port=mc.qop_port, 
        cluster_name=mc.cluster_name, 
        octave=mc.octave_config
    )

    #####################
    ### Binary Search ###
    #####################

    config_copy = deepcopy(mc.config)
    iteration = 0
    amplitude_scale = 1
    still_testing = True
    best_amplitude = mc.RR_CONSTANTS[resonator]["amplitude"]
    best_fidelity = 0
    best_iteration = 0
    # to ensure we don't ring around the optimum, decrease changes to the amplitude over time
    ring_down = np.exp(-np.array(range(0,MAX_ITERATIONS))/(MAX_ITERATIONS/5))

    while still_testing and iteration < MAX_ITERATIONS:
        # modify readout amplitude
        config_copy["waveforms"][f"readout_wf_{resonator}"]={
            "type": "constant", 
            "sample": mc.RR_CONSTANTS[resonator]["amplitude"]*amplitude_scale
        }
        # Open the quantum machine
        qm = qmm.open_qm(config_copy)
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
        angle, threshold, fidelity, gg, ge, eg, ee, fig = two_state_discriminator_plot(Ig, Qg, Ie, Qe, b_print=False)

        outlier_count_g, outliers_g, fig_g = cluster_deterimination(Ig, Qg)
        outlier_count_e, outliers_e, fig_e = cluster_deterimination(Ie, Qe)
        
        #########################################
        # The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
        # the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
        # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
        # Once we do this, we can perform active reset using:
        #########################################
        
        IQ_blobs_data[f"I_g_{iteration}"] = Ig
        IQ_blobs_data[f"Q_g_{iteration}"] = Qg
        IQ_blobs_data[f"I_e_{iteration}"] = Ie
        IQ_blobs_data[f"Q_e_{iteration}"] = Qe

        IQ_blobs_data[f"figure_IQ_{iteration}"] = fig
        IQ_blobs_data[f"figure_g_{iteration}"] = fig_g
        IQ_blobs_data[f"figure_e_{iteration}"] = fig_e

        results_dict[READOUT_TYPE][iteration] = {
            "fidelity": fidelity,
            "angle": np.mod(angle + mc.RR_CONSTANTS[resonator]["rotation_angle"], 2*np.pi),  # Update angle
            "threshold": threshold,
            'ground_outliers': outliers_g,
            'excited_outliers': outliers_e,
            'amplitude': mc.RR_CONSTANTS[resonator]["amplitude"]*amplitude_scale,
        }
        plt.close()
        
        # determine quality of results
        high_enough_fidelity = fidelity > SEARCH_FIDELITY_THRESHOLD
        too_many_outliers = outlier_count_e > 1 or outlier_count_g > 1
        
        if fidelity > best_fidelity and not too_many_outliers:
            best_amplitude = mc.RR_CONSTANTS[resonator]["amplitude"]*amplitude_scale
            best_fidelity = fidelity
            best_iteration = iteration
            print(f'Iteration {iteration}, New Best Amplitude: {best_amplitude}, Fidelity: {best_fidelity}')

        if high_enough_fidelity and not too_many_outliers:
            # We've achieved our goal. terminate
            still_testing = False
        else:
            if not high_enough_fidelity and not too_many_outliers:
                amplitude_scale *= 0.5*ring_down[iteration]+1  # exponentially reduce scaling from 1.5 to 1
            elif high_enough_fidelity and too_many_outliers:
                amplitude_scale *= (1-ring_down[iteration])*0.5+0.5  # exponentially increase scaling from 0.5 to 1
            else:
                # fidelity too low and too many outliers, we prefer to get rid of outliers
                amplitude_scale *= (1-ring_down[iteration])*0.5+0.5  # exponentially increase scaling from 0.5 to 1
        

        iteration += 1
    results_dict[READOUT_TYPE]['best'] = results_dict[READOUT_TYPE][best_iteration]
    data_folder = data_handler.save_data(IQ_blobs_data, name=f"{qubit}_resonator_amplitude_binary_search")
    results_dict['data_folder'] = data_folder
    return results_dict


def cluster_deterimination(
    I,
    Q,
    n_clusters = 6,
):
    data_points = np.transpose(np.array([ I,Q]))

    # Specify the number of clusters
    

    # # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_points)

    # Get the mean locations (cluster centers)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    # Calculate the size of each cluster
    cluster_sizes = np.bincount(labels)

    # Sort cluster centers by cluster size
    sorted_indices = np.argsort(-cluster_sizes)  # Sort in descending order
    sorted_cluster_centers = cluster_centers[sorted_indices]

    ## Cluster outlier analysis
    # take all datapoints within 1/5 of data range of largest cluster
    bin_count=5
    I_min = np.min(data_points[:, 0])
    I_max = np.max(data_points[:, 0])
    Q_min = np.min(data_points[:, 1])
    Q_max = np.max(data_points[:, 1])

    I_bin_width = (I_max - I_min)/bin_count
    Q_bin_width = (Q_max - Q_min)/bin_count
    largest_cluster = sorted_cluster_centers[0]
    I_filtered = []
    Q_filtered = []
    for point_I, point_Q in zip(data_points[:, 0], data_points[:, 1]):
        if (
            np.abs(point_I-largest_cluster[0]) < I_bin_width and
            np.abs(point_Q-largest_cluster[1]) < Q_bin_width
        ):
            I_filtered.append(point_I)
            Q_filtered.append(point_Q)

    # Plot the data points and the cluster centers
    fig = plt.figure()
    plt.scatter(data_points[:, 0], data_points[:, 1], marker='o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Data Clusters with Mean Locations')

    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    filtered_stds = []
    for data in [I_filtered, Q_filtered]:
        hist, bin_edges = np.histogram(data, bins=round(50/bin_count), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit the function to the histogram data
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[np.max(hist), np.mean(data), np.std(data)])
        filtered_stds.append(popt[2])

    outlier_count = 0
    outliers = []
    for I_cc, Q_cc in zip (cluster_centers[:, 0], cluster_centers[:, 1]):
        I_dist = np.abs(I_cc - np.mean(I_filtered))
        Q_dist = np.abs(Q_cc - np.mean(Q_filtered))
        if I_dist > 2*filtered_stds[0] or Q_dist > 2*filtered_stds[1]:
            plt.plot(I_cc, Q_cc, 'o', color = (1-outlier_count/n_clusters, 1-outlier_count/n_clusters, 1-outlier_count/n_clusters)) #1-outlier_count/n_clusters
            outliers.append([I_cc, Q_cc])
            outlier_count += 1
    plt.close()
    return outlier_count, outliers, fig

if __name__ == "__main__":
    readout_amplitude_binary_search(
        'q3_xy',
        'q3_rr',
    )