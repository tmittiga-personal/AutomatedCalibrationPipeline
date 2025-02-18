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
PERMITTED_RESONATORS = ['q3_re']

def tri_readout_amplitude_binary_search(
    qubit,
    resonator,
    i_attempt = 0,
    n_runs = 10_000,
):
    mc = create_multiplexed_configuration()
    initial_amplitude = mc.RR_CONSTANTS[resonator]["amplitude"]
    if i_attempt > 0: # and qubit[-2:]=='ef':
        df = pull_latest_n_calibrated_values(
            qubits = [qubit],
            search_parameter_names = ['tri_readout_amplitude'],
            n_latest = i_attempt,
            all_attempts = True,
        )
        # Loop over this round of attempts to calbirate and take an improved values
        for ii in range(i_attempt):
            if df['miscellaneous'].iloc[-1*(ii+1)]['results_dict']['improved']:
                initial_amplitude = df['calibration_value'].iloc[-1*(ii+1)]
                break
    assert resonator in PERMITTED_RESONATORS, f'Resonator {resonator} must be in {PERMITTED_RESONATORS}'

    df = pull_latest_n_calibrated_values(
        qubits = ['q3_xy'],
        search_parameter_names = ['readout_frequency'],
        n_latest = 1,
    )
    rfrr = df['calibration_value'].values[0]
    df = pull_latest_n_calibrated_values(
        qubits = ['q3_ef'],
        search_parameter_names = ['readout_frequency'],
        n_latest = 1,
    )
    rfre = df['calibration_value'].values[0]

    new_freq = (rfrr + rfre)/2

    ###################
    # The QUA program #
    ###################

    IQ_blobs_data = {
        "n_runs": n_runs,
        "resonator_LO": mc.RL_CONSTANTS["rl1"]["LO"],
        "readout_amp": initial_amplitude,
        "qubit_LO": mc.MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
        "qubit_IF": mc.QUBIT_CONSTANTS[qubit]["IF"],
        "ge_threshold": mc.RR_CONSTANTS[resonator]["ge_threshold"],
        "rotation_angle": mc.RR_CONSTANTS[resonator]["rotation_angle"],
        'resonator_freq': new_freq,
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
        I_f = declare(fixed)
        Q_f = declare(fixed)
        I_f_st = declare_stream()
        Q_f_st = declare_stream()

        with for_(n, 0, n < n_runs, n + 1):
            #################
            ### measure g ###
            #################
            align()
            update_frequency(resonator, new_freq)
            wait(100, resonator)

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
            align()

            #################
            ### measure e ###
            #################

            # State prep into e
            play("x180", qubit.replace('ef','xy'))
            align()

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
            # Wait for the qubit to decay to the ground state in the case of measurement induced transitions
            wait(mc.thermalization_time * mc.u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams for the e state
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)

            align()  # global align
            
            #################
            ### measure f ###
            #################

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
                    dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I_f),
                    dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q_f),
                )
            else:
                measure(
                    f"readout",
                    resonator,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_f),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_f),
                )

            # Wait for the qubit to decay to the ground state
            wait(mc.thermalization_time * mc.u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
            save(I_f, I_f_st)
            save(Q_f, Q_f_st)

        with stream_processing():
            # Save all streamed points for plotting the IQ blobs
            I_g_st.save_all("I_g")
            Q_g_st.save_all("Q_g")
            I_e_st.save_all("I_e")
            Q_e_st.save_all("Q_e")
            I_f_st.save_all("I_f")
            Q_f_st.save_all("Q_f")

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
    best_amplitude = initial_amplitude
    best_hm = 0
    first_valid_hm = 0
    first_valid_iteration = 0
    best_iteration = 0
    # to ensure we don't ring around the optimum, decrease changes to the amplitude over time
    ring_down = np.exp(-np.array(range(0,MAX_ITERATIONS))/(MAX_ITERATIONS/5))

    while still_testing and iteration < MAX_ITERATIONS:
        # modify readout amplitude
        config_copy["waveforms"][f"readout_wf_{resonator}"]={
            "type": "constant", 
            "sample": initial_amplitude*amplitude_scale
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
        If = res_handles.get("I_f").fetch_all()["value"]
        Qf = res_handles.get("Q_f").fetch_all()["value"]
        # For 3 state discrimination, we only care about the three states being separated by as much as possible, without adding additional lobes
        # Find distances
        d_ge = np.sqrt( (np.mean(Ie)-np.mean(Ig))**2 +  (np.mean(Qe)-np.mean(Qg))**2 )
        d_gf = np.sqrt( (np.mean(If)-np.mean(Ig))**2 +  (np.mean(Qf)-np.mean(Qg))**2 )
        d_ef = np.sqrt( (np.mean(Ie)-np.mean(If))**2 +  (np.mean(Qe)-np.mean(Qf))**2 )

        # maximize the harmonic mean of the three distances to try to reduce the possibility of minimizing a distance
        hm = 3*(1/d_ge + 1/d_gf + 1/d_ef)**-1

        outlier_count_g, outliers_g, fig_g, stds_g = cluster_deterimination(Ig, Qg)
        outlier_count_e, outliers_e, fig_e, stds_e = cluster_deterimination(Ie, Qe)
        outlier_count_f, outliers_f, fig_f, stds_f = cluster_deterimination(If, Qf)
        std_g = np.mean(stds_g)
        std_e = np.mean(stds_e)
        std_f = np.mean(stds_f)
        
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
        IQ_blobs_data[f"I_f_{iteration}"] = If
        IQ_blobs_data[f"Q_f_{iteration}"] = Qf
        # Find thresholding parameters
        avg_f, fidelity_m, fig_trisect, vertical_ordering, threshold_funcs = fidelity_3_state_discrimination(IQ_blobs_data, iteration)   

        IQ_blobs_data[f"figure_g_{iteration}"] = fig_g
        IQ_blobs_data[f"figure_e_{iteration}"] = fig_e
        IQ_blobs_data[f"figure_f_{iteration}"] = fig_f
        IQ_blobs_data[f"figure_trisect_{iteration}"] = fig_trisect


        results_dict[READOUT_TYPE][iteration] = {
            "harmonic_mean_distance": hm,
            'ground_outliers': outliers_g,
            'excited_outliers': outliers_e,
            'fstate_outliers': outliers_f,
            'amplitude': initial_amplitude*amplitude_scale,
            'avg_fidelity': avg_f,
            'fidelity_matrix': fidelity_m,
            'vertical_ordering': vertical_ordering,
            'threshold_funcs': threshold_funcs,
        }
        plt.close()
        # determine quality of results
        # If 
        high_enough_distances = d_ef > 2*(std_e + std_f) and d_ge > 2*(std_e + std_g) and d_gf > 2*(std_g + std_f)
        too_many_outliers = outlier_count_e > 1 or outlier_count_g > 1 or outlier_count_f > 1
        
        if hm > best_hm and not too_many_outliers:
            best_amplitude = initial_amplitude*amplitude_scale
            best_hm = hm
            best_iteration = iteration

            if first_valid_hm == 0:
                first_valid_hm = hm
                first_valid_iteration = iteration
            print(f'Iteration {iteration}, New Best Amplitude: {best_amplitude}, Harmonic Mean Distance: {best_hm}')

        if high_enough_distances and not too_many_outliers:
            # We've achieved our goal. terminate
            still_testing = False
        else:
            if not high_enough_distances and not too_many_outliers:
                amplitude_scale *= 0.5*ring_down[iteration]+1  # exponentially reduce scaling from 1.5 to 1
            elif high_enough_distances and too_many_outliers:
                amplitude_scale *= (1-ring_down[iteration])*0.5+0.5  # exponentially increase scaling from 0.5 to 1
            else:
                # fidelity too low and too many outliers, we prefer to get rid of outliers
                amplitude_scale *= (1-ring_down[iteration])*0.5+0.5  # exponentially increase scaling from 0.5 to 1
        

        iteration += 1
    # Check for improvement 
    improved = False
    if first_valid_iteration > 0 or best_hm > first_valid_hm:
        # Wend from an amplitude that produced too many outliers, to a valid amplitude        
        # Or went from a valid amplitude to a better amplitude
        improved = True 
 
    results_dict[READOUT_TYPE]['best'] = results_dict[READOUT_TYPE][best_iteration]
    results_dict['improved'] = improved
    IQ_blobs_data['results_dict'] = results_dict        
    data_folder = data_handler.save_data(IQ_blobs_data, name=f"{qubit}_3statediscrimination_amplitude_binary_search")
    results_dict['data_folder'] = data_folder
    return results_dict


def fidelity_3_state_discrimination(file_contents, iteration):
    I_g = file_contents[f'I_g_{iteration}']
    I_e = file_contents[f'I_e_{iteration}']
    I_f = file_contents[f'I_f_{iteration}']
    Q_g = file_contents[f'Q_g_{iteration}']
    Q_e = file_contents[f'Q_e_{iteration}']
    Q_f = file_contents[f'Q_f_{iteration}']

    # Find location of each blob
    loc_g = [np.mean(I_g), np.mean(Q_g)]
    loc_e = [np.mean(I_e), np.mean(Q_e)]
    loc_f = [np.mean(I_f), np.mean(Q_f)]

    # Figure out the ordering of the states in IQ space
    ### Assuming a roughly EQUILATERAL triangular arrangement of the states 
    ### (and that the states will never be exactly on top/bottom of each other)
    ### There are only 4 distinguishable arrangements:
    ### 1&2) top state is leftmost/rightmost state
    ### 3&4) bottom state is leftmost/rightmost

    origI = np.array([loc_g[0], loc_e[0], loc_f[0]])
    origQ = np.array([loc_g[1], loc_e[1], loc_f[1]])
    bottom_sorted_indices = np.argsort(origQ)
    left_sorted_indices = np.argsort(origI)
    bottomQ = origQ[bottom_sorted_indices]
    leftI= origI[left_sorted_indices]

    # dict to convert between gef and top/mid/bottom
    simple_ind = np.array([0,1,2])
    vertical_ordering = {
        'g': simple_ind[bottom_sorted_indices==0][0],
        'e': simple_ind[bottom_sorted_indices==1][0], 
        'f': simple_ind[bottom_sorted_indices==2][0],
    }

    # Determine which arrangement we're in
    if left_sorted_indices[0] == bottom_sorted_indices[-1]:
        # if the topmost state is also the leftmost state
        loc_t = [leftI[0], bottomQ[2]]
        loc_m = [leftI[2], bottomQ[1]]
        loc_b = [leftI[1], bottomQ[0]]
    elif left_sorted_indices[-1] == bottom_sorted_indices[-1]:
        # if the topmost state is also the rightmost state
        loc_t = [leftI[2], bottomQ[2]]
        loc_m = [leftI[0], bottomQ[1]]
        loc_b = [leftI[1], bottomQ[0]]
    elif left_sorted_indices[0] == bottom_sorted_indices[0]:
        # if the bottommost state is also the leftmost state
        loc_t = [leftI[1], bottomQ[2]]
        loc_m = [leftI[2], bottomQ[1]]
        loc_b = [leftI[0], bottomQ[0]]
    elif left_sorted_indices[-1] == bottom_sorted_indices[0]:
        # if the bottommost state is also the rightmost state
        loc_t = [leftI[1], bottomQ[2]]
        loc_m = [leftI[0], bottomQ[1]]
        loc_b = [leftI[2], bottomQ[0]]

    # create lines passing through the center of each blob and tri-secting IQ space optimally
    mtm = -1*(loc_t[0] - loc_m[0])/(loc_t[1] - loc_m[1])
    mmb = -1*(loc_m[0] - loc_b[0])/(loc_m[1] - loc_b[1])
    mtb = -1*(loc_t[0] - loc_b[0])/(loc_t[1] - loc_b[1])

    t_func = lambda x: mmb*(x-loc_t[0]) + loc_t[1]
    m_func = lambda x: mtb*(x-loc_m[0]) + loc_m[1]
    b_func = lambda x: mtm*(x-loc_b[0]) + loc_b[1]
    #store for use by experiments
    funcs = [b_func, m_func, t_func]

    read_t_i = []
    read_t_q = []
    read_m_i = []
    read_m_q = []
    read_b_i = []
    read_b_q = []
    # parse
    fidelity_matrix = np.zeros((3,3))

    ### ONlY true for roughly equilateral triangle
    for i_state, (I_s, Q_s) in enumerate(zip([I_g, I_e, I_f], [Q_g, Q_e, Q_f])):
        num_points = len(I_s)
        t_count = 0
        m_count = 0
        b_count = 0
        for i, q in zip(I_s, Q_s):
            if q > m_func(i) and q > b_func(i):
                read_t_i.append(i)
                read_t_q.append(q)
                t_count += 1
            elif q > t_func(i) and q < b_func(i):
                read_m_i.append(i)
                read_m_q.append(q)
                m_count += 1
            elif q < t_func(i) and q < m_func(i):
                read_b_i.append(i)
                read_b_q.append(q)
                b_count += 1
        for i_tb, count in enumerate([b_count, m_count, t_count]):

            if i_tb == vertical_ordering['g']:
                second_index = 0
            elif i_tb == vertical_ordering['e']:
                second_index = 1
            elif i_tb == vertical_ordering['f']:
                second_index = 2
                
            fidelity_matrix[i_state][second_index] = count/num_points

    fig_trisect = plt.figure(figsize=(7,7))
    plt.plot(read_t_i, read_t_q, 'r.', markersize=2)
    plt.plot(read_m_i, read_m_q, 'b.', markersize=2)
    plt.plot(read_b_i, read_b_q, 'g.', markersize=2)

    print(fidelity_matrix)
    avg_f = (fidelity_matrix[0][0] + fidelity_matrix[1][1] + fidelity_matrix[2][2])/3
    return avg_f, fidelity_matrix, fig_trisect, vertical_ordering, funcs


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
    filtered_stds = np.array(filtered_stds)

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
    return outlier_count, outliers, fig, filtered_stds

if __name__ == "__main__":
    readout_amplitude_binary_search(
        'q3_ef',
        'q3_re',
    )