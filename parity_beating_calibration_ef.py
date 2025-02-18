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
from utils import *
import time
from qm import CompilerOptionArguments
import peakutils
from scipy import optimize

class parity_ramsey:

    def __init__(
        self,
        probe_qubit,
    ):
        self.mc = create_multiplexed_configuration()
        self.probe_qubit = probe_qubit
        self.filename_qubits = ''
        self.filename_qubits += self.probe_qubit +'_'
        self.f_artificial = 0.0 * self.mc.u.MHz #0.1
        self.n_avg_per_job = 30  # The number of averages taken for a single job
        self.n_rounds = 7  # The number of rounds to repeat each chunk of the program
        self.program_divisions = 10  # the number of chunks to divide the program into
        # Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
        # taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus
        # self.taus = np.arange(4, 301240//4 , (1240)//2)
        self.taus = np.arange(4, 21240//4 , (200)//4)


        
        self.qubit_resonator_correspondence = {qu: res for qu, res in zip(self.mc.QUBIT_CONSTANTS.keys(), self.mc.RR_CONSTANTS.keys())}
        self.resonator = self.qubit_resonator_correspondence[self.probe_qubit]
        self.threshold = self.mc.RR_CONSTANTS[self.resonator]["ge_threshold"]

        self.multiplex_ramsey_data = {
            "n_avg": self.n_avg_per_job * self.n_rounds,
            "resonator": self.resonator,
            "RL_CONSTANTS": self.mc.RL_CONSTANTS,
            "qubit_IF": self.mc.QUBIT_CONSTANTS[self.probe_qubit]['IF'],
            "MULTIPLEX_DRIVE_CONSTANTS": self.mc.MULTIPLEX_DRIVE_CONSTANTS,
            "probe_qubit": self.probe_qubit,
            "qubit_octave_gain": self.mc.qubit_octave_gain,
            "taus": self.taus,
        }
        
        self.precomile_muliplexed_ramsey()

    def precomile_muliplexed_ramsey(self):

        self.data_handler = DataHandler(root_data_folder="./")

        ###################
        # The QUA program #
        ###################        
        

        ###########################################
        ### Split taus among different programs ###
        ###########################################
        assert isinstance(self.program_divisions, int), 'programs_divisions must be an integer'
        index_length = len(self.taus)//self.program_divisions
        index_remainder = len(self.taus)%self.program_divisions

        self.tau_arrays = []
        for i in range(self.program_divisions):
            if i ==self.program_divisions-1:
                self.tau_arrays.append(self.taus[i*index_length:i*index_length+index_length+index_remainder])
            else:
                self.tau_arrays.append(self.taus[i*index_length:i*index_length+index_length])
        # Automatically select resonators associated with the probed qubits
        assert set([self.probe_qubit]).issubset(self.mc.QUBIT_CONSTANTS.keys()), "All probe qubits must be found in QUBIT_CONSTANTS"

        #########################
        ### Precompile Pulses ###
        #########################
        # generate config pulses to match time sweep
        config_copy = deepcopy(self.mc.config)  # copy the config for safety
        probe_qubit = self.probe_qubit
        resonator = self.resonator

        copy_x90_pulse = deepcopy(self.mc.config['pulses'][f'x90_pulse_{probe_qubit}'])  # Copy the x180 pulse dict so we don't change the original
        x90_I_wf_copy = deepcopy(self.mc.config['waveforms'][f"x90_I_wf_{probe_qubit}"])  # Same for waveform. Will be used for both I and Q
        x90_amp = self.mc.QUBIT_CONSTANTS[probe_qubit]["pi_half_amplitude"]
        pulse_length = self.mc.QUBIT_CONSTANTS[probe_qubit]["pi_half_len"]
        drag_coef = self.mc.QUBIT_CONSTANTS[probe_qubit]['drag_coefficient']
        anharmonicity = self.mc.QUBIT_CONSTANTS[probe_qubit]['anharmonicity']
        AC_stark_detuning = self.mc.QUBIT_CONSTANTS[probe_qubit]['ac_stark_shift']

        # Calculate new waveforms for I and Q
        Ivar_wf, Qvar_wf = np.array(
            drag_gaussian_pulse_waveforms(x90_amp, pulse_length, pulse_length/5, drag_coef, anharmonicity, AC_stark_detuning)
        )

        for t in self.taus:
            final_phase = self.f_artificial * 1e-9 * 4 * t

            config_copy['elements'][probe_qubit]['operations'][f'xvar{t}'] = f'xvar{t}_pulse_{probe_qubit}'
            
            # Rotate components by final phase
            I_wf = Ivar_wf*np.cos(2*np.pi*final_phase) - Qvar_wf*np.sin(2*np.pi*final_phase)
            Q_wf = Ivar_wf*np.sin(2*np.pi*final_phase) + Qvar_wf*np.cos(2*np.pi*final_phase)

            # Assign modified waveform to I
            x90_I_wf_copy['samples'] = deepcopy(I_wf)
            config_copy['waveforms'][f'xvar{t}_I_wf_{probe_qubit}'] = deepcopy(x90_I_wf_copy)
            # Assign modified waveform to Q
            x90_I_wf_copy['samples'] = deepcopy(Q_wf)
            config_copy['waveforms'][f'xvar{t}_Q_wf_{probe_qubit}'] = deepcopy(x90_I_wf_copy)

            # Create new pulse
            copy_x90_pulse['waveforms']['I'] = f'xvar{t}_I_wf_{probe_qubit}'  # refer to new I waveform
            copy_x90_pulse['waveforms']['Q'] = f'xvar{t}_Q_wf_{probe_qubit}'  # refer to new Q waveform
            # Assign copy of modified dict so each new pulse points to a unique dictionary, rather than the same mutible one
            config_copy['pulses'][f'xvar{t}_pulse_{probe_qubit}']=deepcopy(copy_x90_pulse)  

        programs = []
        
        for i_i in range(self.program_divisions):
            # program_code = f"""
            with program() as multiplex_ramsey: #_{i_i}:
                n = declare(int)  # QUA variable for the averaging loop
                t = declare(int)  # QUA variable for the wait time
                I_cases = declare(fixed)  # QUA variable for the measured 'I' quadrature in each case
                Q_cases = declare(fixed)  # QUA variable for the measured 'Q' quadrature in each case
                I_thresholded = declare(int)  # QUA variable for storing when I > or < voltage threshold

                n_st = declare_stream()  # Stream for the averaging iteration 'n'

                I_thresholded_st = declare_stream()  # Stream for the 'I' quadrature in each case
                I_st = declare_stream()
                Q_st = declare_stream()

                final_phase = declare(fixed, 0)

                with for_(n, 0, n < self.n_avg_per_job, n + 1):        
                    for t in self.tau_arrays[i_i]:  
                        assign(final_phase, Cast.mul_fixed_by_int(self.f_artificial * 1e-9, 4 * t))
                        
                        ##################
                        ### Individual ###
                        ##################
                        for i_case in range(2):
                            # Loop between T1 and T2* measurements.
                            # we're interested in T1 of |e> so we only have to wait for T1. No Pulses after state prep to |e>
                            with strict_timing_():           
                                align()                                   
                                
                                # create superposition for T2* measurement
                                if resonator[-2:] == 're':
                                    # If this is a e-f qubit
                                    # State prep into e
                                    play("x180", probe_qubit.replace('ef','xy'))
                                    align()
                                match i_case:
                                    case 1:
                                        # create superposition for T2* measurement
                                        play("x90", probe_qubit)


                                wait(t, probe_qubit)
                                match i_case:
                                    case 1:
                                        # Project the T2* measurement
                                        # play(f'xvar{t}', probe_qubit)
                                        frame_rotation_2pi(final_phase, probe_qubit)
                                        play("x90", probe_qubit)
                                        reset_frame(probe_qubit)
                                align()

                                measure(
                                    "readout",
                                    resonator,
                                    None,
                                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                                )       
                                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                                assign(I_thresholded, Util.cond(I_cases>self.threshold, 1, 0))
                                save(I_thresholded, I_thresholded_st)
                                save(I_cases, I_st)
                                save(Q_cases, Q_st)
                        # Wait for the qubit to decay to the ground state
                        wait(self.mc.thermalization_time * self.mc.u.ns, resonator)
                    # Save the averaging iteration to get the progress bar
                    save(n, n_st)

                with stream_processing():
                    # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                    # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
                    # get_equivalent_log_array() is used to get the exact values used in the QUA program.
                    
                    I_thresholded_st.buffer(2).buffer(len(self.tau_arrays[i_i])).average().save(f"I_thresholded")
                    I_st.buffer(2).buffer(len(self.tau_arrays[i_i])).average().save(f"I")
                    Q_st.buffer(2).buffer(len(self.tau_arrays[i_i])).average().save(f"Q")
                    n_st.save("iteration") #"""
            # exec(program_code)
            # eval(f'programs.append(multiplex_ramsey_{i_i})')
            programs.append(multiplex_ramsey)
        self.programs = programs
        self.config = config_copy

        
        #####################################
        #  Open Communication with the QOP  #
        #####################################
        qmm = QuantumMachinesManager(host=self.mc.qop_ip, 
                                     port=self.mc.qop_port, 
                                     cluster_name=self.mc.cluster_name, 
                                     octave=self.mc.octave_config)
        # Open the quantum machine
        # qm = qmm.open_qm(config)
        self.qm = qmm.open_qm(self.config)
        ###########################
        ### Precompile programs ###
        ###########################
        self.precompiled_programs =[]
        my_compiler_options = CompilerOptionArguments(flags=["auto-element-thread"])
        for i_i in range(self.program_divisions):
            self.precompiled_programs.append(self.qm.compile(self.programs[i_i], compiler_options=my_compiler_options))


    def run_parity_ramsey(self):       
        # Initialize data dicts
        resonator = self.resonator
        self.data_dict = {resonator: {}}
        self.program_data_dict = {}
        # print('Compiled')

        start_time = time.time()
        n_jobs = 0
        # Send the QUA program to the OPX, which compiles and executes it
        for _ in range(self.n_rounds):
            pending_jobs = []
            for i_i in range(len(self.precompiled_programs)):
                pending_jobs.append(self.qm.queue.add_compiled(self.precompiled_programs[i_i]))
            ###############################
            ### Wait for Queue to Empty ###
            ###############################
            # print('Submittted')
            
            # Pull results from each job submitted
            for i_pj, pending_job in enumerate(pending_jobs): 

                # Get results from QUA program
                fetch_names = ["iteration", "I_thresholded", "I", "Q"]
                # Wait for results from QUA program
                # print('Waiting for execution')
                job = pending_job.wait_for_execution()
                # print('waiting for values')
                job.result_handles.wait_for_all_values()
                # results = fetching_tool(job, data_list=fetch_names)
                progress_counter(n_jobs, self.program_divisions*self.n_rounds, start_time=start_time)                
                P = job.result_handles.I_thresholded.fetch_all()
                P = P.transpose()
                I = job.result_handles.I.fetch_all()
                I = I.transpose()
                Q = job.result_handles.Q.fetch_all()         
                Q = Q.transpose()

                self.program_data_dict[job.id] = {resonator: {}}
                    
                self.program_data_dict[job.id][resonator].update(
                    {
                        'taus':self.tau_arrays[i_pj],
                        'P_T1': P[0],
                        'I_T1': I[0],
                        'Q_T1': Q[0],
                        'P_T2star': P[1],
                        'I_T2star': I[1],
                        'Q_T2star': Q[1],
                    }
                )
                n_jobs += 1
        # Extract all data and sort into data_dict
        first_job_id = int(list(self.program_data_dict.keys())[0])
        P_T2star = np.zeros(len(self.taus))
        I_T2star = np.zeros(len(self.taus))
        Q_T2star = np.zeros(len(self.taus))
        P_T1 = np.zeros(len(self.taus))
        I_T1 = np.zeros(len(self.taus))
        Q_T1 = np.zeros(len(self.taus))
        last_index = 0
        n_pd = self.program_divisions
        for i_pd in range(n_pd):
            for i_r in range(self.n_rounds):
                if i_r == 0:
                    temp_P = self.program_data_dict[str(first_job_id+i_pd)][resonator]['P_T2star']
                    temp_I = self.program_data_dict[str(first_job_id+i_pd)][resonator]['I_T2star']
                    temp_Q = self.program_data_dict[str(first_job_id+i_pd)][resonator]['Q_T2star']
                    temp_P1 = self.program_data_dict[str(first_job_id+i_pd)][resonator]['P_T1']
                    temp_I1 = self.program_data_dict[str(first_job_id+i_pd)][resonator]['I_T1']
                    temp_Q1 = self.program_data_dict[str(first_job_id+i_pd)][resonator]['Q_T1']
                    program_length = len(temp_P)
                else:
                    temp_P += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['P_T2star']
                    temp_I += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['I_T2star']
                    temp_Q += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['Q_T2star']
                    temp_P1 += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['P_T1']
                    temp_I1 += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['I_T1']
                    temp_Q1 += self.program_data_dict[str(first_job_id+n_pd*i_r+i_pd)][resonator]['Q_T1']
            P_T2star[last_index: last_index+program_length] = np.array(temp_P)/self.n_rounds
            I_T2star[last_index: last_index+program_length] = np.array(temp_I)/self.n_rounds
            Q_T2star[last_index: last_index+program_length] = np.array(temp_Q)/self.n_rounds
            P_T1[last_index: last_index+program_length] = np.array(temp_P1)/self.n_rounds
            I_T1[last_index: last_index+program_length] = np.array(temp_I1)/self.n_rounds
            Q_T1[last_index: last_index+program_length] = np.array(temp_Q1)/self.n_rounds
            last_index += program_length
            

        cis = wilson_score_interval(P_T2star, self.n_avg_per_job*self.n_rounds, z=1)
        P_T2star_err = errorbars_from_intervals(P_T2star, cis)
        cis = wilson_score_interval(P_T1, self.n_avg_per_job*self.n_rounds, z=1)
        P_T1_err = errorbars_from_intervals(P_T1, cis)

        self.data_dict[resonator].update(
            {
                'P_T2star': P_T2star,
                'P_T2star_err': P_T2star_err,
                'I_T2star': I_T2star,
                'Q_T2star': Q_T2star,
                'P_T1': P_T1,
                'P_T1_err': P_T1_err,
                'I_T1': I_T1,
                'Q_T1': Q_T1,
            }
        )

        self.fit_parity_beat()

        # Save to file
        self.multiplex_ramsey_data["figure"] = self.fig
        self.multiplex_ramsey_data["measurement_data"] = self.data_dict
        # self.multiplex_ramsey_data['all_program_data'] = self.program_data_dict
        self.data_folder = self.data_handler.save_data(self.multiplex_ramsey_data, name=f"parity_beating_{self.filename_qubits}")
        plt.close()

        return 


    def fit_parity_beat(self):
        resonator = self.resonator
        taus = np.array(self.taus)*4
        
        It2 = self.data_dict[resonator]['I_T2star']
        Qt2 = self.data_dict[resonator]['Q_T2star']
        S = self.mc.u.demod2volts(It2 + 1j * Qt2, self.mc.RR_CONSTANTS[resonator]["readout_length"])
        y = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase

        # Rough estimates for seeding
        t2_amp_estimate = (np.max(y)-np.min(y))/2
        t1 = np.mean(taus)
        t2_bk_estimate = np.mean(y)
        t2_bk_shift = 0.2*np.abs(t2_bk_estimate)  # IF values are negative, we need this

        # get freq components
        fft_result = np.fft.fft(np.array(y) - np.mean(y))
        fft_freq = np.fft.fftfreq(len(y), d=(taus[1] - taus[0]))

        # Only keep the positive frequencies
        positive_freq_indices = np.where(fft_freq >= 0)
        fft_result = fft_result[positive_freq_indices]
        fft_freq = fft_freq[positive_freq_indices]
        df = fft_freq[1]-fft_freq[0]

        strengths = np.abs(fft_result)

        # try to pick two peak frequencies. Otherwise take two strongest
        
        try:
            indexes = peakutils.indexes(strengths, thres=np.std(strengths/np.max(strengths)), min_dist=1)
            peak_amps = strengths[indexes]
            peak_ind = np.argpartition(peak_amps, -2)[-2:]
            strongest_freqs = fft_freq[indexes][peak_ind]
            # print(strongest_freqs)
            peak_amps = peak_amps[peak_ind]

        except:
            print('two peaks not detected')
            peak_ind = np.argpartition(strengths, -2)[-2:]
            strongest_freqs = fft_freq[peak_ind]            
            peak_amps = strengths[peak_ind]
        strongest_freqs_amps = peak_amps/np.max(peak_amps) # normalize
        strongest_freqs_amps *= 1/(np.sum(strongest_freqs_amps))

        best_fit_resq = 1e10
        best_with_beat = False
        for fit_with_beating in [True, False]:
            if fit_with_beating:
                f_seeds = [[i,j] for i in np.arange(-0.5,0.6,0.5) for j in np.arange(-0.5,0.6,0.5)]
            else:
                f_seeds = [-1, 0, 1]

            for f_seed in f_seeds:
                    
                if fit_with_beating:
                    fitfunc = lambda x, *p, :  -1*((t2_amp_estimate-p[0])*np.cos(2*np.pi*(p[1]*x)) + p[0]*np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
                    # fitfunc = lambda x, *p, :  ((0.5-p[5])*p[0]*np.cos(2*np.pi*(p[1]*x)) + p[5]*np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
                    # amp1, freq1, background, t2star, freq2, amp2
                    p0=[t2_amp_estimate*strongest_freqs_amps[0], #(np.max(y)-np.min(y))/4*strongest_freqs_amps[1], 
                        strongest_freqs[1]+df*f_seed[1], 
                        t2_bk_estimate, 
                        t1, 
                        strongest_freqs[0]+df*f_seed[0]] #(np.max(y)-np.min(y))/4*strongest_freqs_amps[0]]
                    
                    bounds = ([0,strongest_freqs[1]-1.5*df, t2_bk_estimate - t2_bk_shift, 500,  strongest_freqs[0]-1.5*df],
                                [t2_amp_estimate,strongest_freqs[1]+1.5*df, t2_bk_estimate + t2_bk_shift, 2.05*t1, strongest_freqs[0]+1.5*df])
                    
                else:
                    fitfunc = lambda x, *p, :  -1*p[0]*np.cos(2*np.pi*(p[1]*x))*np.exp(-x/p[3]) + p[2]
                    p0=[t2_amp_estimate, strongest_freqs[1] +df*f_seed, t2_bk_estimate, t1] 
                    bounds = ([0,   0.5*strongest_freqs[1], t2_bk_estimate - t2_bk_shift, 500 ],
                                [0.5, 1.5*strongest_freqs[1], t2_bk_estimate + t2_bk_shift, 2.1*t1])


                try:
                    fit_paramsT, covar, infodict, mesg, ier = optimize.curve_fit(fitfunc, taus,y, p0=p0, full_output= True,
                        bounds=bounds) #, , sigma=ysig

                    sum_resq = np.sum(np.array(infodict['fvec'])**2)
                    if sum_resq < best_fit_resq:
                        fit_params1 = fit_paramsT
                        std_vec1 = np.sqrt(np.diag(covar))
                        best_fit_resq = sum_resq
                        best_with_beat = True if fit_with_beating else False                            
                    
                except:
                    print('Try again')
                    continue

        if best_with_beat:
            fitfunc = lambda x, *p, :  -1*((t2_amp_estimate-p[0])*np.cos(2*np.pi*(p[1]*x)) + p[0]*np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
            # fitfunc = lambda x, *p, :  p[0]*(np.cos(2*np.pi*(p[1]*x)) + np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
            fit_dict = {
                'amplitude1': fit_params1[0],
                'frequency1': fit_params1[1],
                'background': fit_params1[2],
                'T2star':fit_params1[3],
                'frequency2':fit_params1[4],
                'amplitude1_uncertainty': std_vec1[0],
                'frequency1_uncertainty': std_vec1[1],
                'background_uncertainty': std_vec1[2],
                'T2star_uncertainty':std_vec1[3],
                'frequency2_uncertainty':std_vec1[4],
            }
        else:
            fitfunc = lambda x, *p, :  -1*p[0]*np.cos(2*np.pi*(p[1]*x))*np.exp(-x/p[3]) + p[2]
            fit_dict = {
                'amplitude': fit_params1[0],
                'frequency': fit_params1[1],
                'background': fit_params1[2],
                'T2star':fit_params1[3],
                'amplitude_uncertainty': std_vec1[0],
                'frequency_uncertainty': std_vec1[1],
                'background_uncertainty': std_vec1[2],
                'T2star_uncertainty':std_vec1[3],
            }


        color = (0,0,1)
        fig, ax = plt.subplots(1, 1, squeeze=False)
        fig.suptitle(f"Multiplexed Ramseys")
        i_pq = 0
        ax[i_pq, 0].cla()
        ax[i_pq, 0].set_title(resonator)
        
        ax[i_pq, 0].errorbar(4 * np.array(self.taus), y, label=f'T2star',
                                    color='r', ecolor='r', fmt='.', elinewidth=1, capsize=None)
        ax[i_pq, 0].set_ylabel("Excited State Population")
        xvals = np.linspace(taus.min(), taus.max(), 10000)
        fit_y = fitfunc(xvals,*fit_params1)
        ax[i_pq,0].plot(xvals, fit_y , "-", label = 'fit', linewidth=2, color = color)
        ax[i_pq, 0].legend()
        ax[i_pq,0].set_xlabel("Ramsey FID time (ns)")
        # ax[i_pq,0].set_ylim(0,1)
        
        plt.tight_layout() 
        self.fig = fig
        self.fit_dict = fit_dict


if __name__ == "__main__":
    mr = parity_ramsey("q3_ef")
    # while True:
    #     try:
    # for i in range(2):
    mr.run_parity_ramsey()
    print(mr.fit_dict)
        # except Exception as e:
        #     print(e)
        #     continue