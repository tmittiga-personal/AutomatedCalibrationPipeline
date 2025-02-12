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

class ef_ramseyspinlock:
    def __init__(
        self,   
        f1,     
        f2,
        f_art = 500_000,
        probe_qubit = "q3_ef",
    ):
        # Convert to Hz from GHz
        self.f1 = f1*1e9
        self.f2 = f2*1e9
        self.f_beat = np.abs(f1-f2)
        self.f_art = f_art
        self.mc = create_multiplexed_configuration()
        self.probe_qubit = probe_qubit
        self.state_prep_qubit = self.probe_qubit.replace('ef','xy')
        self.f_artificial = 0 * self.mc.u.kHz
        self.n_avg = 175  # The number of averages
        self.program_divisions = 1
        # Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
        self.t_risefall = 50  # rise/fall time (in ns) of Spin Lock Pulse
        self.taus = np.concatenate(
            (np.arange(2*self.t_risefall//4, 80_000//4 , 4_000//4), # 0.1 MHz  #np.arange(4, 200_000//4 , 500//4)
            )
        )
        
        self.resonator = 'q3_re'

        # shift IF to address the higher parity state
        upper_f = self.f1 if self.f1 > self.f2 else self.f2
        self.new_IF = self.mc.QUBIT_CONSTANTS[probe_qubit]['IF'] + (self.f_art - upper_f)

        self.multiplex_ramsey_data = {
            "n_avg": self.n_avg,
            "resonator": self.resonator,
            "RR_CONSTANTS": self.mc.RR_CONSTANTS,
            "RL_CONSTANTS": self.mc.RL_CONSTANTS,
            "QUBIT_CONSTANTS": self.mc.QUBIT_CONSTANTS,
            "new_IF": self.new_IF,
            "frequency1":self.f1,
            "frequency2":self.f2,
            'previous_f_artificial': self.f_art,
            "MULTIPLEX_DRIVE_CONSTANTS": self.mc.MULTIPLEX_DRIVE_CONSTANTS,
            "probe_qubit": self.probe_qubit,
            "qubit_octave_gain": self.mc.qubit_octave_gain,
            "taus": self.taus,
            'f_artificial': self.f_artificial,
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
        copy_x90_pulse = deepcopy(self.mc.config['pulses'][f'x180_pulse_{probe_qubit}'])  # Copy the x180 pulse dict so we don't change the original
        x90_I_wf_copy = deepcopy(self.mc.config['waveforms'][f"x180_I_wf_{probe_qubit}"])  # Same for waveform. Will be used for both I and Q
        x180_amp = self.mc.QUBIT_CONSTANTS[probe_qubit]["pi_amplitude"]
        pulse_length = self.mc.QUBIT_CONSTANTS[probe_qubit]["pi_len"]
        original_drive_freq = 1/(2*pulse_length)
        # scale down the original amplitude linearly to match the dispersion shift
        new_amp = self.f_beat/original_drive_freq*x180_amp
        drag_coef = self.mc.QUBIT_CONSTANTS[probe_qubit]['drag_coefficient']
        anharmonicity = self.mc.QUBIT_CONSTANTS[probe_qubit]['anharmonicity']
        AC_stark_detuning = self.mc.QUBIT_CONSTANTS[probe_qubit]['ac_stark_shift']


        for t in self.taus:
            # Calculate new waveforms for I and Q
            I_wf0, Q_wf0 = np.array(
                flattop_gaussian_risefall_waveforms(
                    amplitude = new_amp, 
                    flatlength = round(4*t) - round(2*self.t_risefall), 
                    risefalllength = self.t_risefall,
                    sigma = self.t_risefall/5,
                )
            )  # create a pulse with only I_wf

            # Rotate phase of pulse to y-axis
            I_wf = I_wf0*np.cos(2*np.pi*1/4) - Q_wf0*np.sin(2*np.pi*1/4)
            Q_wf = I_wf0*np.sin(2*np.pi*1/4) + Q_wf0*np.cos(2*np.pi*1/4)

            config_copy['elements'][probe_qubit]['operations'][f'xvar{t}'] = f'xvar{t}_pulse_{probe_qubit}'

            # Assign modified waveform to I
            x90_I_wf_copy['samples'] = deepcopy(I_wf)
            config_copy['waveforms'][f'xvar{t}_I_wf_{probe_qubit}'] = deepcopy(x90_I_wf_copy)
            # Assign modified waveform to Q
            x90_I_wf_copy['samples'] = deepcopy(Q_wf)
            config_copy['waveforms'][f'xvar{t}_Q_wf_{probe_qubit}'] = deepcopy(x90_I_wf_copy)

            # Create new pulse
            copy_x90_pulse['length'] = round(4*t)
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
                I_st = declare_stream()  # Stream for the 'I' quadrature in each case
                Q_st = declare_stream()  # Stream for the 'I' quadrature in each case
                n_st = declare_stream()  # Stream for the averaging iteration 'n'
                final_phase = declare(fixed, 0)

                with for_(n, 0, n < self.n_avg, n + 1):        
                    for t in self.tau_arrays[i_i]:  
                            assign(final_phase, Cast.mul_fixed_by_int(self.f_artificial * 1e-9, 4 * t))
                            # Strict_timing ensures that the sequence will be played without gaps
                            with strict_timing_():   
                                update_frequency(probe_qubit, self.new_IF)
                                wait(t, probe_qubit)

                                ################
                                #### T2star ####
                                ################
                                align()
                                play("x180", self.state_prep_qubit)
                                align(self.state_prep_qubit, probe_qubit)
                                play("x90", probe_qubit)
                                wait(t, probe_qubit)
                                # play(f'xvar{int(t)}', probe_qubit)
                                # frame_rotation_2pi(final_phase, probe_qubit)
                                play("x90", probe_qubit)
                                # reset_frame(probe_qubit)
                                align()
                                measure(
                                    "readout",
                                    resonator,
                                    None,
                                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                                )
                                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                                save(I_cases, I_st)
                                save(Q_cases, Q_st)
                                # Wait for the qubit to decay to the ground state
                                wait(self.mc.thermalization_time * self.mc.u.ns, resonator)

                                ####################
                                #### Background ####
                                ####################
                                align()
                                play("x180", self.state_prep_qubit)
                                align(self.state_prep_qubit, probe_qubit)
                                play("x90", probe_qubit)
                                wait(t, probe_qubit)
                                align()
                                measure(
                                    "readout",
                                    resonator,
                                    None,
                                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                                )
                                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                                save(I_cases, I_st)
                                save(Q_cases, Q_st)
                                # Wait for the qubit to decay to the ground state
                                wait(self.mc.thermalization_time * self.mc.u.ns, resonator)

                                ###################
                                #### Spin Lock ####
                                ###################
                                align()
                                play("x180", self.state_prep_qubit)
                                align(self.state_prep_qubit, probe_qubit)
                                play("x90", probe_qubit)
                                # wait(t, probe_qubit)
                                play(f'xvar{int(t)}', probe_qubit)
                                # frame_rotation_2pi(final_phase, probe_qubit)
                                play("x90", probe_qubit)
                                # reset_frame(probe_qubit)
                                align()
                                measure(
                                    "readout",
                                    resonator,
                                    None,
                                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                                )
                                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
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
                    
                    I_st.buffer(3).buffer(len(self.tau_arrays[i_i])).average().save(f"I")
                    Q_st.buffer(3).buffer(len(self.tau_arrays[i_i])).average().save(f"Q")
                    n_st.save("iteration") #"""
            # exec(program_code)
            # eval(f'programs.append(multiplex_ramsey_{i_i})')
            programs.append(multiplex_ramsey)
        self.programs = programs
        self.config = config_copy
        
        #####################################
        #  Open Communication with the QOP  #
        #####################################
        qmm = QuantumMachinesManager(
            host=self.mc.qop_ip, 
            port=self.mc.qop_port, 
            cluster_name=self.mc.cluster_name, 
            octave=self.mc.octave_config
        )
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


    def run_ef_ramseyspinlock(self):       
        # Initialize data dicts
        resonator = self.resonator
        self.data_dict = {resonator: {}}
        self.program_data_dict = {}

        # Send the QUA program to the OPX, which compiles and executes it
        pending_jobs = []
        for i_i in range(self.program_divisions):
            pending_jobs.append(self.qm.queue.add_compiled(self.precompiled_programs[i_i]))
        ###############################
        ### Wait for Queue to Empty ###
        ###############################
        i_job = 0
        # Get first job sent to QM, which is probably running by now. If not, wait for it to run
        job = None
        while job is None:
            job = self.qm.get_running_job() 
            if job is None:
                time.sleep(1)
        previous_job_id = job.id
        # While queue has pending jobs or there is a job currently running
        while self.qm.queue.count > 0 or job is not None: 
            
            if job.id != previous_job_id:
                i_job +=1
                previous_job_id = job.id
            status = job.status
            if status != "running":
                time.sleep(1)
                continue
            
            # If the job is running, stream results
            print(job.execution_report())
            # Get results from QUA program
            fetch_names = ["iteration", "I", "Q"]
            # Get results from QUA program
            results = fetching_tool(job, data_list=fetch_names, mode="live")    # Live plotting
            # Live plotting. 
            fig, ax = plt.subplots(2, 1, squeeze=False)
            fig.suptitle(f"Multiplexed Ramseys")
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
            while results.is_processing():
                # Fetch results
                iteration = job.result_handles.iteration.fetch_all()
                # Progress bar
                progress_counter(iteration, self.n_avg, start_time=results.get_start_time())
                I = job.result_handles.I.fetch_all()
                I = I.transpose()
                Q = job.result_handles.Q.fetch_all()
                Q = Q.transpose()
                # S = u.demod2volts(I + 1j * Q, RR_CONSTANTS[resonator]["readout_length"])
                # R = np.abs(S)  # Amplitude
                # phase = np.angle(S)  # Phase
                self.program_data_dict[job.id] = {resonator: {}}
                i_pq = 0  # TODO: remove
                ax[2*i_pq, 0].cla()
                ax[2*i_pq, 0].set_title(resonator)
                
                ax[2*i_pq, 0].errorbar(4 * self.tau_arrays[i_job], I[1], label=f'Background',
                                            ecolor='r', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq, 0].errorbar(4 * self.tau_arrays[i_job], I[0], label=f'I_T2*',
                                            ecolor='b', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq, 0].errorbar(4 * self.tau_arrays[i_job], I[2], label=f'I_SpinLock*',
                                            ecolor='k', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq, 0].set_xlabel('durations (ns)')
                ax[2*i_pq, 0].set_ylabel("I Voltage")
                ax[2*i_pq, 0].legend()

                ax[2*i_pq+1, 0].cla()
                ax[2*i_pq+1, 0].set_title(resonator)
                
                ax[2*i_pq+1, 0].errorbar(4 * self.tau_arrays[i_job], Q[1], label=f'Background',
                                            ecolor='r', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq+1, 0].errorbar(4 * self.tau_arrays[i_job], Q[0], label=f'Q_T2*',
                                            ecolor='b', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq+1, 0].errorbar(4 * self.tau_arrays[i_job], Q[2], label=f'Q_SpinLock*',
                                            ecolor='b', fmt='.', elinewidth=1, capsize=None)
                ax[2*i_pq+1, 0].set_xlabel('durations (ns)')
                ax[2*i_pq+1, 0].set_ylabel("Q Voltage")
                ax[2*i_pq+1, 0].legend()
                self.program_data_dict[job.id][resonator].update(
                    {
                        'taus':self.tau_arrays[i_job],
                        'I_T2star':I[0],
                        'Q_T2star':Q[0],
                        'I_B':I[1],
                        'Q_B':Q[1],
                        'I_SpinLock':I[2],
                        'Q_SpinLock':Q[2],
                    }
                )
                plt.pause(2)
                plt.tight_layout()
            job = self.qm.get_running_job() 
        # Close all open plots
        for i_i in range(self.program_divisions):
            plt.close()
        # Extract all data and sort into data_dict
        
        for i_pd, data in enumerate(self.program_data_dict.values()):
            # For python >= 3.7, insertion order of dictionaries is preserved, 
            # so no need to sort; they are already in job-order = tau-order
            if i_pd == 0:
                I_T2star = np.array(data[resonator]['I_T2star'])
                Q_T2star = np.array(data[resonator]['Q_T2star'])
                I_B = np.array(data[resonator]['I_B'])
                Q_B = np.array(data[resonator]['Q_B'])
                I_SpinLock = np.array(data[resonator]['I_SpinLock'])
                Q_SpinLock = np.array(data[resonator]['Q_SpinLock'])
            else:
                I_T2star = np.concatenate((I_T2star,np.array(data[resonator]['I_T2star'])))
                Q_T2star = np.concatenate((Q_T2star,np.array(data[resonator]['Q_T2star'])))
                I_B = np.concatenate((I_B,np.array(data[resonator]['I_B'])))
                Q_B = np.concatenate((Q_B,np.array(data[resonator]['Q_B'])))
                I_SpinLock = np.concatenate((I_SpinLock,np.array(data[resonator]['I_SpinLock'])))
                Q_SpinLock = np.concatenate((Q_SpinLock,np.array(data[resonator]['Q_SpinLock'])))

        self.data_dict[resonator].update(
            {
                'I_T2star': I_T2star,
                'Q_T2star': Q_T2star,
                'I_B': I_B,
                'Q_B': Q_B,
                'I_SpinLock': I_SpinLock,
                'Q_SpinLock': Q_SpinLock,
            }
        )
        # Create final plot, in case the previous figure missed some iterations
        fig, ax = plt.subplots(2, 1, squeeze=False)
        fig.suptitle(f"Multiplexed Ramseys")
        i_pq = 0

        ax[2*i_pq, 0].cla()
        ax[2*i_pq, 0].set_title(resonator)            
        ax[2*i_pq, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['I_T2star'], label=f'I_T2*',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['I_B'], label=f'Background',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['I_SpinLock'], label=f'SpinLock',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq, 0].set_xlabel('durations (ns)')
        ax[2*i_pq, 0].set_ylabel("I Voltage")
        ax[2*i_pq, 0].legend()

        ax[2*i_pq+1, 0].cla()
        ax[2*i_pq+1, 0].set_title(resonator)            
        ax[2*i_pq+1, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['Q_T2star'], label=f'Q_T2*',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq+1, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['Q_B'], label=f'Background',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq+1, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['Q_SpinLock'], label=f'SpinLock',
                                    ecolor='b', fmt='.', elinewidth=1, capsize=None)
        ax[2*i_pq+1, 0].set_xlabel('durations (ns)')
        ax[2*i_pq+1, 0].set_ylabel("Q Voltage")
        ax[2*i_pq+1, 0].legend()
        plt.tight_layout()
        
        plt.close()

        # self.fit_parity_beat()

        # Save to file
        self.multiplex_ramsey_data["figure"] = fig
        # self.multiplex_ramsey_data["fit_figure"] = self.fit_fig
        self.multiplex_ramsey_data["measurement_data"] = self.data_dict
        self.data_folder = self.data_handler.save_data(self.multiplex_ramsey_data, name="e_f_ramseyspinlock")



    def fit_parity_beat(self):
        resonator = self.resonator
        taus = np.array(self.taus)*4
        
        It2 = self.data_dict[resonator]['I_T2star']
        IB = self.data_dict[resonator]['I_B']
        y = It2 - IB

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

        # Remove any freqs < 1/2 f_art and > 1.5*f_art
        f_art = self.f_artificial*1e-9 # convert to GHz
        lower_freq_ind = np.argmax(fft_freq > f_art/2)
        upper_freq_ind = np.argmax(fft_freq > 1.5*f_art)
        strengths = strengths[lower_freq_ind:upper_freq_ind]
        fft_freq = fft_freq[lower_freq_ind:upper_freq_ind]

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
                    fitfunc = lambda x, *p, :  ((t2_amp_estimate-p[0])*np.cos(2*np.pi*(p[1]*x)) + p[0]*np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
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
            fitfunc = lambda x, *p, :  ((t2_amp_estimate-p[0])*np.cos(2*np.pi*(p[1]*x)) + p[0]*np.cos(2*np.pi*p[4]*x))*(np.exp(-x/p[3])) + p[2]
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
            fitfunc = lambda x, *p, :  p[0]*np.cos(2*np.pi*(p[1]*x))*np.exp(-x/p[3]) + p[2]
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
        plt.close()
        self.fit_fig = fig
        self.fit_dict = fit_dict

if __name__ == "__main__":
    mr = ef_ramseyspinlock(
        f1 = 0.0005113771197281403,
        f2 = 0.0004876083908931942,
    )
    # while True:
    #     try:
    # for i in range(2):
    mr.run_ef_ramseyspinlock()
    # print(mr.fit_dict)
        # except Exception as e:
        #     print(e)
        #     continue