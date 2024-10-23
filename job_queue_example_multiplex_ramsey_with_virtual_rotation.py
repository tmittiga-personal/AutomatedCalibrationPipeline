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
from multiplexed_configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
import itertools
from copy import deepcopy
from AutomatedCalibrationPipeline.utils import *
import time
from qm import CompilerOptionArguments

class multiplex_ramsey:
    def __init__(
        self,
    ):
        self.probe_qubits = ["q1_xy", "q3_xy", "q5_xy"] #, "q3_xy", "q4_xy", "q5_xy", "q6_xy"]
        self.f_artificial = 0.1 * u.MHz
        self.n_avg = 1000  # The number of averages
        self.program_divisions = 2
        # Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
        # taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus
        self.taus = np.concatenate(
            (np.arange(4, 21240//4 , 1240//2), # 0.1 MHz
            np.arange(4, 21240//4 , 1240//2)+40000//4, 
            np.arange(4, 21240//4 , 1240//2)+80000//4,
            # np.arange(4, 21240//4 , 1240//2)+140000//4,
            # np.arange(4, 21240//4 , 1240//2)+200000//4,
            # np.arange(4, 21240//8 , 1240//2)+250000//4,

            # (np.arange(4, 21240//8 , 1240//4), # 0.2 MHz
            # np.arange(4, 21240//8 , 1240//4)+20000//4, 
            # np.arange(4, 21240//8 , 1240//4)+50000//4, 
            # np.arange(4, 21240//8 , 1240//4)+90000//4,
            # np.arange(4, 21240//8 , 1240//4)+150000//4,

            # (np.arange(4, 2124//8 , 124//4), # 2 MHz
            #  np.arange(4, 2124//8 , 124//4)+10000//4, 
            # np.arange(4, 2124//8 , 124//4)+20000//4, 
            # np.arange(4, 2124//8 , 124//4)+50000//4, 
            # np.arange(4, 2124//8 , 124//4)+80000//4,

            # (np.arange(4, 80//4 , 1),  # 62.5 MHz
            #  np.arange(4, 80//4 , 1)+3000//4,
            #  np.arange(4, 80//4 , 1)+7000//4,
            #  np.arange(4, 80//4 , 1)+10000//4, 
            #  np.arange(4, 80//4 , 1)+13000//4, 
            #  np.arange(4, 80//4 , 1)+17000//4, 
            #  np.arange(4, 80//4 , 1)+20000//4, 
            #  np.arange(4, 80//4 , 1)+25000//4, 
            #  np.arange(4, 80//4 , 1)+30000//4, 
            #  np.arange(4, 80//4 , 1)+40000//4,  
            #  np.arange(4, 80//4 , 1)+50000//4,
            #  np.arange(4, 2124//4 , 124//4)+80000//4,
            )
        )

        self.lenpq = len(self.probe_qubits)  # for plotting

        
        self.qubit_resonator_correspondence = {qu: res for qu, res in zip(QUBIT_CONSTANTS.keys(), RR_CONSTANTS.keys())}
        self.resonators = [self.qubit_resonator_correspondence[key] for key in self.probe_qubits]
        self.thresholds = [RR_CONSTANTS[resonator]["ge_threshold"] for resonator in self.resonators]

        self.multiplex_ramsey_data = {
            "n_avg": self.n_avg,
            "resonators": self.resonators,
            "RR_CONSTANTS": RR_CONSTANTS,
            "RL_CONSTANTS": RL_CONSTANTS,
            "QUBIT_CONSTANTS": QUBIT_CONSTANTS,
            "MULTIPLEX_DRIVE_CONSTANTS": MULTIPLEX_DRIVE_CONSTANTS,
            "probe_qubits": self.probe_qubits,
            "qubit_octave_gain": qubit_octave_gain,
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
        noduplicates = list(dict.fromkeys(self.probe_qubits))
        assert len(noduplicates) == len(self.probe_qubits), "No duplicates permitted in probe_qubits list"
        assert set(self.probe_qubits).issubset(QUBIT_CONSTANTS.keys()), "All probe qubits must be found in QUBIT_CONSTANTS"

        #########################
        ### Precompile Pulses ###
        #########################
        # generate config pulses to match time sweep
        config_copy = deepcopy(config)  # copy the config for safety

        for i_q, probe_qubit in enumerate(self.probe_qubits):
            copy_x90_pulse = deepcopy(config['pulses'][f'x90_pulse_{probe_qubit}'])  # Copy the x180 pulse dict so we don't change the original
            x90_I_wf_copy = deepcopy(config['waveforms'][f"x90_I_wf_{probe_qubit}"])  # Same for waveform. Will be used for both I and Q
            x90_amp = QUBIT_CONSTANTS[probe_qubit]["pi_half_amplitude"]
            pulse_length = QUBIT_CONSTANTS[probe_qubit]["pi_half_len"]
            drag_coef = QUBIT_CONSTANTS[probe_qubit]['drag_coefficient']
            anharmonicity = QUBIT_CONSTANTS[probe_qubit]['anharmonicity']
            AC_stark_detuning = QUBIT_CONSTANTS[probe_qubit]['ac_stark_shift']

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
                I_thresholded_st = declare_stream()  # Stream for the 'I' quadrature in each case
                n_st = declare_stream()  # Stream for the averaging iteration 'n'
                final_phase = declare(fixed, 0)

                with for_(n, 0, n < self.n_avg, n + 1):        
                    for t in self.tau_arrays[i_i]:  
                        # Strict_timing ensures that the sequence will be played without gaps
                        with strict_timing_():           
                            align()
                            for probe_qubit in self.probe_qubits:
                                play("x90", probe_qubit)
                            align()
                            wait(t, self.probe_qubits[0])
                            align()
                            for probe_qubit in self.probe_qubits:
                                play(f'xvar{t}', probe_qubit)
                            align()
                            for i_q, (probe_qubit, resonator) in enumerate(zip(self.probe_qubits, self.resonators)):
                                measure(
                                    "readout",
                                    resonator,
                                    None,
                                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_cases),
                                )
                                # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                                assign(I_thresholded, Util.cond(I_cases>self.thresholds[i_q], 1, 0))
                                save(I_thresholded, I_thresholded_st)
                            # Wait for the qubit to decay to the ground state
                            wait(thermalization_time * u.ns, resonator)
                    # Save the averaging iteration to get the progress bar
                    save(n, n_st)

                with stream_processing():
                    # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
                    # If log sweep, then the swept values will be slightly different from np.logspace because of integer rounding in QUA.
                    # get_equivalent_log_array() is used to get the exact values used in the QUA program.
                    
                    I_thresholded_st.buffer(self.lenpq).buffer(len(self.tau_arrays[i_i])).average().save(f"I_thresholded")
                    n_st.save("iteration") #"""
            # exec(program_code)
            # eval(f'programs.append(multiplex_ramsey_{i_i})')
            programs.append(multiplex_ramsey)
        self.programs = programs
        self.config = config_copy

        
        #####################################
        #  Open Communication with the QOP  #
        #####################################
        qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)
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


    def run_multipled_ramsey(self):       
        # Initialize data dicts
        self.data_dict = {resonator: {} for resonator in self.resonators}
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
            fetch_names = ["iteration", "I_thresholded"]
            # Get results from QUA program
            results = fetching_tool(job, data_list=fetch_names, mode="live")    # Live plotting
            # Live plotting. 
            fig, ax = plt.subplots(self.lenpq, 1, squeeze=False)
            fig.suptitle(f"Multiplexed Ramseys")
            interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
            while results.is_processing():
                # Fetch results
                iteration = job.result_handles.iteration.fetch_all()
                # Progress bar
                progress_counter(iteration, self.n_avg, start_time=results.get_start_time())
                P = job.result_handles.I_thresholded.fetch_all()
                P = P.transpose()
                N = job.result_handles.iteration.fetch_all()
                self.program_data_dict[job.id] = {resonator: {} for resonator in self.resonators}
                for i_pq, (pq, resonator) in enumerate(zip(self.probe_qubits, self.resonators)):
                    ax[i_pq, 0].cla()
                    ax[i_pq, 0].set_title(resonator)

                    cis = wilson_score_interval(P[i_pq], N, z=1)
                    errorbars = errorbars_from_intervals(P[i_pq], cis)
                    
                    ax[i_pq, 0].errorbar(4 * self.tau_arrays[i_job], P[i_pq], yerr=errorbars, label=f'I',
                                                ecolor='b', fmt='.', elinewidth=1, capsize=None)
                    ax[i_pq, 0].set_xlabel('durations (ns)')
                    ax[i_pq, 0].set_ylabel("Excited State Population")
                    ax[i_pq, 0].legend()
                    self.program_data_dict[job.id][resonator].update(
                        {
                            'taus':self.tau_arrays[i_job],
                            'P':P[i_pq],
                            'P_err':errorbars,
                        }
                    )
                plt.pause(2)
                plt.tight_layout()
            job = self.qm.get_running_job() 
        # Close all open plots
        for i_i in range(self.program_divisions):
            plt.close()
        # Extract all data and sort into data_dict
        for resonator in self.resonators:
            for i_pd, data in enumerate(self.program_data_dict.values()):
                # For python >= 3.7, insertion order of dictionaries is preserved, 
                # so no need to sort; they are already in job-order = tau-order
                if i_pd == 0:
                    P = np.array(data[resonator]['P'])
                    P_err = np.array(data[resonator]['P_err'])
                else:
                    P = np.concatenate((P,np.array(data[resonator]['P'])))
                    P_err = np.concatenate((P_err,np.array(data[resonator]['P_err'])), axis = 1)

            self.data_dict[resonator].update(
                {
                    'P':P,
                    'P_err':P_err,
                }
            )
        # Create final plot, in case the previous figure missed some iterations
        fig, ax = plt.subplots(self.lenpq, 1, squeeze=False)
        fig.suptitle(f"Multiplexed Ramseys")
        for i_pq, (pq, resonator) in enumerate(zip(self.probe_qubits, self.resonators)):
            ax[i_pq, 0].cla()
            ax[i_pq, 0].set_title(resonator)
            
            ax[i_pq, 0].errorbar(4 * np.array(self.taus), self.data_dict[resonator]['P'], yerr=self.data_dict[resonator]['P_err'], label=f'I',
                                        ecolor='b', fmt='.', elinewidth=1, capsize=None)
            ax[i_pq, 0].set_xlabel('durations (ns)')
            ax[i_pq, 0].set_ylabel("Excited State Population")
            ax[i_pq, 0].legend()
        plt.tight_layout()

        # Save to file
        self.multiplex_ramsey_data["figure"] = fig
        self.multiplex_ramsey_data["measurement_data"] = self.data_dict
        self.data_handler.save_data(self.multiplex_ramsey_data, name="multiplex_job_queue_ramsey")
        plt.close()

if __name__ == "__main__":
    mr = multiplex_ramsey()
    # while True:
    #     try:
    for i in range(2):
        mr.run_multipled_ramsey()
        # except Exception as e:
        #     print(e)
        #     continue