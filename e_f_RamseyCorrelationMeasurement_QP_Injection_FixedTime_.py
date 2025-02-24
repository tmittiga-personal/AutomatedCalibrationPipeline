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

# Assumes 2 MHz detuning. 4 points per oscillation
DEFAULT_TAUS = np.concatenate(
            (np.arange(4, 80_000//4 , 500//4), # 0.1 MHz  #np.arange(4, 200_000//4 , 500//4)
            )
        )

class ef_ramseycorrelation:
    def __init__(
        self,   
        f1,     
        f2,
        f_art = 500_000,
        probe_qubit = "q3_ef",
        evolution_times: np.typing.NDArray = DEFAULT_TAUS,
    ):
        self.QP_wait = 120_000 # clock cycles
        # Convert to Hz from GHz
        self.f1 = f1*1e9
        self.f2 = f2*1e9
        self.f_beat = np.abs(f1-f2) # in GHz
        self.f_art = f_art
        self.mc = create_multiplexed_configuration()
        self.probe_qubit = probe_qubit
        self.state_prep_qubit = self.probe_qubit.replace('ef','xy')
        self.f_artificial = 0 * self.mc.u.kHz
        self.n_avg = 200 #1_500_000 #10_000  # The number of averages
        self.random_phases = np.random.random(self.n_avg)
        self.program_divisions = 1
        self.evolution_times = evolution_times
        
        self.resonator = 'q3_re'
        self.resonatorge = 'q3_rr'
        self.resonatorQP = 'q1_rr'

        # shift IF to address the higher parity state
        upper_f = self.f1 if self.f1 > self.f2 else self.f2
        self.new_IF = self.mc.QUBIT_CONSTANTS[probe_qubit]['IF'] + (self.f_art - upper_f)
        self.tau = 0.5/self.f_beat # in ns
        self.threshold = self.mc.RR_CONSTANTS[self.resonator]['ge_threshold']
        self.thresholdge = self.mc.RR_CONSTANTS[self.resonatorge]['ge_threshold']
        self.pi_length = self.mc.QUBIT_CONSTANTS[probe_qubit]['pi_len']
        self.ge_pi_length = self.mc.QUBIT_CONSTANTS[self.state_prep_qubit]['pi_len']

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
            "evolution_time": self.tau,
            'f_artificial': self.f_artificial,
            "QP_wait (ns)": 4*self.QP_wait
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

        # Automatically select resonators associated with the probed qubits
        assert set([self.probe_qubit]).issubset(self.mc.QUBIT_CONSTANTS.keys()), "All probe qubits must be found in QUBIT_CONSTANTS"

        #########################
        ### Precompile Pulses ###
        #########################
        # generate config pulses to match time sweep
        config_copy = deepcopy(self.mc.config)  # copy the config for safety
        probe_qubit = self.probe_qubit
        resonator = self.resonator

        programs = []

        random_phases = self.random_phases
        evolution_times = self.evolution_times
        
        for i_i in range(self.program_divisions):
            # program_code = f"""
            with program() as multiplex_ramsey: #_{i_i}:
                random_phase = declare(fixed)  # QUA variable for the averaging loop
                evolution = declare(int)
                I_cases = declare(fixed)  # QUA variable for the measured 'I' quadrature in each case
                I_st = declare_stream()  # Stream for the 'I' quadrature in each case

                
                update_frequency(probe_qubit, self.new_IF)
                wait(10, probe_qubit)
                with for_each_(random_phase, random_phases):
                    with for_(*from_array(evolution, evolution_times)):
                        # Strict_timing ensures that the sequence will be played without gaps
                        # with strict_timing_():   

                        ################
                        #### T2star ####
                        ################
                        align()
                        play("cw", self.resonatorQP)  # Amplitude is already set by cw pulse
                        wait(self.QP_wait - (evolution + 2*int(self.tau//4) + 4*self.pi_length + self.ge_pi_length), self.resonatorQP)
                        align()
                        # initialize to e state
                        play("x180", self.state_prep_qubit)
                        align(self.state_prep_qubit, probe_qubit)
                        # perform 1st ramsey
                        play("x90", probe_qubit)
                        wait(int(self.tau//4), probe_qubit)
                        play("x90", probe_qubit)
                        # rotate the qubit frame for the next ramsey. Do it before the wait, just in case
                        frame_rotation_2pi(random_phase, probe_qubit)

                        # Wait for the correlation time
                        wait(evolution, probe_qubit)

                        # second ramsey
                        play("x90", probe_qubit)
                        wait(int(self.tau//4), probe_qubit)
                        play("x90", probe_qubit)
                        # reset frame at the end
                        reset_frame(probe_qubit)
                        align()
                        # measure along ef
                        measure(
                            "readout",
                            resonator,
                            None,
                            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_cases),
                        )
                        wait(self.mc.thermalization_time * self.mc.u.ns, self.resonator)
                        # Save the 'I_e' & 'Q_e' quadratures to their respective streams
                        save(I_cases, I_st)

                with stream_processing():
                    # Save all streamed points for plotting the IQ blobs
                    I_st.buffer(len(evolution_times)).average().save("I")

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


    def run_ef_ramseycorrelation(self):       
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

            # Get results from QUA program
            res_handles = job.result_handles
            # Waits (blocks the Python console) until all results have been acquired
            res_handles.wait_for_all_values()
            I = res_handles.get("I").fetch_all()
            self.program_data_dict[job.id] = {resonator: {}}
            self.program_data_dict[job.id][resonator].update(
                {
                    'I': I,
                }
            )
            job = self.qm.get_running_job() 
        # Extract all data and sort into data_dict
        
        for i_pd, data in enumerate(self.program_data_dict.values()):
            # For python >= 3.7, insertion order of dictionaries is preserved, 
            # so no need to sort; they are already in job-order = tau-order
            if i_pd == 0:
                I = np.array(data[resonator]['I'])
            else:
                I = np.concatenate((I,np.array(data[resonator]['I'])))

        self.data_dict[resonator].update(
            {
                'I': I,
            }
        )

        # Save to file
        self.multiplex_ramsey_data["correlation_tau"] = self.evolution_times
        self.multiplex_ramsey_data["measurement_data"] = self.data_dict
        self.data_folder = self.data_handler.save_data(self.multiplex_ramsey_data, name="e_f_ramseycorrelation_QPFixed")
        print('Done')


if __name__ == "__main__":
    df = pull_latest_calibrated_values(
        qubits=['q3_ef'],
        search_parameter_names=['IF'],
        # all_attempts=True,
    )
    mval = df['miscellaneous'].values[-1]
    f1 = mval['fit_dict']['frequency1']
    f2 = mval['fit_dict']['frequency2']
    mr = ef_ramseycorrelation(
        f1 = f1,
        f2 = f2,
    )
    # while True:
    #     try:
    # for i in range(2):
    mr.run_ef_ramseycorrelation()
    # print(mr.fit_dict)
        # except Exception as e:
        #     print(e)
        #     continue