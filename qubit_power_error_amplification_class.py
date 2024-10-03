"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
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
    - Update the qubit pulse amplitude (x180_amp) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from multiplexed_configuration import *
from utils import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler
from scipy import optimize
from typing import *

MINIMUM_SLOPE_SCALE = 0.1  # Factor of original slope in peak_turnover function that determines cut-off condition

class Power_error_amplification:
    def __init__(
        self,
        qubit: str,
        parameter_name: str,
    ):
        self.qubit = qubit
        self.resonator = qubit_resonator_correspondence[self.qubit]
        self.parameter_name = parameter_name
        self.amplitude = QUBIT_CONSTANTS[qubit][parameter_name+'amplitude']
        self.amplitude_sweep = np.array([])
        self.x_label = 'Pulse Amplitude [V]'
        self.y_label = 'Population'
        self.is_negative_amplitude = False
        self.roll_number = 3

    def power_rabi_pulse(
        self,
        simulate: bool = False,
        n_avg: int = 25,
        a_min: float = 0.5,
        a_max: float = 1.5,
        n_a: int = 100,
        max_nb_of_pulses: int = 30,
        nb_pulse_step: int = 2,
    ):
        """
        :param simulate: set True to simulate the OPX program
        :param n_avg: number of averages for each step in sweep
        :param a_min: minumum of pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
        :param a_max: maximum of pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
        :param n_a: number of steps in pulse amplitude sweep
        :param max_nb_of_pulses: Maximum number of qubit pulses
        """
        data_handler = DataHandler(root_data_folder="./")
        ###################
        # The QUA program #
        ###################
        
        self.amplitude_sweep = np.linspace(a_min, a_max, n_a)
        # Number of applied Rabi pulses sweep
        nb_of_pulses = np.arange(0, max_nb_of_pulses, nb_pulse_step)  # Always play an odd/even number of pulses to end up in the same state

        power_rabi_error_amplification_data = {
            "qubit": self.qubit,
            "n_avg": n_avg,
            "resonator_LO": RL_CONSTANTS["rl1"]["LO"],
            "readout_amp": RR_CONSTANTS[qubit_resonator_correspondence[self.qubit]]["amplitude"],
            "qubit_LO": MULTIPLEX_DRIVE_CONSTANTS["drive1"]["LO"],
            "qubit_IF": QUBIT_CONSTANTS[self.qubit]["IF"],
            self.parameter_name+"_amp": self.amplitude,
            self.parameter_name+"_len": QUBIT_CONSTANTS[self.qubit][self.parameter_name+"len"],
            "amplitudes": self.amplitude_sweep,
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
                    with for_(*from_array(a, self.amplitude_sweep)):  # QUA for_ loop for sweeping the pulse amplitude
                        # Loop for error amplification (perform many qubit pulses with varying amplitudes)
                        with for_(n2, 0, n2 < n_rabi, n2 + 1):
                            if self.parameter_name == 'pi_':
                                play("x180" * amp(a), self.qubit)
                            elif self.parameter_name == 'pi_half_':
                                play("x90" * amp(a), self.qubit)
                            else:
                                raise ValueError(f'{self.parameter_name} unsupported. Must be "pi_" or "pi_half_')
                        # Align the two elements to measure after playing the qubit pulses.
                        align(self.qubit, self.resonator)
                        # Measure the state of the resonator
                        # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
                        if RR_CONSTANTS[self.resonator]["use_opt_readout"]:
                            measure(
                                f"readout",
                                self.resonator,
                                None,
                                dual_demod.full("opt_cos", "out1", "opt_sin", "out2", I),
                                dual_demod.full("opt_minus_sin", "out1", "opt_cos", "out2", Q),
                            )
                        else:
                            measure(
                                f"readout",
                                self.resonator,
                                None,
                                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                            )
                        # Wait for the qubit to decay to the ground state
                        wait(thermalization_time * u.ns, self.resonator)
                        # Save the 'I' & 'Q' quadratures to their respective streams
                        save(I, I_st)
                        save(Q, Q_st)
                # Save the averaging iteration to get the progress bar
                save(n, n_st)

            with stream_processing():
                # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
                I_st.buffer(len(self.amplitude_sweep)).buffer(len(nb_of_pulses)).average().save("I")
                Q_st.buffer(len(self.amplitude_sweep)).buffer(len(nb_of_pulses)).average().save("Q")
                n_st.save("iteration")

        #####################################
        #  Open Communication with the QOP  #
        #####################################
        qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

        ###########################
        # Run or Simulate Program #
        ###########################

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
                I, Q = u.demod2volts(I, RR_CONSTANTS[self.resonator]["readout_length"]), u.demod2volts(Q, RR_CONSTANTS[self.resonator]["readout_length"])
                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.get_start_time())
                # Plot results
                plt.suptitle("Power Rabi with error amplification")
                plt.subplot(221)
                plt.cla()
                plt.pcolor(self.amplitude_sweep * self.amplitude, nb_of_pulses, I)
                plt.xlabel("Rabi pulse amplitude [V]")
                plt.ylabel("# of Rabi pulses")
                plt.title("I quadrature [V]")
                plt.subplot(222)
                plt.cla()
                plt.pcolor(self.amplitude_sweep * self.amplitude, nb_of_pulses, Q)
                plt.xlabel("Rabi pulse amplitude [V]")
                plt.title("Q quadrature [V]")
                plt.subplot(212)
                plt.cla()
                plt.plot(self.amplitude_sweep * self.amplitude, np.sum(Q, axis=0))
                plt.xlabel("Rabi pulse amplitude [V]")
                plt.ylabel("Sum along the # of Rabi pulses")
                plt.pause(1)
                plt.tight_layout()
            power_rabi_error_amplification_data["I"] = I
            power_rabi_error_amplification_data["Q"] = Q
            power_rabi_error_amplification_data["figure"] = fig
            plt.close()

            fit_dict, fig_fit = self.power_analysis(I, Q)
            power_rabi_error_amplification_data["fit_dict"] = fit_dict
            power_rabi_error_amplification_data["fit_figure"] = fig_fit

            data_folder = data_handler.save_data(
                power_rabi_error_amplification_data, 
                name=f"{self.qubit}_power_{self.parameter_name}_error_amplification"
            )
            plt.close()

            return fit_dict, data_folder


    def power_analysis(self, I, Q):
        """
        Analyze the data from a power rabi sweep measurement.
        Assumes the data swept both the pulse amplitude and the number of pulses.
        Assumes the IQ Blob has been calibrated, so one of I or Q is high contrast data.

        Averages the I or Q data along the power-sweep axis to form 1D data. Finds the datapoint closest to the
        peak location, then creates a crop window around the peak before fitting.

        :param amplitudes: the swept amplitude array
        :param I: I data
        :param Q: Q data

        :return: dictionary of fit parameters and uncertainties, figure of the fit
        """
        # Pick I or Q based on contrast
        I_lineout = np.mean(I, axis=0)
        Q_lineout = np.mean(Q, axis=0)

        if np.abs(I_lineout.max() - I_lineout.min()) > np.abs(Q_lineout.max() - Q_lineout.min()):
            self.lineout = I_lineout
            self.is_negative_amplitude = True
        else:
            self.lineout = Q_lineout
            self.is_negative_amplitude = False

        fig_fit, fit_params, std_vec = self.peak_fit()

        # Turn the fit params and uncertainties into a dict
        fit_dict = {
            'fit_data_name': 'I' if self.is_negative_amplitude else 'Q',
            'original_amplitude': self.amplitude,
            'amplitudes': self.amplitude_sweep,
            'fit_data': self.lineout,
            'fit_values': {
                'peak_amplitude': fit_params[0],
                'center': fit_params[1],
                'width': fit_params[2],
                'background': fit_params[3],
            },
            'fit_uncertainties':{
                'peak_amplitude': std_vec[0],
                'center': std_vec[1],
                'width': std_vec[2],
                'background': std_vec[3],
            },
        }


        return fit_dict, fig_fit


    def peak_fit(
        self,
        fitname='gaussian',
    ):
        """
        Fits a gaussian or lorentzian to 1D data cropped to a window.

        :param x: the x-axis array
        :param y: the y-axis data
        :param x_window_start: minimum x-axis value of the cropped data
        :param x_window_end: max x-axis value of the cropped data
        :param x_label: str for labeling the x axis
        :param y_label: str for labeling the y axis
        :param is_negative_amplitude: set True to indicate the peak has a negative amplitude
        :param fitname: choose the peak type from guassian or lorentzian

        :return: figure of the fit, fit_params array, and fit parameter uncertainty array
        """
        assert fitname in ['gaussian', 'lorentzian'], f"{fitname=} not in  ['gaussian', 'lorentzian']"

        # Grab data just within the window
        window_indices = self.power_peak_turnover()
        x = self.amplitude_sweep
        y = self.lineout
        y = np.array(y[window_indices[0]:window_indices[1]])
        x = np.array(x[window_indices[0]:window_indices[1]])

        if fitname == 'gaussian':
            fitfunc = lambda x, *p, :  p[0]*np.exp(-( ( (x-p[1])/p[2] )**2 )/2) + p[3]
        else:
            fitfunc = lambda x, *p, :  p[0]/( (1+ ((x-p[1])/p[2])**2)  ) + p[3]

        ymax, ymin = y.max(), y.min()
        if self.is_negative_amplitude:
            s = np.flatnonzero(y < (ymax + ymin)/2) # Find data points above middle y value. First and last points used to estimate width
            # Seed paramters: amplitude, center, width, background
            p0 = [-1*(ymax-ymin), x[np.argmin(y)], np.abs(x[s[0]]- x[s[-1]]), ymax]
        else:
            s = np.flatnonzero(y > (ymax + ymin)/2) # Find data points above middle y value. First and last points used to estimate width
            # Seed paramters: amplitude, center, width, background
            p0 = [(ymax-ymin), x[np.argmax(y)], np.abs(x[s[0]]- x[s[-1]]), ymin]


        # Fit
        fit_params, covar = optimize.curve_fit(fitfunc, x,y, p0=p0) #, sigma=wd9e, bounds=([109,-1, 0, 0.5 ],[110,0, 1, 1.5 ]))
        std_vec = np.sqrt(np.diag(covar))

        fig_peak = plt.figure()
        plt.errorbar(x*self.amplitude,y, color=(0, 0, 1), ls='none', #yerr=wd9e,
            fmt='.', elinewidth=1, capsize=None, label='Data')
        xvals = np.linspace(x.min(), x.max(), 500)
        # fit_y = fitfunc(xvals,*p0)  # plot seed fit func for debugging
        fit_y = fitfunc(xvals,*fit_params)
        plt.plot(xvals*self.amplitude, fit_y , "r-", label='Fit', linewidth=5)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        plt.title(f'Location Fit: {round(fit_params[1]*self.amplitude,5)} +- {round(std_vec[1]*self.amplitude,5)}')
        return fig_peak, fit_params, std_vec


    def power_peak_turnover(    
        self,
    ):
        """
        Given data associated with a single peak, find array indices that can crop the data to a reasonable
        region of interest (ROI) centered on the peak.
        This function finds the indices by:
            1) Find the datapoint closest to the peak
            2) Take a number of datapoints (roll_number) stepping away from the peak in the negative direction
            3) Fit the ("rolling average") slope of those datapoints.
            4) Continue stepping away by 1 index, until the rolling average slope either:
                a) changes polarity
                b) becomes significantly smaller in modulus than the original slope
            5) note the index when one of the above conditions is met. This is the lower index of the ROI
            6) Repeat All above steps for the positive direction for the upper index of the ROI.
        
        :param x: the sweep array
        :param y: the data
        :param is_negative_amplitude: True when the peak has a negative amplitude

        :return: an array with two elements, [lower index, upper index]
        """
        i_peak = np.argmin(self.lineout) if self.is_negative_amplitude else np.argmax(self.lineout)
            
        fitfunc = lambda x, *p, :  p[0]*x

        low_high_indices = [0,0]
        for i_low_high in range(2):
            found = False
            i_step = 0

            while not found:

                if i_low_high == 0:
                    first_index = i_peak-(i_step+self.roll_number)
                    second_index = i_peak-i_step
                else:
                    first_index = i_peak+i_step
                    second_index = i_peak+i_step+self.roll_number

                x_crop = self.amplitude_sweep[first_index:second_index]
                y_crop = self.lineout[first_index:second_index]

                p0 = [(y_crop[-1]-y_crop[0])/(x_crop[-1]-x_crop[0])]
                fit_params, _ = optimize.curve_fit(fitfunc, x_crop-x_crop[0], y_crop-y_crop[0], p0=p0)
                new_slope = fit_params[0]

                if i_step == 0:
                    i_step += 1
                    original_slope = new_slope
                    old_slope_polarity_positive = new_slope > 0
                    continue
                
                # If the rolling average slope polarity changes, the peak has ended
                # If the polarities are the same, but the new slope is 10x flatter, the peak has ended
                if (
                    old_slope_polarity_positive != (new_slope > 0)
                    or
                    (old_slope_polarity_positive == (new_slope > 0)) and (np.abs(new_slope) < np.abs(MINIMUM_SLOPE_SCALE*original_slope))
                ):
                    found = True
                    neg_pos = -1 if i_low_high == 0 else 1
                    low_high_indices[i_low_high] = int(np.ceil(neg_pos*(i_step+self.roll_number/2) + i_peak))

                i_step += 1
            
        return low_high_indices
        


if __name__ == "__main__":
    pea = Power_error_amplification(
        qubit = "q3_xy",
        parameter_name = 'pi_half',
    )
    pea.power_rabi_pulse(
        a_min = 0.75,
        a_max = 1.25,
        nb_pulse_step=4,
    )