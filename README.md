# Quantum Machines Automated Qubit Calibration Pipeline

## Introduction
I noticed that the Laboratory for Physical Sciences, University of Maryland, lacked an automated calibration system for qubits. 
Since the facility recently purchased Quantum Machines control systems, I wrote a basic automation codebase based on their systems.
I wrote the calibration pipeline itself fairly generally, so those not using Quantum Machines could adapt it if needed.
Note, I wrote this pipeline in about a week: and have been slowly cleaning and optimizing it as time permits, so while master remains bug-free, there are many TODOs.

For those unfamiliar with the terms, a "pipeline" is a chain of modules where the output of one module is the input of the next.
In the case of a calibration pipeline, the calibration parameters are updated and used in each subsequent module.
A "node" is an object controlling the full calibration measurement and data processing that produces one or more calibration parameters.
Multiple nodes make up the pipeline chain, each node producing a subset of the entire set of calibration parameters.
The automation involved is the timing of running each node and logic governing how nodes relate to or trigger each other.

The pipeline, as is, is capable of recalibrating qubit pulses (pi and pi/2) and readout (resonator frequency, pulse duration, pulse amplitude).
It is, overall, not optimized.
Some modules, notably `readout_amplitude_binary_search.py` and `qubit_power_error_amplification_class.py`, are fairly robust and optimal calibrations.
Others, such as `ramsey_w_virtual_rotation.py` and `readout_frequency_optimization.py` are slow, but robust, as the apparatus currently needs.


### Job Queue
A job queue is a queue that manages which parts of an experiment run at what times.
For example, if multiple users want to submit experiments to the same Quantum Machines, the job queue would decide the order in which the experiments (or parts of an experiment) run.
This is particularly important for a fully automated calibration pipeline because it would permit running the pipeline constantly in the background.
Then, manually submitted experiments (eg experiments during business hours for research purposes) could interleave seamlessly with calibrations occurring at regular intervals.

Having the job queue manage ALL experiments submitted to the Quantum Machines controlling our one apparatus is a low priority for our academic research and calibration needs, which is why it is not ubiquitous in all the experiments of this pipeline.
Still, I have found the job queue necessary for splitting up large experiments into smaller memory chunks that the Quantum Machines can handle.
Since no one at LPS has used the Quantum Machines job queue, I have included an example of using the Quantum Machines job queue: `job_queue_example_multiplex_ramsey_with_virtual_rotation.py`.


## Instructions
1. Clone this repo
2. Create a dataframe pkl database file with the appropriate columns and assign the path and name to the `DATAFRAME_FILE` global variable, currently located in `calibration_nodes.py`. The `dataframe_database_creation.py` script does this for the database used by this repo.
3. Follow the examples in the `calibration_nodes.py` file for writing new Node classes, or use the ones provided.
   - Define a new child class that inherits the `Node` parent class.
   - In the new class, define the `calibration_measurement` method to call the appropriate experiment, and potentially the `success_condition` method if a different method of determining success is needed.
   - Modify child class further, as needed.
4. Follow the examples in the `Run_calibration.py` file for running a basic pipeline without any dependency between nodes.
   - Call node instances with desired timing and calibration parameter names. `calibration_parameter_name` can be a string, if only a single parameter is calibrated, or a list of strings, if multiple parameters are calibrated.
   - Call the `calibrate` method for each node in the desired order in the loop. This method returns a bool indicating the success/failure of the calibration attempt, which can be used for logical dependency between nodes.
   - The first time a node's `calibrate` method is called, set `initialize=True` to avoid complications related to that calibration parameter not being in the database.
5. Run the `Run_calibration.py` script to start the calibration pipeline.

## TODOs
- Implement parent-child relationships between nodes to give the user some dependency structure. Triggering a child node would force a parent node to run first. Then, only if the parent succeeds, would the child node run.
- Implement retry_time, which controls how long a node waits before retrying after failure.
- Add Quantum Machines job queue to all pipeline experiments so the pipeline can run in the background at all times.
