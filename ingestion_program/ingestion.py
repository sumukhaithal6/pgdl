# Copyright 2020 The PGDL Competition organizers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Yiding Jiang, July 2020
# Modified from the Codalab Iris competition ingestion program.

#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers for the PGDL competition at NeurIPS 2020.
# This program also runs on the challenge platform to test your code.
#
# The input directory input_dir (e.g. sample_data/) contains the dataset(s), including:
#   - dataname/dataset_1                 -- the feature names (column headers of data matrix)
#     - train                            -- data shard for the training data of this group of models
#     - test                             -- data shard for the test data of this group of models
#   - dataname/model_*                            -- the individual model
#     - config.json                      -- configuration for building the model
#     - weights.hdf5                     -- weights of the trained model
#   - dataname/model_configs.json                 -- configurations of all models and training/test numerics 
#
# The output directory output_dir (e.g. sample_result_submission/) 
# will receive the predicted values (no subdirectories):
# 	dataname.predict            
#
# The code directory submission_program_dir (e.g. sample_code_submission/) should contain your 
# code submission model.py (an possibly other functions it depends upon).

# =========================== BEGIN OPTIONS ==============================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER MODEL. 
max_time_per_model = 60 * 5

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "../"
default_input_dir = root_dir + "sample_data"
default_output_dir = root_dir + "sample_result_submission"
default_program_dir = root_dir + "ingestion_program"
default_submission_dir = root_dir + "sample_code_submission"

# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Version of the sample code
version = 1 

# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import datetime
import glob
import inspect

the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

import tensorflow as tf
# enable tf 2.0 behavior
tf.compat.v1.enable_v2_behavior()

filter_filenames = [".ds_store", ".DS_Store", "__MACOSX"]

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:	
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
    if verbose: 
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)
    import data_io                       # general purpose input/output functions
    from data_io import vprint           # print only in verbose mode
    from data_manager import DataManager # load/save data and get info about them
    from complexity import complexity # complexity measure

    should_pass_submission_dir = 'program_dir' in inspect.getfullargspec(complexity).args

    if debug_mode >= 4:
      print('File structure')
      data_io.list_files('..')

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")
        
    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = os.listdir(input_dir)
    # change input dir to compensate for the single file unzipping
    if 'input_data' in datanames:
        input_dir = os.path.join(input_dir, 'input_data')
        datanames = os.listdir(input_dir)
    # Overwrite the "natural" order
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = [] # Do not proceed with learning and testing
        
    #### MAIN LOOP OVER DATASETS: 
    overall_time_budget = 0
    time_left_over = 0
    time_exceeded = False
    for basename in datanames: # Loop over datasets
        if basename in filter_filenames:
            continue
        vprint( verbose,  "\n========== Ingestion program version " + str(version) + " ==========\n") 
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")
        
        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()
        
        # ======== Creating a data object with data, informations about it (write a new data manager for loading the models)
        vprint(verbose,  "========= Reading and converting data ==========")
        D = DataManager(basename, input_dir)
        print(D)
        #vprint( verbose,  "[+] Size of uploaded data  %5.2f bytes" % data_io.total_size(D))
        
        # ======== Keeping track of time
        #if debug_mode<1:
        #    time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        #else:
        #    time_budget = max_time
        time_budget = D.num_models * max_time_per_model

        overall_time_budget = overall_time_budget + time_budget
        vprint( verbose,  "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over form previous dataset: time_budget += time_left_over
        vprint( verbose,  "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))

        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue
        
        # ========= Creating a model ========== 
        vprint( verbose,  "======== Creating model ==========")

        training_data = D.load_training_data()
        complexity_value = {}
        for mid in D.model_ids:
            if time_exceeded:
                complexity_value[mid] = 'EXCEEDED'
                continue
            tf.keras.backend.clear_session()
            model = D.load_model(mid)

            if should_pass_submission_dir:
                measure_val = complexity(model, training_data, program_dir=submission_dir)
            else:
                measure_val = complexity(model, training_data)

            try:
                measure_val = float(measure_val)
            except:
                print('Incorrect measure data type!')
                raise TypeError('Measure should be a scalar float or numpy float but got type: {}'.format(type(measure_val)))

            complexity_value[mid] = measure_val
            time_left_over = time_budget - time.time() + start
            if time_left_over <= 0:
                time_exceeded = True

        if verbose:
            print(complexity_value)

        if time_exceeded:
            vprint(verbose, "[+] Time exceeded: time limit is {} but program has run for {}".format(time_budget, time.time() - start))
        else:
            vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        # -------------
        filename_train = basename + '.predict'
        vprint( verbose, "======== Saving results to: " + output_dir + " as " + filename_train)
        data_io.save_json(os.path.join(output_dir, filename_train), complexity_value)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start 
        time_left_over = time_budget - time_spent
        vprint( verbose,  "[+] Time left %5.2f sec" % time_left_over)
        #if time_left_over<=0: break
               
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)

    if time_exceeded:
        print("Exceeding the time budge of {} sec/model ({} seconds total)!".format(max_time_per_model, time_budget))
      # raise TimeoutError("Exceeding the time budge of {} sec/model ({} seconds total).".format(max_time_per_model, time_budget)) 
