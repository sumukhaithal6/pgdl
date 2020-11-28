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

# Functions performing various input/output operations for PGDL competition at NeurIPS 2020
# Main contributor: Yiding Jiang, July 2020
# Modified from the ChaLearn AutoML challenge data manager.

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import os
import time
import glob
import json
from model_utils import model_def_to_keras_sequential
import tensorflow as tf
# tf.compat.v1.enable_v2_behavior()


filter_filenames = ['.DS_Store', '__MACOSX']

def name_filter(name):
    for fn in filter_filenames:
        if fn in name:
            return True
    return False


class DataManager:
    """Object for loading models and data."""
    
    def __init__(self, basename, input_dir):
        self.basename = basename
        self.input_dir = input_dir
        self.parent_directory = os.path.join(input_dir, basename)
        self.data_dir = os.path.join(self.parent_directory, 'dataset_1')
        self.model_ids = [mid for mid in sorted(os.listdir(self.parent_directory)) if 'model' in mid and 'json' not in mid]
        self.full_model_paths = [os.path.join(self.parent_directory, mid) for mid in self.model_ids]
        self.num_models = len(self.model_ids)

    def __repr__(self):
        return "DataManager : " + self.basename

    def __str__(self):
        val = "DataManager : " + self.basename + "\ninfo:\n"
        val = val + "number of models: {}".format(len(self.model_ids))
        return val

    def load_model(self, model_id):
        """Loads the model weight and the initial weight, if any."""
        model_directory = os.path.join(self.parent_directory, model_id)
        with open(os.path.join(model_directory, 'config.json'), 'r') as f:
            config = json.load(f)
        model_instance = model_def_to_keras_sequential(config['model_config'])
        model_instance.build([0] + config['input_shape'])
        weights_path = os.path.join(model_directory, 'weights.hdf5')
        initial_weights_path = os.path.join(model_directory, 'weights_init.hdf5')
        if os.path.exists(initial_weights_path):
          try:
            model_instance.load_weights(initial_weights_path)
            model_instance.initial_weights = model_instance.get_weights()
          except ValueError as e:
            print('Error while loading initial weights of {} from {}'.format(model_id, initial_weights_path))
            print(e)
        model_instance.load_weights(weights_path)
        return model_instance
    
    def load_training_data(self):
        """Loads the training data."""
        path_to_shards = glob.glob(os.path.join(self.data_dir, 'train', 'shard_*.tfrecord'))
        dataset = tf.data.TFRecordDataset(path_to_shards)

        def _deserialize_example(serialized_example):
            record = tf.io.parse_single_example(
            serialized_example,
            features={
                'inputs': tf.io.FixedLenFeature([], tf.string),
                'output': tf.io.FixedLenFeature([], tf.string)
            })
            inputs = tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
            output = tf.io.parse_tensor(record['output'], out_type=tf.int32)
            return inputs, output

        return dataset.map(_deserialize_example)
    
