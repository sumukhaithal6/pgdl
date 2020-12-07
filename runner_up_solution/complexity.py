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

# Utilities for loading models for PGDL competition at NeurIPS 2020
# Main contributor: Yiding Jiang, July 2020

# This complexity compute a specific notion of sharpness of a function.
# This is the runner up solution to the PGDL Competition 
# Solution contributors: Sumukh Aithal, Dhruva Kashyap

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image

def complexity(model, dataset):

    score = 0.0

    def random_erase_np_v2(images , probability = 1, sl = 0.02, sh = 0.4, r1 = 0.3):
        images = images.numpy()
        res = []
        for img in images:
            height = img.shape[0]
            width = img.shape[1]
            channel = img.shape[2]
            area = width * height

            erase_area_low_bound = np.round( np.sqrt(sl * area * r1) ).astype(np.int)
            erase_area_up_bound = np.round( np.sqrt((sh * area) / r1) ).astype(np.int)
            if erase_area_up_bound < height:
                h_upper_bound = erase_area_up_bound
            else:
                h_upper_bound = height
            if erase_area_up_bound < width:
                w_upper_bound = erase_area_up_bound
            else:
                w_upper_bound = width

            h = np.random.randint(erase_area_low_bound, h_upper_bound)
            w = np.random.randint(erase_area_low_bound, w_upper_bound)

            x1 = np.random.randint(0, height+1 - h)
            y1 = np.random.randint(0, width+1 - w)

            x1 = np.random.randint(0, height - h)
            y1 = np.random.randint(0, width - w)
            # img[x1:x1+h, y1:y1+w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.uint8)
            img[x1:x1+h, y1:y1+w, :] = np.zeros(shape=(h, w, channel)).astype(np.uint8)

            res.append(img)
        
        return tf.convert_to_tensor(res,dtype=tf.float32)
    
    @tf.function
    def predict(x):
        logits = model(x)
        return logits

    batch_size = 64
    MAX_INDEX = 200
    grayscale = False
    for index, (x, y) in enumerate(dataset.batch(batch_size)):

        logits_orig = tf.nn.softmax(predict(x), axis=1)
        
        if(tf.shape(x)[-1]==1):
            grayscale = True
        
        x1 = tf.image.flip_left_right(x)
        
        if grayscale:
            x2 = tf.image.rgb_to_grayscale(tf.image.random_saturation(tf.image.grayscale_to_rgb(x), 0.6, 1.6))
        else:
            x2 = tf.image.random_saturation(x,0.6,1.6)

        x3 = tf.image.central_crop(x,0.9)
        x3 = tf.image.resize_with_pad(x3,tf.shape(x)[1],tf.shape(x)[2])

        x4 = tf.image.sobel_edges(x)
        x4 = x4[...,1]/4+0.5
        # x4 = tf.image.flip_up_down(x)
        x5 = tf.image.adjust_brightness(x,0.5)
        
        if grayscale:
            x6 = tf.image.rgb_to_grayscale(tf.image.random_saturation(tf.image.grayscale_to_rgb(x1), 0.6, 1.6))
        else:
            x6 = tf.image.random_saturation(x1,0.6,1.6)

        x7 = random_erase_np_v2(x)
        # gnoise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.5, dtype=tf.float32)
        # x = tf.image.central_crop(x,0.95)
        # x = tf.add(x,gnoise)
        # x = tf.image.random_saturation(x, 0.6, 1.6)
        # x = tf.image.resize_with_pad(x,32,32)
        
        logits_1 = tf.nn.softmax(predict(x1), axis=1)
        logits_2 = tf.nn.softmax(predict(x2), axis=1)
        logits_3 = tf.nn.softmax(predict(x3), axis=1)
        logits_4 = tf.nn.softmax(predict(x4), axis=1)
        logits_5 = tf.nn.softmax(predict(x5), axis=1)
        logits_6 = tf.nn.softmax(predict(x6), axis=1)
        logits_7 = tf.nn.softmax(predict(x7), axis=1)


        prob_1 = tf.reduce_max(logits_1, axis=-1)
        pred_1 = tf.cast(tf.argmax(logits_1, axis=-1), tf.int32)

        prob_2 = tf.reduce_max(logits_2, axis=-1)
        pred_2 = tf.cast(tf.argmax(logits_2, axis=-1), tf.int32)

        prob_3 = tf.reduce_max(logits_3, axis=-1)
        pred_3 = tf.cast(tf.argmax(logits_3, axis=-1), tf.int32) 

        prob_4 = tf.reduce_max(logits_4, axis=-1)
        pred_4 = tf.cast(tf.argmax(logits_4, axis=-1), tf.int32)

        prob_5 = tf.reduce_max(logits_5, axis=-1)
        pred_5 = tf.cast(tf.argmax(logits_5, axis=-1), tf.int32)

        prob_6 = tf.reduce_max(logits_6, axis=-1)
        pred_6 = tf.cast(tf.argmax(logits_6, axis=-1), tf.int32)

        prob_7 = tf.reduce_max(logits_7, axis=-1)
        pred_7 = tf.cast(tf.argmax(logits_7, axis=-1), tf.int32)
        # # score += tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_1)
        prob_orig = tf.reduce_max(logits_orig, axis=-1)
        pred_orig = tf.cast(tf.argmax(logits_orig, axis=-1), tf.int32) 

        for idx in (range(len(pred_orig))):
            if pred_orig[idx] == pred_1[idx]:
                score += (tf.abs(prob_orig[idx]-prob_1[idx]))
            else:
                score += 6
            if pred_orig[idx] == pred_2[idx]:
                score += (tf.abs(prob_orig[idx]-prob_2[idx]))
            else:
                score += 1
            if pred_orig[idx] == pred_3[idx]:
                score += (tf.abs(prob_orig[idx]-prob_3[idx]))
            else:
                score += 2
            if pred_orig[idx] == pred_4[idx]:
                score += (tf.abs(prob_orig[idx]-prob_4[idx]))
            else:
                score += 3
            if pred_orig[idx] == pred_5[idx]:
                score += (tf.abs(prob_orig[idx]-prob_5[idx]))
            else:
                score += 1
            if pred_orig[idx] == pred_6[idx]:
                score += (tf.abs(prob_orig[idx]-prob_6[idx]))
            else:
                score += 12
            if pred_orig[idx] == pred_7[idx]:
                score += (tf.abs(prob_orig[idx]-prob_7[idx]))
            else:
                score += 2
        if index == MAX_INDEX:
            break
    score = score.numpy()
    return -score

