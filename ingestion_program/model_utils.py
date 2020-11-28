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
# Main contributor: Pierre Foret, July 2020

import tensorflow as tf
import json


class Sequential(tf.keras.Sequential):
  def __call__(self, x, tape=False, *args, **kwargs):
    if tape:
      tape.watch(x)
    return super(Sequential, self).__call__(x, *args, **kwargs)


def wrap_layer(layer_cls, *args, **kwargs):
    """Wraps a layer for computing the jacobian wrt to intermediate layers."""
    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)
    return wrapped_layer(*args, **kwargs)


def model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.

    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.

    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        # layer_cls = wrap_layer(layer_cls)
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return wrap_layer(layer_cls, **_cast_to_integer_if_possible(kwargs))
        # return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return Sequential([parse_layer(l) for l in model_def])


@tf.function()
def get_jacobian(model, inputs):
    """Get jacobians with respect to intermediate layers."""
    with tf.GradientTape(persistent=True) as tape:
        out = model(inputs, tape=tape)
    dct = {}
    for i, l in enumerate(model.layers):
        try:
            dct[i] = tape.batch_jacobian(out, l._last_seen_input)
        except AttributeError:  # no _last_seen_input, layer not wrapped (ex: flatten)
            dct[i] = None
    return dct

