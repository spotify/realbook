#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
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

import tensorflow as tf


def log_base_b(x: tf.Tensor, base: float) -> tf.Tensor:
    """
    Compute log_b(x)
    Args:
        x : input
        base : log base. E.g. for log10 base=10
    Returns:
        log_base(x)
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


class NormalizedLog(tf.keras.layers.Layer):
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """

    def build(self, input_shape: tf.Tensor) -> None:
        self.squeeze_batch = lambda batch: batch
        rank = input_shape.rank
        if rank == 4:
            assert input_shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
            self.squeeze_batch = lambda batch: tf.squeeze(batch, axis=1)
        else:
            assert rank == 3, f"Only ranks 3 and 4 are supported!. Received rank {rank} for {input_shape}."

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.squeeze_batch(inputs)  # type: ignore
        # convert magnitude to power
        power = tf.math.square(inputs)
        log_power = 10 * log_base_b(power + 1e-10, 10)

        log_power_min = tf.reshape(tf.math.reduce_min(log_power, axis=[1, 2]), [tf.shape(inputs)[0], 1, 1])
        log_power_offset = log_power - log_power_min
        log_power_offset_max = tf.reshape(
            tf.math.reduce_max(log_power_offset, axis=[1, 2]),
            [tf.shape(inputs)[0], 1, 1],
        )
        log_power_normalized = tf.math.divide_no_nan(log_power_offset, log_power_offset_max)

        return tf.reshape(log_power_normalized, tf.shape(inputs))
