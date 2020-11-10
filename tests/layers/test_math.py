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
import numpy as np
import tensorflow as tf

from realbook.layers import math


def test_normalized_log_basic() -> None:
    norm_log = tf.keras.Sequential([tf.keras.layers.InputLayer((1, 10, 10)), math.NormalizedLog()])
    norm_log.compile()
    x = tf.random.normal([1, 1, 10, 10], mean=5, stddev=10)
    norm_log = norm_log(x).numpy()
    assert np.min(norm_log) >= 0
    assert np.max(norm_log) <= 1


def test_normalized_log_already_scaled() -> None:
    x = tf.random.normal([1, 10, 10], mean=0.1, stddev=0.1)
    norm_log = math.NormalizedLog()(x).numpy()
    assert np.min(norm_log) >= 0
    assert np.max(norm_log) <= 1


def test_normalized_log_zero_centered() -> None:
    x = tf.random.normal([1, 10, 10], mean=0, stddev=1)
    norm_log = math.NormalizedLog()(x).numpy()
    assert np.min(norm_log) >= 0
    assert np.max(norm_log) <= 1


def test_normalized_log_negative() -> None:
    x = tf.random.normal([1, 10, 10], mean=-5, stddev=1)
    norm_log = math.NormalizedLog()(x).numpy()
    assert np.min(norm_log) >= 0
    assert np.max(norm_log) <= 1


def test_normalized_log_batch() -> None:
    """Test that this is applied channel wise and not
    across the whole batch
    """
    gram1 = 0.5 * tf.ones([1, 2, 2])
    norm1 = math.NormalizedLog()(gram1).numpy()
    gram2 = tf.ones([1, 2, 2])
    norm2 = math.NormalizedLog()(gram2).numpy()
    gram3 = 0.25 * tf.ones([1, 2, 2])
    norm3 = math.NormalizedLog()(gram3).numpy()
    gram4 = tf.zeros([1, 2, 2])
    norm4 = math.NormalizedLog()(gram4).numpy()
    gram_batch = tf.concat([gram1, gram2, gram3, gram4], axis=0)
    norm_batch = math.NormalizedLog()(gram_batch).numpy()

    assert np.allclose(norm_batch[0, :, :], norm1[0, :, :], rtol=0)
    assert np.allclose(norm_batch[1, :, :], norm2[0, :, :], rtol=0)
    assert np.allclose(norm_batch[2, :, :], norm3[0, :, :], rtol=0)
    assert np.allclose(norm_batch[3, :, :], norm4[0, :, :], rtol=0)
