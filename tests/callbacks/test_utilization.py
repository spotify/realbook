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
# limitations under the License

import os
import tempfile
import time
from typing import Any

import numpy as np
import tensorflow as tf

from realbook.callbacks.utilization import CpuUtilizationCallback, MemoryUtilizationCallback
from realbook.layers.signal import Spectrogram


class SleeperCallback(tf.keras.callbacks.Callback):
    def __init__(self, duration: float) -> None:
        self.duration = duration

    def on_train_begin(self, logs: Any) -> None:
        time.sleep(self.duration)


TEST_AUDIO = np.linspace(0, 1, num=22050 * 10)


def test_cpu_memory_utilization() -> None:
    fake_data = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices([TEST_AUDIO]),
            tf.data.Dataset.from_tensor_slices([1]),
        )
    ).batch(1)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,)),
            Spectrogram(),
            # Add a final "channel" dimension after the spectrogram
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="binary_crossentropy")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = tf.summary.create_file_writer(tmpdir)
        cpu_callback = CpuUtilizationCallback(
            writer,
            1,
        )

        memory_callback = MemoryUtilizationCallback(
            writer,
            1,
        )

        model.fit(fake_data, callbacks=[cpu_callback, memory_callback, SleeperCallback(5)])

        # A check to make sure we're actually logging something
        # It'd be nice to also make assertions on the cpu and memory percentage, but
        # that's unreliable as it varies from machine to machine.
        cpu_count = 0
        memory_count = 0
        for e in tf.compat.v1.train.summary_iterator(os.path.join(tmpdir, os.listdir(tmpdir)[0])):
            for v in e.summary.value:
                cpu_count += CpuUtilizationCallback.RESOURCE_NAME in v.tag
                memory_count += MemoryUtilizationCallback.RESOURCE_NAME in v.tag
        assert cpu_count > 3 and cpu_count < 10
        assert memory_count > 3 and memory_count < 10
