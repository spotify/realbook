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

import numpy as np
import tensorflow as tf

from realbook.callbacks.speed import TrainingSpeedCallback
from realbook.layers.signal import Spectrogram

TEST_AUDIO = np.linspace(0, 1, num=22050 * 10)


def test_speed_callback() -> None:
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
        speed_callback = TrainingSpeedCallback(writer, batches_per_epoch=1, examples_per_batch=1)

        model.fit(fake_data, callbacks=[speed_callback])
        writer.flush()

        all_points = 0
        for e in tf.compat.v1.train.summary_iterator(os.path.join(tmpdir, os.listdir(tmpdir)[0])):
            for v in e.summary.value:
                all_points += "Training Speed/" in v.tag
                value = tf.io.parse_tensor(v.tensor.SerializeToString(), tf.float32).numpy()
                if "Batches Per Epoch" in v.tag:
                    assert value == 1
                if "Examples Per Batch" in v.tag:
                    assert value == 1
        assert all_points == 7
