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
import torch
import numpy as np
import pytest
import librosa
import librosa.display

from typing import List, Tuple, Union

from realbook.layers import nnaudio as our_nnaudio
from nnAudio.Spectrogram import CQT2010v2

TEST_SAMPLE_RATE = 22050


# Test using this model directly, as well as wrapping it in a Lambda layer.
def get_parameterized_model_variants(
    match_torch_exactly_values: Tuple[bool, bool] = (True, False)
) -> List[tf.keras.layers.Layer]:
    possible_models = [
        our_nnaudio.CQT(match_torch_exactly=v, trainable=trainable)
        for v in match_torch_exactly_values
        for trainable in (True, False)
    ]

    return [
        item
        for models in [
            [tf.keras.Sequential([tf.keras.layers.InputLayer((TEST_SAMPLE_RATE,)), model])]
            + (
                [
                    tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer((TEST_SAMPLE_RATE,)),
                            tf.keras.layers.Lambda(lambda x: model(x)),
                        ]
                    )
                ]
                # Using a layer with trainable weights inside a Lambda layer isn't supported.
                if not model.trainable
                else []
            )
            for model in possible_models
        ]
        for item in models
    ]


@pytest.mark.parametrize(
    "match_torch_exactly,threshold,trainable",
    (
        (True, 0.0001, False),
        (False, 4, False),
        (True, 0.0001, True),
        (False, 4, True),
    ),
)
def test_cqt(match_torch_exactly: bool, threshold: float, trainable: bool) -> None:
    # Ensure that the output of our CQT matches nnAudio's
    signal = librosa.chirp(fmin=32.70, fmax=22050, length=220500, linear=True)
    expected = np.transpose(CQT2010v2(verbose=False)(torch.tensor(signal, dtype=torch.float)).numpy(), (0, 2, 1))
    actual = our_nnaudio.CQT(match_torch_exactly=match_torch_exactly, trainable=trainable)(signal).numpy()

    assert expected.shape == actual.shape
    assert np.allclose(expected, actual, rtol=threshold, atol=threshold)


def build_layer(
    layer: tf.keras.layers.Layer, input_shape: Union[Tuple[int], Tuple[int, int]] = (1, TEST_SAMPLE_RATE)
) -> tf.keras.layers.Layer:
    layer.build(input_shape)
    return layer


def test_cqt_trainable_weights() -> None:
    assert not build_layer(our_nnaudio.CQT(trainable=False)).trainable
    assert not build_layer(our_nnaudio.CQT(trainable=False)).trainable_weights

    assert build_layer(our_nnaudio.CQT(trainable=True)).trainable
    assert len(build_layer(our_nnaudio.CQT(trainable=True)).trainable_variables) == 2
    assert len(build_layer(our_nnaudio.CQT(trainable=True)).trainable_weights) == 2


@pytest.mark.parametrize("train", (True, False))
def test_cqt_trainable_layers_change_on_training(train: bool) -> None:
    # Make a model that's trainable, then train it and ensure the weights change from the default.
    trainable_cqt = our_nnaudio.CQT(trainable=True)
    untrained_cqt = build_layer(our_nnaudio.CQT(trainable=True), (TEST_SAMPLE_RATE * 10,))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer((TEST_SAMPLE_RATE * 10,)),
            trainable_cqt,
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(loss="mse")
    if train:
        signal = librosa.chirp(32.70, TEST_SAMPLE_RATE, length=TEST_SAMPLE_RATE * 10, linear=True)
        noise = np.random.rand(*signal.shape).astype(signal.dtype)

        model.fit(np.array([signal, noise]), np.array([1, 0]), epochs=10, verbose=0)

    for untrained, trained in zip(untrained_cqt.trainable_weights, trainable_cqt.trainable_weights):
        weights_match_untrained = np.allclose(untrained.numpy(), trained.numpy())
        if train:
            assert not weights_match_untrained
        else:
            assert weights_match_untrained
