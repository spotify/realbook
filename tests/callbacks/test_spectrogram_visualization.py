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

from typing import Any

import platform
import pytest
import numpy as np
import tensorflow as tf

try:
    from realbook.callbacks.spectrogram_visualization import SpectrogramVisualizationCallback
except ImportError as e:
    if "numpy.core.multiarray failed to import" in str(e) and platform.system() == "Windows":
        SpectrogramVisualizationCallback = None  # type: ignore
    else:
        raise

from realbook.layers.signal import Spectrogram


try:
    from contextlib import nullcontext
except ImportError:

    class nullcontext:  # type: ignore
        def __init__(self, obj: Any = None) -> None:
            self.obj = obj

        def __enter__(self) -> Any:
            return self.obj

        def __exit__(self, *args: Any, **kwargs: Any) -> None:
            pass


class FakeWriter:
    def as_default(self) -> Any:
        return nullcontext(self)

    def flush(self) -> None:
        pass


DEFAULT_SAMPLE_RATE = 22050
TEST_AUDIO = np.linspace(0, 1, num=DEFAULT_SAMPLE_RATE * 10)


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_spectrogram_visualization_callback() -> None:
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
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="binary_crossentropy")

    cb = SpectrogramVisualizationCallback(
        FakeWriter(),
        fake_data,
        sample_rate=DEFAULT_SAMPLE_RATE,
        raise_on_error=True,
    )

    model.fit(fake_data, callbacks=[cb])
    assert True


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_callback_fails_on_unbatched_input() -> None:
    fake_data = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices([TEST_AUDIO]),
            tf.data.Dataset.from_tensor_slices([1]),
        )
    )

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,)),
            Spectrogram(),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="binary_crossentropy")

    # Cause the callback to error by passing in unbatched data.
    cb = SpectrogramVisualizationCallback(
        FakeWriter(),
        fake_data,
        sample_rate=DEFAULT_SAMPLE_RATE,
        raise_on_error=True,
    )

    with pytest.raises(AssertionError) as excinfo:
        model.fit(fake_data.batch(1), callbacks=[cb])
    assert "shape" in str(excinfo.value)


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_callback_logs_but_doesnt_throw_by_default(caplog: pytest.LogCaptureFixture) -> None:
    fake_data = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices([TEST_AUDIO]),
            tf.data.Dataset.from_tensor_slices([1]),
        )
    )

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None,)),
            Spectrogram(),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="binary_crossentropy")
    # Cause the callback to error by passing in unbatched data.
    cb = SpectrogramVisualizationCallback(FakeWriter(), fake_data, sample_rate=DEFAULT_SAMPLE_RATE)
    model.fit(fake_data.batch(1), callbacks=[cb])
    assert "AssertionError" in caplog.text
    assert "shape" in caplog.text


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_fails_on_no_image_like_layers() -> None:
    fake_data = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices([TEST_AUDIO]),
            tf.data.Dataset.from_tensor_slices([1]),
        )
    ).batch(1)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None, 1)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(loss="binary_crossentropy")

    cb = SpectrogramVisualizationCallback(
        FakeWriter(),
        fake_data,
        sample_rate=DEFAULT_SAMPLE_RATE,
        raise_on_error=True,
    )

    with pytest.raises(ValueError) as excinfo:
        model.fit(fake_data, callbacks=[cb])
    assert isinstance(excinfo.value, ValueError)
    assert "spectrogram" in str(excinfo.value)


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_flexible_with_input_shapes() -> None:
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

    cb = SpectrogramVisualizationCallback(
        FakeWriter(),
        fake_data,
        sample_rate=DEFAULT_SAMPLE_RATE,
        raise_on_error=True,
    )

    model.fit(fake_data, callbacks=[cb])
    assert True


@pytest.mark.skipif(
    SpectrogramVisualizationCallback is None,
    reason="SpectrogramVisualizationCallback import fails on this platform",
)
def test_keras_functional_api_with_tfop_lambda() -> None:
    fake_data = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices([TEST_AUDIO]),
            tf.data.Dataset.from_tensor_slices([1]),
        )
    ).batch(1)

    _input = tf.keras.Input(shape=(None,))
    x = Spectrogram()(_input)
    # Add a final "channel" dimension after the spectrogram
    x = tf.expand_dims(x, -1)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=_input, outputs=x)
    model.compile(loss="binary_crossentropy")

    cb = SpectrogramVisualizationCallback(
        FakeWriter(),
        fake_data,
        sample_rate=DEFAULT_SAMPLE_RATE,
        raise_on_error=True,
    )

    model.fit(fake_data, callbacks=[cb])
    assert True
