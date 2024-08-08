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

from typing import Optional, Union, List

import platform
import numpy as np
import pytest
import tensorflow as tf

try:
    import librosa
    from librosa.core.spectrum import _spectrogram
    from librosa.feature.spectral import melspectrogram
except ImportError as e:
    if "numpy.core.multiarray failed to import" in str(e) and platform.system() == "Windows":
        librosa = None
    else:
        raise

from realbook.layers import signal


def test_stft_channels_should_raise() -> None:
    x = np.random.normal(0, 1, 1024)
    x = tf.expand_dims(tf.expand_dims(x, -1), 1)
    with pytest.raises(ValueError):
        signal.Stft(
            fft_length=256,
            hop_length=256,
            window_length=256,
            center=True,
        )(x)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "center,input_length,fft_length,hop_length,win_length",
    [
        [True, 1024, 256, 256, 256],
        [False, 1024, 256, 256, 256],
        [False, 1024, 256, 256, 128],
        [False, 1024, 256, 512, 256],
    ],
)
def test_stft(center: bool, input_length: int, fft_length: int, hop_length: int, win_length: int) -> None:
    x = np.random.normal(0, 1, input_length)
    print(x.shape)
    librosa_stft = librosa.stft(
        y=x,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    ).T
    rgp_stft = signal.Stft(
        fft_length=fft_length,
        hop_length=hop_length,
        window_length=win_length,
        center=center,
    )(x).numpy()
    assert np.allclose(librosa_stft.real, rgp_stft.real, atol=1e-3, rtol=0)
    assert np.allclose(librosa_stft.imag, rgp_stft.imag, atol=1e-3, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
def test_stft_batch() -> None:
    x = np.random.normal(0, 1, 1024)
    librosa_stft = librosa.stft(
        x,
        n_fft=256,
        hop_length=256,
        win_length=256,
        center=True,
    ).T
    rgp_stft = signal.Stft(
        fft_length=256,
        hop_length=256,
        window_length=256,
        center=True,
    )(tf.expand_dims(x, 0)).numpy()
    assert np.allclose(librosa_stft.real, rgp_stft.real, atol=1e-3, rtol=0)
    assert np.allclose(librosa_stft.imag, rgp_stft.imag, atol=1e-3, rtol=0)


def test_istft_channels_should_raise() -> None:
    x = np.random.normal(0, 1, 1024)
    x = tf.expand_dims(x, 0)
    x_stft = signal.Stft(
        fft_length=256,
        hop_length=256,
        window_length=256,
        center=True,
    )(x)
    with pytest.raises(ValueError):
        signal.Istft(
            fft_length=256,
            hop_length=256,
            window_length=256,
            center=True,
        )(tf.expand_dims(x_stft, -1))


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "center,input_length,fft_length,hop_length,win_length",
    [
        [True, 1024, 256, 128, 256],
        [True, 1024, 256, 64, 256],
        [False, 1024, 256, 128, 256],
        [False, 1024, 256, 64, 256],
        # TODO: The following case creates nans in the output. Fix that (nan => 0 does not count).
        # [False, 1024, 256, 128, 128],
        [False, 1024, 256, 256, 256],
        [True, 1025, 256, 128, 256],
        [False, 1025, 256, 128, 256],
        [True, 1023, 256, 128, 256],
        [False, 1023, 256, 128, 256],
    ],
)
def test_istft(center: bool, input_length: int, fft_length: int, hop_length: int, win_length: int) -> None:
    x = np.random.normal(0, 1, input_length)
    stft = signal.Stft(
        fft_length=fft_length,
        hop_length=hop_length,
        window_length=win_length,
        center=center,
    )(x)
    if fft_length == hop_length:
        with pytest.raises(ValueError):
            signal.Istft(
                fft_length=fft_length,
                hop_length=hop_length,
                window_length=win_length,
                center=center,
            )(stft).numpy()
    else:
        istft = signal.Istft(
            fft_length=fft_length,
            hop_length=hop_length,
            window_length=win_length,
            center=center,
        )(stft).numpy()

        librosa_istft = librosa.istft(
            stft.numpy().T,
            center=center,
            hop_length=hop_length,
            win_length=win_length,
        )

        assert np.allclose(
            istft,
            librosa_istft,
            atol=0.02,
            rtol=0,
        )

        if center:
            assert np.allclose(x[: len(istft)], istft, atol=1e-3, rtol=0)
        else:
            istft = istft[hop_length:-hop_length]
            x = x[hop_length:-hop_length]
            x = x[: len(istft)]
            assert np.allclose(x, istft, atol=1e-3, rtol=0)


def test_istft_batch() -> None:
    x = np.random.normal(0, 1, 1024)
    x_stft = signal.Stft(
        fft_length=256,
        hop_length=128,
        window_length=256,
        center=True,
    )(tf.expand_dims(x, 0))
    x_istft = signal.Istft(
        fft_length=256,
        hop_length=128,
        window_length=256,
        center=True,
    )(x_stft).numpy()
    assert np.allclose(x, np.squeeze(x_istft), atol=1e-3, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
def test_spectrogram() -> None:
    x = np.random.normal(0, 1, 1024).astype(np.float32)
    librosa_spec, _ = _spectrogram(
        y=x,
        n_fft=1024,
        hop_length=256,
        power=2,
        win_length=256,
        center=True,
    )
    librosa_spec = librosa_spec.T
    rgp_spec = signal.Spectrogram(
        fft_length=1024,
        hop_length=256,
        window_length=256,
        center=True,
    )(x).numpy()
    assert np.allclose(librosa_spec, rgp_spec, atol=1e-2, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "center,normalization,fmin,htk",
    [
        [True, None, 0.0, True],
        [True, None, 0.0, False],
        [True, None, 125.0, True],
        [False, signal.MelSpectrogram.SLANEY, 0.0, False],
        [False, 1, 0.0, False],
        [False, 2, 0.0, False],
        [False, np.inf, 0.0, False],
    ],
)
def test_mel_spectrogram(
    center: bool,
    normalization: Optional[Union[str, int, float]],
    fmin: float,
    htk: bool,
) -> None:
    x = np.random.normal(0, 1, 1024 * 8)
    librosa_spec = melspectrogram(
        y=x,
        n_fft=256,
        hop_length=128,
        power=2,
        win_length=256,
        center=center,
        htk=htk,
        norm=normalization,
        dtype=np.float32,
        fmin=fmin,
    ).T
    rgp_spec = signal.MelSpectrogram(
        fft_length=256,
        hop_length=128,
        center=center,
        normalization=normalization,
        htk=htk,
        lower_edge_hertz=fmin,
    )(x).numpy()
    # These tolerances make look large, but the values we're comparing are in the 10e2-10e4 range.
    assert np.allclose(librosa_spec, rgp_spec, atol=1e-2, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "input_spec",
    [
        None,
        [
            [0],
            [1],
            [2],
            [1j],
            [2j],
        ],
        [
            [1 + 2j],
            [3 - 4j],
            [-5 + 6j],
            [-7 - 8j],
            [9 + 10j],
        ],
    ],
)
def test_magnitude(input_spec: Optional[List[np.complex64]]) -> None:
    if input_spec is None:
        x = np.random.normal(0, 1, 1024)
        x_stft = librosa.stft(
            x,
            n_fft=256,
            hop_length=256,
            win_length=256,
            center=True,
        ).T
    else:
        x_stft = np.array(input_spec, dtype=np.complex64)

    np_magnitude = np.abs(x_stft)
    layer_magnitude = signal.Magnitude()(tf.expand_dims(x_stft, 0)).numpy()

    assert np.allclose(np_magnitude, layer_magnitude, atol=1e-3, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "input_spec",
    [
        None,
        [
            [0],
            [1],
            [2],
            [1j],
            [2j],
        ],
        [
            [1 + 2j],
            [3 - 4j],
            [-5 + 6j],
            [-7 - 8j],
            [9 + 10j],
        ],
    ],
)
def test_phase(input_spec: Optional[List[np.complex64]]) -> None:
    if input_spec is None:
        x = np.random.normal(0, 1, 1024)
        x_stft = librosa.stft(
            x,
            n_fft=256,
            hop_length=256,
            win_length=256,
            center=True,
        ).T
    else:
        x_stft = np.array(input_spec, dtype=np.complex64)

    np_phase = np.angle(x_stft)
    layer_phase = signal.Phase()(tf.expand_dims(x_stft, 0)).numpy()

    assert np.allclose(np_phase, layer_phase, atol=1e-3, rtol=0)


@pytest.mark.skipif(librosa is None, reason="Librosa failed to import on this platform.")
@pytest.mark.parametrize(
    "ref,amin,top_db",
    [
        [1.0, 1e-10, 80.0],
        [2.0, 1e-5, 60.0],
        [0, 1e-5, 40.0],
        [0.5, 1e-5, 40.0],
    ],
)
def test_magnitude_to_decibel(ref: float, amin: float, top_db: float) -> None:
    x = np.random.normal(0, 1, 1024)
    x_stft_magnitude = np.abs(
        librosa.stft(
            x,
            n_fft=256,
            hop_length=256,
            win_length=256,
            center=True,
        ).T
    )

    librosa_magnitude_to_decibel = librosa.power_to_db(x_stft_magnitude, ref=ref, amin=amin, top_db=top_db)
    layer_magnitude_to_decibel = signal.MagnitudeToDecibel(ref=ref, amin=amin, top_db=top_db)(
        tf.expand_dims(x_stft_magnitude, 0)
    ).numpy()

    assert np.allclose(librosa_magnitude_to_decibel, layer_magnitude_to_decibel, atol=1e-3, rtol=0)
