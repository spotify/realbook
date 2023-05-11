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

import warnings
from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf
import numpy as np

from realbook.layers.math import log_base_b
from realbook.vendor import librosa_filters


def _create_padded_window(
    window_fn: Callable[[int, tf.dtypes.DType], tf.Tensor],
    unpadded_window_length: int,
    fft_length: int,
) -> Callable[[int, tf.dtypes.DType], tf.Tensor]:
    lpad = (fft_length - unpadded_window_length) // 2
    rpad = fft_length - unpadded_window_length - lpad

    def padded_window(window_length: int, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
        # This is a trick to match librosa's way of handling window lengths < their fft_lengths
        # In that case the window is 0 padded such that the window is centered around 0s
        # In the Tensorflow case, the window is computed, multiplied against the frame and then
        # Right padded with 0's.
        return tf.pad(window_fn(unpadded_window_length, dtype=dtype), [[lpad, rpad]])  # type: ignore

    return padded_window


class Stft(tf.keras.layers.Layer):
    def __init__(
        self,
        fft_length: int = 2048,
        hop_length: Optional[int] = None,
        window_length: Optional[int] = None,
        window_fn: Callable[[int, tf.dtypes.DType], tf.Tensor] = tf.signal.hann_window,
        pad_end: bool = False,
        center: bool = True,
        pad_mode: str = "CONSTANT",
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        A Tensorflow Keras layer that calculates an STFT.
        The input is real-valued with shape (num_batches, num_samples).
        The output is complex-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            hop_length: The "stride" or number of samples to iterate before the start of the next frame.
            fft_length: FFT length.
            window_length: Window length. If None, then fft_length is used.
            window_fn: A callable that takes a window length and a dtype and returns a window.
            pad_end: Whether to pad the end of signals with zeros when the provided frame length and step produces
                a frame that lies partially past its end.
            center:
                If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
                If False, then D[:, t] begins at y[t * hop_length].
            pad_mode: Padding to use if center is True. One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
            name: Name of the layer.
            dtype: Type used in calcuation.
        """
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=False)
        self.fft_length = fft_length
        self.window_length = window_length if window_length else self.fft_length
        self.hop_length = hop_length if hop_length else self.window_length // 4
        self.window_fn = window_fn
        self.final_window_fn = window_fn
        self.pad_end = pad_end
        self.center = center
        self.pad_mode = pad_mode

    def build(self, input_shape: tf.TensorShape) -> None:
        if input_shape.rank > 2:
            raise ValueError(
                "realbook.layers.signal.Stft received an input shape of "
                f"{input_shape}, but only supports inputs shaped "
                "like (num_samples,) or (num_batches, num_samples)."
            )

        if self.window_length < self.fft_length:
            self.final_window_fn = _create_padded_window(
                window_fn=self.window_fn,
                unpadded_window_length=self.window_length,
                fft_length=self.fft_length,
            )

        if self.center:
            self.spec = tf.keras.layers.Lambda(
                lambda x: tf.pad(
                    x,
                    [[0, 0] for _ in range(input_shape.rank - 1)] + [[self.fft_length // 2, self.fft_length // 2]],
                    mode=self.pad_mode,
                )
            )
        else:
            self.spec = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.signal.stft(
            signals=self.spec(inputs),
            frame_length=self.fft_length,
            frame_step=self.hop_length,
            fft_length=self.fft_length,
            window_fn=self.final_window_fn,
            pad_end=self.pad_end,
        )

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        config.update(
            {
                "fft_length": self.fft_length,
                "window_length": self.window_length,
                "hop_length": self.hop_length,
                "window_fn": self.window_fn,
                "pad_end": self.pad_end,
                "center": self.center,
                "pad_mode": self.pad_mode,
            }
        )
        return config


class Istft(tf.keras.layers.Layer):
    def __init__(
        self,
        fft_length: int = 2048,
        hop_length: Optional[int] = None,
        window_length: Optional[int] = None,
        window_fn: Callable[[int, tf.dtypes.DType], tf.Tensor] = tf.signal.hann_window,
        center: bool = True,
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.complex64,
    ):
        """
        A Tensorflow Keras layer that calculates an STFT.
        The input is complex-valued with shape (num_batches, num_frames, fft_length // 2 + 1)
            or (num_frames, fft_length // 2 + 1).
        The output is complex-valued with shape (num_batches, num_samples) or (num_samples).

        Args:
            hop_length: The "stride" or number of samples to iterate before the start of the next frame.
            fft_length: FFT length.
            window_length: Window length. If None, then fft_length is used.
                Warning. If this is less than fft_length then nans may appear in the output.
            window_fn: A callable that takes a window length and a dtype and returns a window.
                This will be converted to an inverse_window_fn.
                using tf.signal.inverse_stft_window_fn.
            center:
                If True, undo the centering done from stft (y(t) was padded so
                    that frame D[:, t] is centered at y[t * hop_length]).
                If False, don't modify the output of the istft.
                    You may need to trim the first and last hop_length samples
                    to match the original signal.
            name: Name of the layer.
            dtype: Type used in calcuation.
        """
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=False)

        # Because layer saves dtype as string???
        self.dtypes_type = dtype
        self.fft_length = fft_length
        self.window_length = window_length if window_length else self.fft_length
        self.window_fn = window_fn
        self.hop_length = hop_length if hop_length else self.window_length // 4
        self.center = center

        if self.fft_length <= self.hop_length:
            # If this is true then the inverse window function will contain a nan in some cases.
            raise ValueError("FFT Length must be less than or equal to hop length or else nans will appear.")

        if self.window_length < self.fft_length:
            warnings.warn(
                f"Istft layer received a window length ({self.window_length:,}) smaller than the provided FFT length"
                f" ({self.fft_length:,}).  NaN values may appear in this layer's output."
            )

        if not self.center:
            warnings.warn(
                "center=False has been passed to an Istft meaning perfect reconstruction is not obtainable."
                "The output in the range [hop_length:-hop_length] will still be valid."
            )

    def build(self, input_shape: tf.TensorShape) -> None:
        if input_shape.rank > 3 or input_shape.rank <= 1:
            raise ValueError(
                "realbook.layers.signal.Istft received an input shape of "
                f"{input_shape}, but only supports inputs shaped "
                "like (num_frames, num_bins) or (num_batches, num_frames, num_bins)."
            )

        self.window = _create_padded_window(self.window_fn, self.window_length, self.fft_length)(
            self.fft_length
        )  # type: ignore

        self.window_sum = librosa_filters.window_sumsquare(  # type: ignore
            window=self.window.numpy(),
            n_frames=input_shape[0] if input_shape.rank == 2 else input_shape[1],
            win_length=self.window_length,
            n_fft=self.fft_length,
            hop_length=self.hop_length,
            dtype=self.dtypes_type.as_numpy_dtype,
        )

        self.window_sum = tf.constant(
            np.where(
                self.window_sum > np.finfo(self.dtypes_type.as_numpy_dtype).tiny,
                self.window_sum,
                np.ones_like(self.window_sum),
            ),
            self.dtypes_type.real_dtype,
        )

        self.slice_op = tf.keras.layers.Lambda(lambda x: x)
        if self.center:
            if input_shape.rank == 2:  # unbatched
                self.slice_op = tf.keras.layers.Lambda(
                    lambda x: x[int(self.fft_length // 2) : -int(self.fft_length // 2)]
                )
            else:  # batched
                self.slice_op = tf.keras.layers.Lambda(
                    lambda x: x[:, int(self.fft_length // 2) : -int(self.fft_length // 2)]
                )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        istft = (
            tf.signal.overlap_and_add(self.window * tf.signal.irfft(inputs), frame_step=self.hop_length)
            / self.window_sum
        )
        return self.slice_op(istft)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        config.update(
            {
                "dtypes_type": self.dtypes_type,
                "fft_length": self.fft_length,
                "window_length": self.window_length,
                "hop_length": self.hop_length,
                "window_fn": self.window_fn,
                "center": self.center,
            }
        )
        return config


class Spectrogram(Stft):
    def __init__(
        self,
        power: int = 2,
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A Tensorflow Keras layer that calculates the magnitude spectrogram.
        The input is real-valued with shape (num_batches, num_samples).
        The output is real-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            power: Exponent to raise abs(stft) to.
            name: Name of the layer.
            dtype: Type used in calcuation.
            **kwargs: Any arguments that you'd pass to Stft
        """
        super().__init__(
            name=name,
            dtype=dtype,  # type: ignore
            *args,
            **kwargs,
        )
        self.power = power

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.pow(
            tf.math.abs(super().call(inputs)),
            self.power,
        )

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        config.update(
            {
                "power": self.power,
            }
        )
        return config


class MelSpectrogram(Spectrogram):
    SLANEY = "slaney"  # slaney

    # the "type" of np.inf is np.float
    def __init__(
        self,
        sample_rate: int = 22050,
        fft_length: int = 2048,
        n_mels: int = 128,
        lower_edge_hertz: float = 0.0,
        upper_edge_hertz: Optional[float] = None,
        htk: bool = False,
        normalization: Optional[Union[float, int, str]] = SLANEY,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A Tensorflow Keras layer that calculates the Mel Spectrogram of a signal.
        The input is real-valued with shape (num_batches, num_samples).
        The output is real-valued with shape (num_batches, time, n_mels)

        This function inherits from Spectrogram and supports all arguments from its
        __init__.

        Args:
            sample_rate: Sample rate of the input signal.
            fft_length: FFT length. If this is not a power of 2, the next power of 2 will be used as the FFT length.
            n_mels: How many bands in the resulting mel spectrum.
            lower_edge_hertz: Lower bound on the frequencies to be used in the Mel Spectrogram.
            upper_edge_hertz: Upper bound on the frequencies to be used in the Mel Spectrogram.
            htk: Use HTK formula for mel filters calculation instead of slaney.
            normalization: If 'slaney', divide the triangular mel weights by the width of the mel band
                (area normalization).
                Else, any value supported by [tf.norm](https://www.tensorflow.org/api_docs/python/tf/norm).
                If None, leave all the triangles aiming for a peak value of 1.0
        """
        fft_length = 1 << (fft_length - 1).bit_length()
        super().__init__(fft_length=fft_length, *args, **kwargs)
        self.n_mels = n_mels
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz if upper_edge_hertz else float(sample_rate) / 2.0
        self.htk = htk
        self.sample_rate = sample_rate
        self.normalization = normalization

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)

        self.mel_weight_matrix = tf.constant(
            librosa_filters.mel(  # type: ignore
                sr=self.sample_rate,
                n_fft=self.fft_length,
                n_mels=self.n_mels,
                fmin=self.lower_edge_hertz,
                fmax=self.upper_edge_hertz,
                htk=self.htk,
                norm=self.normalization,
                dtype=np.float64,
            ).T,
            dtype=self.dtype,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(super().call(inputs), self.mel_weight_matrix)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        config.update(
            {
                "n_mels": self.n_mels,
                "lower_edge_hertz": self.lower_edge_hertz,
                "upper_edge_hertz": self.upper_edge_hertz,
                "htk": self.htk,
                "sample_rate": self.sample_rate,
                "normalization": self.normalization,
            }
        )
        return config


class Magnitude(tf.keras.layers.Layer):
    def __init__(
        self,
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A Tensorflow Keras layer that calculates the magnitude of a complex tensor.

        Args:
            name: Name of the layer.
            dtype: Type used in calculation.
        """
        super().__init__(name=name, dtype=dtype, *args, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.abs(inputs)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        return config


class Phase(tf.keras.layers.Layer):
    def __init__(
        self,
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A Tensorflow Keras layer that calculates the phase of a complex tensor.

        Args:
            name: Name of the layer.
            dtype: Type used in calculation.
        """
        super().__init__(name=name, dtype=dtype, *args, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.angle(inputs)

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        return config


class MagnitudeToDecibel(tf.keras.layers.Layer):
    def __init__(
        self,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0,
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        A Keras layer that converts a real-valued tensor to decibel scale.

        Args:
            ref: Reference power that would be scaled to 0 dB.
            amin: Minimum power threshold.
            top_db: Minimum negative cut-off in decibels.
            name: Name of the layer.
            dtype: Type used in calculation.
        """
        super().__init__(name=name, dtype=dtype)
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        log_spec = 10.0 * (
            log_base_b(tf.math.maximum(inputs, self.amin), 10.0)
            - log_base_b(tf.math.maximum(self.amin, self.ref), 10.0)
        )
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec, keepdims=True) - self.top_db)

        return log_spec

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config().copy()
        config.update(
            {
                "ref": self.ref,
                "amin": self.amin,
                "top_db": self.top_db,
            }
        )
        return config
