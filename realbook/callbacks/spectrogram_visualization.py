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

import logging
from typing import Any, Iterable, Tuple, Union, Callable
import numpy as np
import io
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
    import librosa.display
except ImportError as e:
    raise ImportError(str(e) + " (did you 'pip3 install realbook[tensorboard]'?)")


def plot_to_image(figure: plt.figure) -> tf.Tensor:
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    return tf.expand_dims(tf.image.decode_png(buf.getvalue(), channels=4), 0)


class SpectrogramVisualizationCallback(tf.keras.callbacks.Callback):
    """
    If your model trains on image-like (i.e.: Spectrogram) data, this callback will:
     - identify which layers of your model are the "input" layers (by looking at their trainability)
     - interpret the output of the last non-trainable layer as a spectrogram
     - plot that spectrogram to TensorBoard on training start

    If your model has multiple branching paths in its frontend (i.e.: it produces two spectrograms)
    this callback will only pick the first. Subclass or copy this.

    Parameters:
     - tensorboard_writer: a TensorBoard SummaryWriter to use
     - example_batches: an iterator providing pairs of (batch, label) training data to visualize.
                        can be a tf.data.Dataset.
     - convert_to_dB: a callable to use to convert the post-spectrogram data to dB for Librosa
                      visualization. By default, this is librosa.amplitude_to_db - you may want to
                      use librosa.power_to_db instead if your spectrogram is a power spectrogram.
     - sample_rate: the sample rate of the input audio.
     - raise_on_error: if true, throw an exception if this callback causes an error.
                       by default, log exceptions but don't interrupt training.
     - name: the name of the summary data sent to TensorBoard.
     - Any remaining keyword arguments are passed through to librosa.display.specshow.
    """

    def __init__(
        self,
        tensorboard_writer: tf.summary.SummaryWriter,
        example_batches: Iterable[Tuple[tf.Tensor, tf.Tensor]],
        convert_to_dB: Union[bool, Callable[[tf.Tensor], tf.Tensor]] = True,
        sample_rate: int = 22050,
        # By default log exceptions but don't halt the training process.
        raise_on_error: bool = False,
        name: str = "Input to First Trainable Layer",
        **kwargs: Any,
    ):
        self.tensorboard_writer = tensorboard_writer
        self.example_batches = example_batches
        self.convert_to_dB = convert_to_dB
        self.sample_rate_hz = sample_rate
        self.raise_on_error = raise_on_error
        self.name = name

        self.specshow_arguments = kwargs
        if "x_axis" not in self.specshow_arguments:
            self.specshow_arguments["x_axis"] = "time"

    def on_train_begin(self, logs: Any = None) -> None:
        try:
            # Create a tempoary model using only the frontend of the model,
            # as defined by "the largest sequence of non-trainable layers at the start."
            non_trainable_input_layers = []
            for layer in self.model.layers:
                if len(layer.trainable_variables) + len(layer.trainable_weights) > 0:
                    break
                else:
                    non_trainable_input_layers.append(layer)

            if not non_trainable_input_layers:
                raise ValueError("No non-trainable input layers could be inferred for spectrogram visualization.")

            # Don't use tf.keras.models.Sequential here, as the input may not be traditional Layers.
            # (Yes, you'd think that self.model.layers returns all layers - but that doesn't seem to be the case.)
            input_to_image = tf.keras.models.Model(
                inputs=non_trainable_input_layers[0].input, outputs=non_trainable_input_layers[-1].output
            )

            with self.tensorboard_writer.as_default():
                # Pull n random batches from the dataset and send them to TensorBoard.
                for data, _ in self.example_batches:
                    assert tf.rank(data) == 2, "Expected input data to be of rank 2, with shape (batch, audio)."
                    assert tf.shape(data)[0] < tf.shape(data)[1], (
                        "Expected input data to be of rank 2, with shape (batch, audio), but got shape"
                        f" {tf.shape(data)}."
                    )

                    spectrograms = input_to_image(data)
                    assert tf.rank(spectrograms) in (3, 4), (
                        "Expected non-trainable input layers to produce output of shape (batch, x, y) "
                        f"or (batch, x, y, 1), but got {tf.shape(spectrograms)}"
                    )
                    if tf.rank(spectrograms) == 4:
                        assert tf.shape(spectrograms)[-1] == 1, (
                            "Expected non-trainable input layers to produce output with one channel, but shape is"
                            f" {tf.shape(spectrograms)}"
                        )
                        # Ignore the single channel dimension, if it exists.
                        spectrograms = spectrograms[:, :, :, 0]

                    # We can infer the hop length, as we know the input audio length
                    # and sample rate used in the spectrogram
                    length_in_samples = data.shape[-1]
                    length_in_frames = spectrograms.shape[-2]
                    hop_length = int(tf.math.ceil(length_in_samples / length_in_frames))

                    figs = []
                    for spectrogram in spectrograms:
                        plt.clf()
                        fig, ax = plt.subplots()
                        spectrogram = np.abs(spectrogram).T

                        if self.convert_to_dB is True:
                            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
                        elif callable(self.convert_to_dB):
                            spectrogram = self.convert_to_dB(spectrogram)

                        librosa.display.specshow(
                            spectrogram,
                            sr=self.sample_rate_hz,
                            hop_length=hop_length,
                            ax=ax,
                            **self.specshow_arguments,
                        )

                        figs.append(plot_to_image(fig))
                    tf.summary.image(
                        self.name,
                        np.concatenate(figs),
                        step=0,  # We only output this once, so epoch doesn't matter.
                        max_outputs=1000000,
                    )
                    plt.clf()
            self.tensorboard_writer.flush()
        except Exception as e:
            if self.raise_on_error:
                raise
            logging.error(f"{self.__class__.__name__} failed: ", exc_info=e)
