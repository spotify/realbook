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

import time
from typing import Any, Optional

import tensorflow as tf


class TrainingSpeedCallback(tf.keras.callbacks.Callback):
    """
    A tiny Keras callback to log to TensorBoard:
     - Seconds per [Epoch, Batch, Example]
     - Batches per Epoch
     - Examples per [Batch, Second]
    all grouped under the "Training Speed" header.
    """

    def __init__(
        self,
        tensorboard_writer: tf.summary.SummaryWriter,
        batches_per_epoch: int,
        examples_per_batch: Optional[int] = None,
    ):
        self.tensorboard_writer = tensorboard_writer
        self.batches_per_epoch = batches_per_epoch
        self.examples_per_batch = examples_per_batch

    def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
        self.epoch_begin_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        epoch_end_time = time.time()
        seconds_per_epoch = epoch_end_time - self.epoch_begin_time
        seconds_per_batch = seconds_per_epoch / self.batches_per_epoch
        with self.tensorboard_writer.as_default():
            tf.summary.scalar(
                "Training Speed/Seconds Per Epoch",
                seconds_per_epoch,
                step=epoch,
            )
            tf.summary.scalar("Training Speed/Batches Per Epoch", self.batches_per_epoch, step=epoch)
            tf.summary.scalar(
                "Training Speed/Seconds Per Batch",
                seconds_per_batch,
                step=epoch,
            )
            if self.examples_per_batch:
                tf.summary.scalar(
                    "Training Speed/Seconds Per Example",
                    seconds_per_batch / self.examples_per_batch,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Training Speed/Examples Per Second",
                    self.examples_per_batch / seconds_per_batch,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Training Speed/Examples Per Batch",
                    self.examples_per_batch,
                    step=epoch,
                )
                tf.summary.scalar(
                    "Training Speed/Examples Per Epoch",
                    self.examples_per_batch * self.batches_per_epoch,
                    step=epoch,
                )
