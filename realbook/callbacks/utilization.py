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

import abc
import threading
import time
import warnings
from typing import Any, Callable, Dict, cast

import tensorflow as tf

try:
    import psutil
    import nvsmi
except ImportError as e:
    raise ImportError(str(e) + " (did you 'pip3 install realbook[tensorboard]'?)")


class UtilizationCallback(tf.keras.callbacks.Callback):
    CHECK_FINISHED_TIMEOUT: float = 0.1
    RESOURCE_NAME: str = ""

    def __init__(self, tensorboard_writer: tf.summary.SummaryWriter, poll_period_seconds: float = 30):
        """
        A base class that can be used to create callbacks for measuring utilization of some resource.
        To override, override RESOURCE_NAME and resource_fn. resource_fn is a function that measures
        some resource and returns a percentage translating to amount utilized.

        This callback is not meant to be used in distributed training environments.

        Args:
            tensorboard_writer: A tensorboard SummaryWriter to write scalar summary events to.
            poll_period_seconds: The period (in seconds) at which to poll the resource.
        """
        if self.RESOURCE_NAME == "":
            raise ValueError("RESOURCE_NAME must be specified. Did you forget to overwrite it?")
        self.tensorboard_writer = tensorboard_writer
        # TF Can't really serialize objects for multiprocessing so...we get threads!
        self.measurement_thread = threading.Thread(
            target=self.measurement_poller,
            args=(
                self.tensorboard_writer,
                self.resource_fn,
                self.RESOURCE_NAME,
                poll_period_seconds,
            ),
            daemon=True,
        )
        self.finished = False

    def on_train_begin(self, logs: Dict[Any, Any]) -> None:
        """
        Runs at the beginning of training and starts the measurement thread.
        """
        self.measurement_thread.start()

    def on_train_end(self, logs: Dict[Any, Any]) -> None:
        """
        Runs at the end of training. Stops the measurement thread and flushes any buffered
        measurements.
        """
        self.finished = True
        self.measurement_thread.join()
        self.tensorboard_writer.flush()

    def measurement_poller(
        self,
        tensorboard_writer: tf.summary.SummaryWriter,
        resource_fn: Callable[[], float],
        resource_name: str,
        poll_period_seconds: float,
    ) -> None:
        """
        Function which is meant to run during the duration of an entire training run. This is
        what polls resource_fn(). This function takes arguments even though its run through
        threading in case we ever have the ability to switch it over to using multiprocessing.

        Args:
            tensorboard_writer: The SummaryWriter instance which holds tensorboard information.
            resource_fn: The function which runs the resource measurement logic.
            resource_name: The name of the resource to measure.
            poll_period_seconds: The period at which to poll in seconds.
        """

        reference_time = time.time()
        with tensorboard_writer.as_default():
            while not self.finished:
                last_update = time.time()
                util = resource_fn()
                tf.summary.scalar("Utilization/" + resource_name, util, int(last_update - reference_time))
                while not self.finished and (time.time() - last_update) < poll_period_seconds:
                    time.sleep(self.CHECK_FINISHED_TIMEOUT)

    @abc.abstractmethod
    def resource_fn(self) -> float:
        """
        A function to measure the utilization of a resource. Override this function with your
        resource measurement logic.

        Returns:
            The percentage used of a resource.
        """
        pass


class CpuUtilizationCallback(UtilizationCallback):
    """
    A utilization callback which measures cpu utilization.
    """

    RESOURCE_NAME = "CPU Utilization (%)"

    def resource_fn(self) -> float:
        return cast(float, psutil.cpu_percent())


class MemoryUtilizationCallback(UtilizationCallback):
    """
    A utilization callback which measures RAM utilization.
    """

    RESOURCE_NAME = "Memory Used (%)"

    def resource_fn(self) -> float:
        mem_stats = psutil.virtual_memory()
        # Don't use used as the psutil docs say that
        # "total - free does not necessarily match used."
        return (1.0 - float(mem_stats.available) / float(mem_stats.total)) * 100.0


class GpuResourceCallback(UtilizationCallback):
    """
    Base class for GPU utilization callbacks. This checks to see if a GPU is present.
    This class assumes that an NVIDIA GPU will be used.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Inititialize a GPU utilization callback. This callback will become a no-op
        if a GPU is not present of if nvidia-smi is not installed.
        """
        self.enable = True
        if not nvsmi.is_nvidia_smi_on_path():
            warnings.warn(
                "nvidia-smi is not on path. Please disable this callback. GPU usage data will be unavailable."
            )
            self.enable = False

        if self.enable and len(list(nvsmi.get_gpus())) == 0:
            warnings.warn("No GPUs detected. Please disable this callback. GPU usage data will be unavailable.")
            self.enable = False

        super().__init__(*args, **kwargs)

    def on_train_begin(self, logs: Dict[Any, Any]) -> None:
        """
        Runs at the beginning of training. Only starts the measurement thread if a GPU
        is available.
        """
        if self.enable:
            super().on_train_begin(logs)

    def on_train_end(self, logs: Dict[Any, Any]) -> None:
        """
        Runs at the end of training. Only stops the measurement thread if a GPU is
        available.
        """
        if self.enable:
            super().on_train_end(logs)


class GpuUtilizationCallback(GpuResourceCallback):
    """
    This class reports GPU utilization.
    """

    RESOURCE_NAME = "GPU Utilization (%)"

    def resource_fn(self) -> float:
        """
        Return: GPU utilization
        """
        return cast(float, next(nvsmi.get_gpus()).gpu_util)


class GpuMemoryUtilizationCallback(GpuResourceCallback):
    """
    This class reports GPU Memory utilization.
    """

    RESOURCE_NAME = "GPU Memory Utilization (%)"

    def resource_fn(self) -> float:
        """
        Return: GPU Memory utilization
        """
        return cast(float, next(nvsmi.get_gpus()).mem_util)
