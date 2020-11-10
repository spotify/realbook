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

import faulthandler
import os
import signal
import sys
import tensorflow as tf
import time
import threading
from typing import Any, List, Optional


class HangDebugCallback(tf.keras.callbacks.Callback):
    """
    Sometimes, when training models, training jobs can hang for unknown reasons.
    This callback will dump stack traces to standard error if an epoch takes more
    than 2x the average time, and will exit the process if an epoch takes longer
    than 5x the average time.

    Useful when training on cloud-based training systems that don't allow for sending
    SIGINT/KeyboardInterrupt/Ctrl-C to a process.
    """

    CHECK_INTERVAL_SECONDS: float = 1.0

    def __init__(
        self,
        print_stacks_if_epoch_time_exceeds_multiple: float = 2,
        abort_if_epoch_time_exceeds_multiple: float = 5,
    ):
        """
        Create a new HangDebugCallback.

        Args:
            print_stacks_if_epoch_time_exceeds_multiple (float):
                if a single epoch takes this many times longer than the average,
                stack traces of all threads will be printed to the standard error stream.
            abort_if_epoch_time_exceeds_multiple (float):
                if a single epoch takes this many times longer than the average,
                this process will be exited with return code 6 (SIGABRT).
        """
        self.monitoring_thread = threading.Thread(target=self.monitoring_fn, name="hang_debug_thread", daemon=True)
        self.epoch_start_time: Optional[float] = None
        self.finished = False
        self.epoch_times: List[float] = []
        self.dumped_stacks_already = False
        self.print_stacks_if_epoch_time_exceeds_multiple = print_stacks_if_epoch_time_exceeds_multiple
        self.abort_if_epoch_time_exceeds_multiple = abort_if_epoch_time_exceeds_multiple

    def on_train_begin(self, logs: Any) -> None:
        """
        Runs at the beginning of training and starts the measurement thread.
        """
        self.monitoring_thread.start()

    def on_train_end(self, logs: Any) -> None:
        """
        Runs at the end of training. Stops the measurement thread and flushes any buffered
        measurements.
        """
        self.finished = True

    def on_epoch_begin(self, epoch: int, logs: Any) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Any) -> None:
        if self.current_epoch_duration:
            self.epoch_times.append(self.current_epoch_duration)

    @property
    def current_epoch_duration(self) -> Optional[float]:
        if self.epoch_start_time is not None:
            return time.time() - self.epoch_start_time
        return None

    @property
    def mean_epoch_time(self) -> Optional[float]:
        if not self.epoch_times:
            return None
        return sum(self.epoch_times) / len(self.epoch_times)

    @property
    def should_dump_stacks(self) -> bool:
        if not self.current_epoch_duration or not self.mean_epoch_time:
            return False
        return self.current_epoch_duration > (self.mean_epoch_time * self.print_stacks_if_epoch_time_exceeds_multiple)

    @property
    def should_abort(self) -> bool:
        if not self.current_epoch_duration or not self.mean_epoch_time:
            return False
        return self.current_epoch_duration > (self.mean_epoch_time * self.abort_if_epoch_time_exceeds_multiple)

    def abort(self) -> None:
        # AI Platform will retry the job if we raise SIGABRT.
        # We're in a thread here, so we need to kill the whole
        # process with os._exit - otherwise we'll just kill the
        # current thread and not stop the process.
        os._exit(signal.SIGABRT)

    def monitoring_fn(self) -> None:
        while not self.finished:
            if self.should_dump_stacks and not self.dumped_stacks_already and self.mean_epoch_time:
                sys.stderr.write(
                    f"Mean epoch time is {self.mean_epoch_time:2.2f} "
                    "seconds, but current epoch is taking "
                    f"{self.current_epoch_duration:2.2f} seconds. "
                    "Printing stacks for all threads:\n"
                )
                sys.stderr.flush()
                faulthandler.dump_traceback()
                sys.stderr.flush()
                sys.stderr.write(
                    "If the current epoch does not finish within "
                    f"{5 * self.mean_epoch_time} seconds, this job "
                    "will exit.\n"
                )
                sys.stderr.flush()
                self.dumped_stacks_already = True
            if self.should_abort:
                sys.stderr.write(
                    f"Mean epoch time is {self.mean_epoch_time:2.2f} "
                    "seconds, but current epoch is taking "
                    f"{self.current_epoch_duration:2.2f} seconds. "
                    "Aborting job.\n"
                )
                sys.stderr.flush()
                self.abort()
            time.sleep(self.CHECK_INTERVAL_SECONDS)
