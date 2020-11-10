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

import faulthandler
import time
from typing import Iterator, List, Tuple
import tensorflow as tf

import pytest
import pytest_mock
from realbook.callbacks.debugging import HangDebugCallback


def test_hang_debug_callback_does_not_fail() -> None:
    def generate_data() -> Iterator[Tuple[List[int], List[int]]]:
        for i in range(0, 10):
            yield ([1], [1])

    fake_data = tf.data.Dataset.from_generator(
        generate_data,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ),
    )

    model = tf.keras.Sequential([tf.keras.Input(shape=(1)), tf.keras.layers.Dense(1)])
    model.compile(loss="binary_crossentropy")
    model.fit(fake_data, steps_per_epoch=1, epochs=10, callbacks=[HangDebugCallback()])


def test_hang_debug_callback_prints_stacks_if_slow(
    mocker: pytest_mock.MockerFixture, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch("faulthandler.dump_traceback")
    print_stacks_time_multiple = 2
    abort_time_multiple = 100000
    hang_debug_callback = HangDebugCallback(print_stacks_time_multiple, abort_time_multiple)

    # Artificially speed up the callback's check frequency:
    hang_debug_callback.CHECK_INTERVAL_SECONDS = 0.1

    def generate_data() -> Iterator[Tuple[List[int], List[int]]]:
        duration = hang_debug_callback.CHECK_INTERVAL_SECONDS
        for i in range(0, 9):
            time.sleep(duration)
            yield ([1], [1])
        time.sleep(duration * print_stacks_time_multiple * 10)
        yield ([1], [1])

    fake_data = tf.data.Dataset.from_generator(
        generate_data,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ),
    )

    model = tf.keras.Sequential([tf.keras.Input(shape=(1)), tf.keras.layers.Dense(1)])
    model.compile(loss="binary_crossentropy")
    model.fit(
        fake_data,
        steps_per_epoch=1,
        epochs=10,
        callbacks=[hang_debug_callback],
        verbose=0,
    )
    captured = capsys.readouterr()
    assert "Printing stacks" in captured.err
    faulthandler.dump_traceback.assert_called_once()  # type: ignore


def test_hang_debug_callback_aborts_if_really_slow(
    mocker: pytest_mock.MockerFixture, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch("faulthandler.dump_traceback")
    print_stacks_time_multiple = 2
    abort_time_multiple = 3
    hang_debug_callback = HangDebugCallback(print_stacks_time_multiple, abort_time_multiple)

    # Artificially speed up the callback's check frequency:
    hang_debug_callback.CHECK_INTERVAL_SECONDS = 0.1

    # Patch the "abort" function to ensure we don't actually kill this Python process:
    hang_debug_callback.abort = mocker.MagicMock()  # type: ignore

    def generate_data() -> Iterator[Tuple[List[int], List[int]]]:
        duration = hang_debug_callback.CHECK_INTERVAL_SECONDS
        for i in range(0, 9):
            time.sleep(duration)
            yield ([1], [1])
        time.sleep(duration * abort_time_multiple * 10)
        yield ([1], [1])

    fake_data = tf.data.Dataset.from_generator(
        generate_data,
        output_signature=(
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ),
    )

    model = tf.keras.Sequential([tf.keras.Input(shape=(1)), tf.keras.layers.Dense(1)])
    model.compile(loss="binary_crossentropy")
    model.fit(
        fake_data,
        steps_per_epoch=1,
        epochs=10,
        callbacks=[hang_debug_callback],
        verbose=0,
    )
    captured = capsys.readouterr()
    assert "Aborting job" in captured.err
    hang_debug_callback.abort.assert_called()  # type: ignore
