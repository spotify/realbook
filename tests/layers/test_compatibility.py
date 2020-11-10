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

import os
import tempfile
from typing import Iterator, List, Tuple
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
import numpy as np
from contextlib import contextmanager
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from realbook.layers.compatibility import (
    FrozenGraphLayer,
    SavedModelLayer,
    get_all_tensors_from_saved_model,
    TensorWrapperLayer,
)

NUM_INPUT_VALUES = 10


def keras_model_to_frozen_graph(
    keras_model: tf.keras.Model,
) -> Tuple[tf.compat.v1.GraphDef, List[tf.Tensor], tf.types.experimental.ConcreteFunction]:
    concrete_function = tf.function(keras_model).get_concrete_function(
        [tf.TensorSpec(i.shape, i.dtype) for i in keras_model.inputs]
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_function)
    input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != tf.resource]
    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        frozen_func.outputs,
        config=get_grappler_config([]),
        graph=frozen_func.graph,
    )
    return graph_def, input_tensors, frozen_func.outputs


@contextmanager
def keras_model_to_savedmodel(keras_model: tf.keras.Model) -> Iterator[str]:
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "temp_model")
        tf.saved_model.save(keras_model, path)
        yield path


def train_addition_model() -> tf.keras.models.Model:
    # Learn a model that sums numbers:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(NUM_INPUT_VALUES),
            tf.keras.layers.Dense(1, name="my_layer"),
        ]
    )
    model.compile(loss="MSE")
    num_examples = 100000
    x = np.random.rand(num_examples, NUM_INPUT_VALUES)
    model.fit(x, np.sum(x, axis=-1))
    return model


def train_named_input_and_output() -> tf.keras.models.Model:
    # Learn a model that sums numbers:
    inputs = {"named_input": tf.keras.layers.Input(NUM_INPUT_VALUES)}
    outputs = {"named_output": tf.keras.layers.Dense(1)(inputs["named_input"])}
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="MSE")
    num_examples = 100000
    x = np.random.rand(num_examples, NUM_INPUT_VALUES)
    model.fit(x, np.sum(x, axis=-1))
    return model


def train_multi_input_multi_output() -> tf.keras.models.Model:
    """Train a dummy model to add numbers, but with multiple inputs and outputs: outAB = inA + inB, etc."""

    input_names = ["inA", "inB", "inC"]
    inputs = {input_name: tf.keras.layers.Input(1) for input_name in input_names}

    output_names = ["outAB", "outBC", "outAC"]
    concatenated = Concatenate()([v for _, v in sorted(inputs.items(), key=lambda t: t[0])])
    outputs = {
        "outAB": Dense(1)(concatenated),
        "outBC": Dense(1)(concatenated),
        "outAC": Dense(1)(concatenated),
    }

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss="MSE")
    num_examples = 100000
    inA, inB, inC = [np.random.rand(num_examples) for _ in range(3)]
    model.fit(
        {"inA": inA, "inB": inB, "inC": inC},
        {"outAB": inA + inB, "outBC": inB + inC, "outAC": inA + inC},
    )
    return model, input_names, output_names


def test_load_frozen_graph_as_layer() -> None:
    model = train_addition_model()
    x = np.random.rand(1, model.input_shape[-1])
    expected_output = model.predict(x)
    graph_def, input_tensors, output_tensors = keras_model_to_frozen_graph(model)

    reloaded_model = tf.keras.Sequential(
        [
            FrozenGraphLayer(
                graph_def,
                input_tensor_names=[t.name for t in input_tensors],
                output_tensor_names=[t.name for t in output_tensors],
            )
        ]
    )
    assert np.allclose(reloaded_model.predict(x), expected_output)


def test_load_frozen_graph_from_disk() -> None:
    model = train_addition_model()
    x = np.random.rand(1, model.input_shape[-1])
    expected_output = model.predict(x)

    graph_def, input_tensors, output_tensors = keras_model_to_frozen_graph(model)

    with tempfile.NamedTemporaryFile("wb") as f:
        f.write(graph_def.SerializeToString())
        f.flush()
        f.seek(0)

        reloaded_model = tf.keras.Sequential(
            [
                FrozenGraphLayer(
                    f.name,
                    input_tensor_names=[t.name for t in input_tensors],
                    output_tensor_names=[t.name for t in output_tensors],
                )
            ]
        )

    assert np.allclose(reloaded_model.predict(x), expected_output)


def test_reserialize_model() -> None:
    model = train_addition_model()
    x = np.random.rand(1, model.input_shape[-1])
    expected_output = model.predict(x)

    graph_def, input_tensors, output_tensors = keras_model_to_frozen_graph(model)

    with tempfile.NamedTemporaryFile("wb") as f:
        f.write(graph_def.SerializeToString())
        f.flush()
        f.seek(0)

        from_frozen = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(NUM_INPUT_VALUES),
                FrozenGraphLayer(
                    f.name,
                    input_tensor_names=[t.name for t in input_tensors],
                    output_tensor_names=[t.name for t in output_tensors],
                ),
            ]
        )

    assert np.allclose(from_frozen.predict(x), expected_output)

    with tempfile.TemporaryDirectory() as tempdir:
        saved_path = os.path.join(tempdir, "output_model")
        from_frozen.save(saved_path)
        del from_frozen
        reloaded = tf.keras.models.load_model(saved_path)
    assert np.allclose(reloaded.predict(x), expected_output)


def test_load_multi_input_graph() -> None:
    # Learn a model that sums numbers, but uses multiple inputs:
    input_1 = tf.keras.layers.Input((NUM_INPUT_VALUES // 2,))
    input_2 = tf.keras.layers.Input((NUM_INPUT_VALUES // 2,))
    concat = tf.keras.layers.Concatenate()([input_1, input_2])
    output = tf.keras.layers.Dense(1)(concat)

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=[output])
    model.compile(loss="MSE")
    num_examples = 100000
    x1 = np.random.rand(num_examples, NUM_INPUT_VALUES // 2)
    x2 = np.random.rand(num_examples, NUM_INPUT_VALUES // 2)
    model.fit((x1, x2), np.sum(np.concatenate((x1, x2), axis=-1), axis=-1))

    x1 = np.random.rand(1, NUM_INPUT_VALUES // 2)
    x2 = np.random.rand(1, NUM_INPUT_VALUES // 2)
    expected_output = model.predict([x1, x2])
    graph_def, input_tensors, output_tensors = keras_model_to_frozen_graph(model)

    reloaded_model = tf.keras.Sequential(
        [
            FrozenGraphLayer(
                graph_def,
                input_tensor_names=[t.name for t in input_tensors],
                output_tensor_names=[t.name for t in output_tensors],
            )
        ]
    )
    assert np.allclose(reloaded_model.predict([x1, x2]), expected_output)


def test_load_saved_model_as_layer_no_names() -> None:
    model = train_addition_model()
    x = np.random.rand(1, model.input_shape[-1])
    expected_output = model.predict(x)

    with keras_model_to_savedmodel(model) as saved_model_path:
        reloaded_model = tf.keras.Sequential([SavedModelLayer(saved_model_path)])
        assert np.allclose(reloaded_model.predict(x), expected_output)


def test_load_saved_model_as_layer_dict_input_and_output() -> None:
    model = train_named_input_and_output()
    x = np.random.rand(1, model.input_shape["named_input"][-1])
    expected_output = model.predict(x)

    with keras_model_to_savedmodel(model) as saved_model_path:
        layer = SavedModelLayer(saved_model_path)
    assert np.allclose(layer({"named_input": x})["named_output"], expected_output["named_output"])


def test_load_saved_model_as_layer_many_dict_input_and_output() -> None:
    model, input_names, output_names = train_multi_input_multi_output()
    x = {input_name: tf.constant([[i]], dtype=tf.float32) for i, input_name in enumerate(input_names)}
    expected_outputs = model.predict(x)

    with keras_model_to_savedmodel(model) as saved_model_path:
        layer = SavedModelLayer(saved_model_path)

    outputs = layer(x)

    assert all([key in outputs for key in output_names])
    for output_name in outputs.keys():
        assert np.allclose(
            outputs[output_name], expected_outputs[output_name]
        ), f"Output {output_name} did not match expected value after reloading."


def test_tensor_wrapper_layer() -> None:
    model = train_addition_model()
    x = np.random.rand(1, model.input_shape[-1])
    expected_output = model.predict(x)

    with keras_model_to_savedmodel(model) as saved_model_path:
        tensors = get_all_tensors_from_saved_model(saved_model_path)

        layer = TensorWrapperLayer(tensors[0].graph.inputs, [t for t in tensors if "my_layer" in t.name][-1])
        reloaded_model = tf.keras.Sequential([layer])
        assert np.allclose(reloaded_model.predict(x), expected_output)


def test_tensor_wrapper_layer_multiple_inputs() -> None:
    model, input_names, output_names = train_multi_input_multi_output()
    x = {input_name: tf.constant([[i]], dtype=tf.float32) for i, input_name in enumerate(input_names)}
    output_dict = model.predict(x)

    expected_outputs = [output_dict[output_name] for output_name in output_names]

    with keras_model_to_savedmodel(model) as saved_model_path:
        tensors = get_all_tensors_from_saved_model(saved_model_path)
        layer = TensorWrapperLayer(tensors[0].graph.inputs, tensors[0].graph.outputs)

    outputs = layer([value for _name, value in sorted(x.items(), key=lambda t: t[0])])  # type: ignore
    for expected, actual in zip(expected_outputs, outputs):
        assert np.allclose(actual, expected)
