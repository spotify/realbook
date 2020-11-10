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
import tensorflow as tf
from google.protobuf.json_format import (
    MessageToDict as SerializeProtobufToDict,
    ParseDict as ParseDictToProtobuf,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tensorflow.python.ops.op_selector import UnliftableError
from tensorflow.python.eager.wrap_function import WrappedFunction


def _load_concrete_function_from_graph_def(
    graph_def: tf.compat.v1.GraphDef,
    input_tensor_names: List[str],
    output_tensor_names: List[str],
) -> WrappedFunction:
    wrapped_import = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(graph_def, name=""), [])
    imported_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(imported_graph.as_graph_element, input_tensor_names),
        tf.nest.map_structure(imported_graph.as_graph_element, output_tensor_names),
    )


class FrozenGraphLayer(tf.keras.layers.Layer):
    """
    Loads a TensorFlow V1 frozen graph (.pb) file from disk as a layer that
    can be used in another model.
    """

    def __init__(
        self,
        path_or_graph_def: Union[str, Dict[Any, Any], tf.compat.v1.GraphDef],
        input_tensor_names: List[str],
        output_tensor_names: List[str],
        name: Optional[str] = None,
    ):
        if isinstance(path_or_graph_def, tf.compat.v1.GraphDef):
            self.graph_def = path_or_graph_def
        elif isinstance(path_or_graph_def, dict):
            self.graph_def = tf.compat.v1.GraphDef()
            ParseDictToProtobuf(path_or_graph_def, self.graph_def)
        else:
            self.graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_or_graph_def, "rb") as f:
                graph_str = f.read()
                self.graph_def.ParseFromString(graph_str)
            name = name or os.path.splitext(os.path.basename(path_or_graph_def))[0].replace(".", "_").replace("-", "_")

        self.input_tensor_names = input_tensor_names
        self.output_tensor_names = output_tensor_names
        self.concrete_function = _load_concrete_function_from_graph_def(
            self.graph_def,
            self.input_tensor_names,
            self.output_tensor_names,
        )

        super().__init__(name=name)

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]],
        training: Any = None,
    ) -> tf.Tensor:
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            result = self.concrete_function(*inputs)
        else:
            result = self.concrete_function(inputs)

        # If we've only got one output tensor, return that tensor on its own as the layer's output.
        # This is the usual case. (Multi-output layers don't fit nicely into `Sequential` models, for instance.)
        if len(self.output_tensor_names) == 1:
            return result[0]
        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            "path_or_graph_def": SerializeProtobufToDict(self.graph_def),
            "name": self.name,
            "input_tensor_names": self.input_tensor_names,
            "output_tensor_names": self.output_tensor_names,
        }


class SavedModelLayer(tf.keras.layers.Layer):
    """
    Loads a TensorFlow V2 SavedModel (saved_model.pb + variables/) from disk
    as a layer that can be used in another model. Use this instead of a Lambda
    layer to avoid variable capture issues and to produce a model that can be
    used correctly when re-saving a model.

    If multiple inputs (or outputs) exist, this layer will expect dictionaries
    on its input (or output):

        outputs = SavedModelLayer("./my_model/")({
            "some_input": <tensor>,
            "other_input": <tensor>
        })

        # Later:
        Dense(10)(outputs["some_output_name"])
    """

    def __init__(self, path_to_saved_model: str, name: Optional[str] = None):
        name = name or os.path.splitext(os.path.basename(path_to_saved_model))[0].replace(".", "_")
        super().__init__(name=name)
        self.model = tf.saved_model.load(path_to_saved_model)

    def call(self, _input: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        return self.model(_input)


def get_saved_model_input_tensors(saved_model_or_path: Union[tf.keras.Model, str]) -> List[tf.Tensor]:
    """
    Given a path to a SavedModel or an already loaded SavedModel,
    return a list of its input tensors. Useful for figuring out the
    names of the input tensors for use with SavedModelLayer.
    """
    if isinstance(saved_model_or_path, str):
        savedmodel = tf.saved_model.load(saved_model_or_path)
        model = savedmodel.signatures["serving_default"]
        model._backref = savedmodel  # Without this, the SavedModel will be GC'd too early
    else:
        model = saved_model_or_path
    if hasattr(model, "signatures"):
        model = model.signatures["serving_default"]

    return [tensor for tensor in model.inputs if tensor.dtype != "resource"]


def get_all_tensors_from_saved_model(saved_model_or_path: Union[tf.keras.Model, str]) -> List[tf.Tensor]:
    """
    Given a path to a SavedModel or an already loaded SavedModel,
    return a list of all of its tensors. Useful for figuring out the
    names of intermediate tensors for use with SavedModelLayer.

    To extract the output of a given Keras layer (which isn't stored in the
    SavedModel, as TensorFlow SavedModels don't acually save layer
    information), try something like:

    ```
        tensors = get_all_tensors_from_saved_model('./my-saved-model')

        layer_name = "some_named_layer"
        probable_output_of_layer = [t for t in tensors if layer_name in t.name][-1]

        # Should probably check that this is the model's input, as expected:
        probable_input_to_model = tensors[0]

        # Create a sub-graph of the loaded model from the input and output tensors you want:
        sub_graph = create_function_from_tensors(probable_input_to_model, probable_output_of_layer)

        # Use that sub-graph wherever you'd like:
        my_model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda input_tensor: sub_graph(input_tensor)),
        ])

        # (Note that if you're extracting multiple inputs, your lambda function
        # must pass each input to sub_graph as a separate argument,
        # i.e.: `lambda input_tensors: sub_graph(*input_tensors)`.)
    ```

    """
    if isinstance(saved_model_or_path, str):
        savedmodel = tf.saved_model.load(saved_model_or_path)
        model = savedmodel.signatures["serving_default"]
        model._backref = savedmodel  # Without this, the SavedModel will be GC'd too early
    else:
        model = saved_model_or_path
    if hasattr(model, "signatures"):
        model = model.signatures["serving_default"]
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(model)

    all_inputs_and_outputs: List[tf.Tensor] = sum(
        [list(op.inputs) + list(op.outputs) for op in frozen_func.graph.get_operations()],
        [],
    )

    # Using this to find unique tensors instead of set() because tf.Tensor is not hashable.
    seen_names = set()
    res = []
    for obj in all_inputs_and_outputs:
        if obj.name not in seen_names:
            seen_names.add(obj.name)
            res.append(obj)
    return res


def get_saved_model_output_tensors(saved_model_or_path: Union[tf.keras.Model, str]) -> List[tf.Tensor]:
    """
    Given a path to a SavedModel or an already loaded SavedModel,
    return a list of its output tensors. Useful for figuring out the
    names of the output tensors for use with SavedModelLayer.
    """
    if isinstance(saved_model_or_path, str):
        savedmodel = tf.saved_model.load(saved_model_or_path)
        model = savedmodel.signatures["serving_default"]
        model._backref = savedmodel  # Without this, the SavedModel will be GC'd too early
    else:
        model = saved_model_or_path
    if hasattr(model, "signatures"):
        model = model.signatures["serving_default"]
    return [tensor for tensor in model.outputs if tensor.dtype != "resource"]


def create_function_from_tensors(
    input_tensors: Union[tf.Tensor, List[tf.Tensor]],
    output_tensors: Union[tf.Tensor, List[tf.Tensor]],
) -> WrappedFunction:
    """
    Given two lists of tensors (input and output), this method will return a tf.function
    that can be used to feed inputs to the input tensors and take outputs from the output tensors.

    Example usage:

    ```
        tensors = get_all_tensors_from_saved_model('./my-saved-model')

        layer_name = "some_named_layer"
        probable_output_of_layer = [t for t in tensors if layer_name in t.name][-1]

        # Should probably check that this is the model's input, as expected:
        probable_input_to_model = tensors[0]

        # Create a sub-graph of the loaded model from the input and output tensors you want:
        sub_graph = create_function_from_tensors(probable_input_to_model, probable_output_of_layer)

        # Use that sub-graph wherever you'd like:
        my_model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda input_tensor: sub_graph(input_tensor)),
        ])

        # (Note that if you're extracting multiple inputs, your lambda function
        # must pass each input to sub_graph as a separate argument,
        # i.e.: `lambda input_tensors: sub_graph(*input_tensors)`.)
    ```

    """
    if isinstance(input_tensors, tf.Tensor):
        input_tensors = [input_tensors]
    if isinstance(output_tensors, tf.Tensor):
        output_tensors = [output_tensors]

    if len(set([t.graph for t in (input_tensors + output_tensors)])) > 1:
        raise ValueError("All input and output tensors must be from the same graph.")

    graph = next(iter(input_tensors + output_tensors)).graph

    graph_input_names = [t.name for t in graph.inputs]

    try:
        return _load_concrete_function_from_graph_def(
            graph.as_graph_def(),
            [t.name for t in input_tensors],
            [t.name for t in output_tensors],
        )
    except UnliftableError as e:
        if len(graph_input_names) > len(input_tensors):
            raise ValueError(
                f"Could not create a function from input tensors {[t.name for t in input_tensors]} and output tensors"
                f" {[t.name for t in output_tensors]}. The graph that contains these tensors expects"
                f" {len(graph_input_names)} input(s), but only {len(input_tensors)} tensor(s) were passed - the outputs"
                " requested may depend on the missing inputs.\n\nUnderlying TensorFlow error:"
                f" {str(e).split(', e.g.:')[0]}."
            )
        else:
            raise


class TensorWrapperLayer(tf.keras.layers.Layer):
    """
    A really small layer class that allows passing input to an arbitrary input tensor
    and fetching input from an arbitrary output tensor. Useful for extracting parts of SavedModels.
    """

    def __init__(
        self,
        input_tensors: Union[tf.Tensor, List[tf.Tensor]],
        output_tensors: Union[tf.Tensor, List[tf.Tensor]],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.func = create_function_from_tensors(input_tensors, output_tensors)

    def call(self, _input: Union[tf.Tensor, List[tf.Tensor]]) -> Union[tf.Tensor, List[tf.Tensor]]:
        if isinstance(_input, dict):
            raise NotImplementedError("Input to TensorWrapperLayer must be a single tensor or list of tensors.")
        elif isinstance(_input, list):
            return self.func(*_input)
        else:
            return self.func(_input)
