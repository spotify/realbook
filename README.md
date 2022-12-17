[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/realbook)
![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)
![Lifecycle](https://img.shields.io/badge/lifecycle-production-1ed760.svg)


# realbook 📒

Realbook is a Python library for easier training of audio deep learning models with [Tensorflow](https://tensorflow.org) made by Spotify's [Spotify's Audio Intelligence Lab](https://research.atspotify.com/audio-intelligence/). Realbook provides callbacks (e.g., spectrogram visualization) and well-tested [Keras layers](https://keras.io/api/layers/) (e.g., STFT, ISTFT, magnitude spectrogram) that we often use when training. These functions have helped standardized consistency across all of our models we and hope realbook will do the same for the open source community.

# Notable Features

Below are a few highlights of what we have written so far.

## Keras Layers

- `FrozenGraphLayer` - Allows you to use a TF V1 graph as a Keras layer.
- `CQT` - Constant-Q transform layers ported from [nnAudio](https://kinwaicheuk.github.io/nnAudio/index.html).
- `Stft`, `Istft`, `MelSpectrogram`, `Spectrogram`, `Magnitude`, `Phase` and `MagnitudeToDecibel` - Layers that perform common audio feature preprocessing. All checked for correctness against [librosa](https://librosa.org/).

## Callbacks

- `Spectrogram visualization` - Allows you to write spectrogram output layers to TensorBoard.
- `Training Speed` - Allows you to visualize on TensorBoard how fast each epoch of training is taking.
- `Utilization` - Allows you to plot on TensorBoard CPU, CPU Memory, GPU and GPU Memory utilization as you train.

## Installation

```shell
pip install realbook

# Or, if using any TensorBoard-related callbacks, install additional dependencies:
pip install realbook[tensorboard]
```

Then, in your code:

```python
import realbook.callbacks.spectrogram_visualization # a nifty TensorBoard callback
```

# Example

## A Binary Classifier With Audio Input

Let's use realbook to train a binary classifier that takes in audio, converts the audio to a spectrogram and then 
runs the spectorgram output through two trainable Dense layers.

```python3
import tensorflow as tf
from realbook.layers.signal import STFT

train_ds = tf.data.TFRecordDataset(training_filenames)
val_ds = tf.data.TFRecordDataset(validation_filenames)

# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((22050,)),
    signal.Stft(fft_length=1024, hop_length=512), 1_266_384),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(2),
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Now train!
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```

## A Binary Classifier With Audio Input and CPU Memory Utilization Measurement

Below is the previous binary classifier example, but we're now going to add a realbook
callback to the model's callback list.

```python3
import tensorflow as tf
from realbook.layers.signal import STFT
from realbook.callbacks.utilization import MemoryUtilizationCallback

train_ds = tf.data.TFRecordDataset(training_filenames)
val_ds = tf.data.TFRecordDataset(validation_filenames)

# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((22050,)),
    signal.Stft(fft_length=1024, hop_length=512), 1_266_384),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(2),
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

writer = tf.summary.create_file_writer(tensorboard_output_location)

# Now train!
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[MemoryUtilizationCallback(writer))]
)
```

## Metrics

Realbook contains a number of layers that convert audio data (i.e.: waveforms)
into various spectral representations (i.e.: spectrograms). For convenience, the amount of memory
required for the most commonly used layers is provided below.

Using an FFT length of 1024 and a hop length of 512, processing one second of audio at 22050Hz requires:

| Layer                                                   | Memory High Watermark |
| ------------------------------------------------------- | --------------------- |
| `realbook.layers.signal.STFT`                           | 1,266,384 bytes       |
| `realbook.layers.signal.Spectrogram`                    | 1,264,324 bytes       |
| `realbook.layers.signal.MelSpectrogram`                 | 1,262,784 bytes       |
| `realbook.layers.nnaudio.CQT`                           | 1,047,216 bytes       |

### GPU Utilization Callbacks

GPU resource utilization callbacks are included as part of the tensorboard extra installable.
These callbacks expect the program `nvidia-smi` to be installed. A program which is only
available on Linux. For example, on Ubuntu, you can install this program with

```shell
apt-get update
apt-get install -y nvidia-utils-<CUDA version number>
```

Where CUDA version number is the version of CUDA installed on your machine e.g. 450.

## Setup Development (of `realbook`)

Create a new virtual environment with for your supported Python version and clone this repo. Within that virtualenv:

```shell
$ pip install -e .[dev]
```

This will install development dependencies, followed by installing this package itself as ["editable"](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs).

## Run Tests

Tests can be invoked in two ways: `pytest` and `tox`.

### Run tests via `pytest`

This must be done within the virtualenv. Note that `pytest` will automatically pick up the config set in `tox.ini`. Comment it out if you want to skip coverage and/or ignore verbosity while iterating.

```sh
# for all tests
(env) $ pytest tests/

# for one module of tests
(env) $ pytest tests/layers/signal.py

# for one specific test
(env) $ pytest tests/layers/signal.py::test_stft
```

More info about pytest can be found [here](https://docs.pytest.org/en/latest/).

### Run tests via `tox`

`tox` should be run **outside** of a virtualenv. This is because `tox` will create separate virtual environments for each test environment. A test environment could be based on python versions, or could be specific to documentation, or whatever else. See `tox.ini` as an example for mulutiple different test environments including: running tests for Python, linting, and checking `MANIFEST.in` to assert a proper setup.

```sh
# run all environments
$ tox

# run a specific environment
$ tox -e check-formatting
$ tox -e py38
```

### Formatting files

Before committing PR's please format your files using tox as some of the formatting options realboook uses is different than the defaults of the [Black](https://black.readthedocs.io/en/stable/) formatter:

```sh
tox -e format
```

See [tox's documentation](https://tox.readthedocs.io/en/latest/) for more information.

## Copyright and License
realbook is Copyright 2022 Spotify AB.

This software is licensed under the Apache License, Version 2.0 (the "Apache License"). You may choose either license to govern your use of this software only upon the condition that you accept all of the terms of either the Apache License.

You may obtain a copy of the Apache License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the Apache License or the GPL License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License for the specific language governing permissions and limitations under the Apache License
