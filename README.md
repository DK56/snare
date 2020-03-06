# SNARE

[![PyPI version](https://badge.fury.io/py/snare-ml.svg)](https://badge.fury.io/py/snare-ml) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

SNARE is a new optimization approach to reduce trained neural networks in size. This implementation uses a sequential Keras model with TensorFlow backend as an input and iteratively performs score-based pruning and re-training. SNARE ouputs a smaller Keras-compatible DNN that is optimized to achieve a similar accuracy as the original network. 

## Installation

SNARE requires [TensorFlow] v1.15.2 to run. To install SNARE and all dependencies use:

[TensorFlow]: <https://tensorflow.org>

```sh
$ pip3 install snare-ml
```

## License
MIT
