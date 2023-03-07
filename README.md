# Full Integer Quantization of a Keras Model
This repository contains code to convert a Keras model to a fully integer quantized TensorFlow Lite model. Integer quantization can significantly reduce the size of a model and improve its performance on edge devices.

## Installation
To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```
## Usage
To convert a Keras model to a fully integer quantized TensorFlow Lite model, run the following command:
```bash
python keras2tflite.py \
    --model /path/to/model.h5 \
    --cal_num 200 \
    --dir /path/to/output/folder \
    --o model_name
```
### Arguments
The following arguments are available:
* `--model`:Path to the Keras model file.
* `--cal_num`:Number of calibration examples to use for quantization.
* `--dir`:Path to the output directory.
* `--o` :The name of the output file.

You can also run `python3 keras2tflite.py --help` to see a list of all available arguments and their descriptions.

