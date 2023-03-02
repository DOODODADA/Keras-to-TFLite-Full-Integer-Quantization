# Full Integer Quantization of a Keras Model
This repository contains code to convert a Keras model to a fully integer quantized TensorFlow Lite model. Integer quantization can significantly reduce the size of a model and improve its performance on edge devices.

## Installation
To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```
## Usage
To convert a Keras model to a fully integer quantized TensorFlow Lite model, follow these steps:
1. Open the `keras2tflite.py` file.
2. Modify the following variables to suit your needs:
* `DATASET`: The dataset used for training and testing.
* `IMAGE_SIZE`: The size of the input images for the model.
* `CALIBRATION_NUM`: The number of images used for calibration.
* `KERAS_MODEL`: The Keras model to be converted.
* `MODEL_DIR`: The directory to save the TFLite model.
* `MODEL_NAME`: The name of the TFLite model.
3. Save the `keras2tflite.py` file.
4. Run the following command:

```bash
python keras2tflite.py
```

The converted TFLite model will be saved in the specified `MODEL_DIR` directory with the specified `MODEL_NAME`.
