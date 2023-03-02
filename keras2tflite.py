import tensorflow as tf
import numpy as np
from keras.applications.resnet import ResNet50
import pathlib

DATASET = tf.keras.datasets.cifar10  # The dataset used for training and testing
IMAGE_SIZE = [224, 224]
CALIBRATION_NUM = 200  # Number of images used for calibration

KERAS_MODEL = ResNet50(weights='imagenet')

MODEL_DIR = "./tflite_model_2"  # Directory to save the TFLite model
MODEL_NAME = "resnet18"

def get_dataset():
    # Load the dataset and normalize the pixel values
    (train_images, train_labels), (test_images, test_labels) = DATASET.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Select a subset of the images for calibration
    train_images = train_images[:CALIBRATION_NUM]
    test_images = test_images[:CALIBRATION_NUM]

    # Resize the images to the desired size
    train_images_resized = tf.image.resize(train_images, IMAGE_SIZE)
    test_images_resized = tf.image.resize(test_images, IMAGE_SIZE)
    return [train_images_resized, test_images_resized]

def representative_data_gen():
    # Generate a representative dataset for quantization
    for input_value in tf.data.Dataset.from_tensor_slices(get_dataset()[0]).batch(1).take(CALIBRATION_NUM):
        yield [input_value]

def convert_to_tflite(model):
    # Convert the Keras model to a quantized TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    return tflite_model_quant

def save_model(model, model_name):
    # Save the TFLite model to disk
    tflite_models_dir = pathlib.Path(MODEL_DIR)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_quant_file = tflite_models_dir/f"{model_name}.tflite"
    tflite_model_quant_file.write_bytes(model)

if __name__ == "__main__":
    # Define the Keras model to convert to TFLite
    model = KERAS_MODEL
    # Convert the Keras model to a quantized TFLite model
    quant_model = convert_to_tflite(model)
    # Save the TFLite model to disk
    save_model(quant_model, MODEL_NAME)
