import tensorflow as tf
import numpy as np
import pathlib
import argparse

from keras.applications.resnet import ResNet50

DATASET = tf.keras.datasets.cifar10  # The dataset used for training and testing
KERAS_MODEL = ResNet50(weights='imagenet')

image_size = [0, 0] # input image size
calabration_number = 0  # Number of images used for calibration

def get_dataset():
    # Load the dataset and normalize the pixel values
    (train_images, train_labels), (test_images, test_labels) = DATASET.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Select a subset of the images for calibration
    train_images = train_images[:calabration_number]
    test_images = test_images[:calabration_number]

    # Resize the images to the desired size
    train_images_resized = tf.image.resize(train_images, image_size)
    test_images_resized = tf.image.resize(test_images, image_size)
    return [train_images_resized, test_images_resized]

def representative_data_gen():
    # Generate a representative dataset for quantization
    for input_value in tf.data.Dataset.from_tensor_slices(get_dataset()[0]).batch(1).take(calabration_number):
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

def save_model(model, model_dir, model_name):
    # Save the TFLite model to disk
    tflite_models_dir = pathlib.Path(model_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_quant_file = tflite_models_dir/f"{model_name}.tflite"
    tflite_model_quant_file.write_bytes(model)

def get_argprase():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='model path')
    parser.add_argument('--cal_num', type=int, default=200,
                        help='number of sample to be calabration data')
    parser.add_argument('--dir', type=str, default="./tflite",
                        help='Directory to save the TFLite model')
    parser.add_argument('--o', type=str, default=None,
                        help='Name of the model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_argprase()
    keras_model = args.model
    calabration_number = args.cal_num
    model_directory = args.dir
    model_name = args.o

    # Define the Keras model to convert to TFLite
    if args.model == None:
        model = KERAS_MODEL
    else:
        model = tf.keras.models.load_model(args.model)
    
    if model_name == None:
        model_name = keras_model.split("/")[-1].split(".")[0]
    
    input_shape = model.layers[0].input_shape[0]
    image_size = (input_shape[1], input_shape[2])

    # Convert the Keras model to a quantized TFLite model
    quant_model = convert_to_tflite(model)
    # Save the TFLite model to disk
    save_model(quant_model, model_directory, model_name)
