import tensorflow as tf
import numpy as np
import keras
import cv2
import pathlib

from keras.applications.resnet import ResNet50
from keras_cv.models.resnet_v1 import ResNet18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import utils


DATASET = tf.keras.datasets.cifar10  # The dataset used for training and testing
IMAGE_SIZE = (224, 224, 3)
CALIBRATION_NUM = 200  # Number of images used for calibration

MODEL_DIR = "./tflite_model"  # Directory to save the TFLite model
MODEL_NAME = "resnet18_cifar10"

TRAIN = True
EPOCH = 5
BATCHSIZE = 256
KERAS_DIR = "./keras_model"

(train_images, train_labels), (test_images, test_labels) = DATASET.load_data()

def representative_data_gen():
    collabration_data = tf.image.resize(train_images[:CALIBRATION_NUM], (224, 224))
    # Generate a representative dataset for quantization
    for input_value in tf.data.Dataset.from_tensor_slices(collabration_data).batch(1).take(CALIBRATION_NUM):
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

def evaluate_model(tflite_file, model_type):
    global test_images
    global test_labels

    predictions = []
    batch_size = 256
    test_image_indices = range(test_images.shape[0])
    batch_num = (len(test_image_indices) + batch_size - 1)//batch_size
    test_image_indices_batches = [[]]

    for batch in range(batch_num):
        test_image_indices_batches.append(test_image_indices[batch*batch_size:(batch+1)*batch_size])

    for test_image_indices in test_image_indices_batches:
        prediction = utils.run_tflite_model(tflite_file, test_image_indices, test_images)
        predictions = np.concatenate((predictions, prediction), axis=0).astype(int)

    accuracy = (np.sum(test_labels.flatten() == predictions) * 100) / test_images.shape[0]

    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
        model_type, accuracy, test_images.shape[0]))

def save_model(model, model_name):
    # Save the TFLite model to disk
    tflite_models_dir = pathlib.Path(MODEL_DIR)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_quant_file = tflite_models_dir/f"{model_name}.tflite"
    tflite_model_quant_file.write_bytes(model)

def train_model(model):
    X_train, Y_train, X_test, Y_test = train_images, train_labels, test_images, test_labels
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

    encoder = OneHotEncoder()
    encoder.fit(Y_train)

    Y_train = encoder.transform(Y_train).toarray()
    Y_test = encoder.transform(Y_test).toarray()
    Y_val =  encoder.transform(Y_val).toarray()

    aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                             height_shift_range=0.05)
    aug.fit(X_train)

    model.compile(optimizer = "adam",loss='categorical_crossentropy', metrics=["accuracy"]) 
    model.summary()
    es = EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_acc")

    STEPS = len(X_train) / BATCHSIZE
    model.fit(aug.flow(X_train,Y_train,batch_size = BATCHSIZE), steps_per_epoch=STEPS, batch_size = BATCHSIZE,\
               epochs=EPOCH, validation_data=(X_val, Y_val),callbacks=[es], preprocessing_function=resize_data)
    
    models_dir = pathlib.Path(KERAS_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    model.save(f"{KERAS_DIR}/resnet18.h5")
    ModelLoss, ModelAccuracy = model.evaluate(X_test, Y_test)

    print('Model Accuracy is {}'.format(ModelAccuracy))
    return model

def train_model_scratch(model):
    X_train, Y_train, X_test, Y_test = train_images, train_labels, test_images, test_labels
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

    encoder = OneHotEncoder()
    encoder.fit(Y_train)

    Y_train = encoder.transform(Y_train).toarray()
    Y_test = encoder.transform(Y_test).toarray()
    Y_val =  encoder.transform(Y_val).toarray()
    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

    # Prepare the training dataset.
    batch_size = 32

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(batch_size)

    epochs = 100
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                x_batch_train_resize = tf.image.resize(x_batch_train, (224, 224))
                
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train_resize, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.

                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
    
    models_dir = pathlib.Path(KERAS_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    model.save(f"{KERAS_DIR}/resnet18_scratch.h5")
    return model

if __name__ == "__main__":
    train_images = ((train_images).astype(np.float32) / 255.0)
    test_images = (test_images.astype(np.float32) / 255.0)
    # Define the Keras model to convert to TFLite
    model = ResNet18(include_top=True, include_rescaling=False, classes=10, input_shape=(224,224,3))
    # model = keras.models.load_model("resnet18.h5")

    # Train model
    if TRAIN:
        model = train_model_scratch(model)

    # Convert the Keras model to a quantized TFLite model 
    quant_model = convert_to_tflite(model)
    # Save the TFLite model to disk
    save_model(quant_model, MODEL_NAME)

    evaluate_model(f"{MODEL_DIR}/{MODEL_NAME}.tflite", model_type="Quantized")
