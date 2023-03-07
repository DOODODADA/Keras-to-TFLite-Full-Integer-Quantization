import tensorflow as tf
import tensorflow.keras as keras
import pathlib
import numpy as np
import argparse

from keras_cv.models.resnet_v1 import ResNet18
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

DATASET = tf.keras.datasets.cifar10
BATCHSIZE = 32
EPOCH = 1
KERAS_DIR = "./keras_model"
KERAS_NAME = "resnet18"
INPUT_SHAPE = (32, 32, 3)
CLASSES = 10

(train_images, train_labels), (test_images, test_labels) = DATASET.load_data()
train_images = (train_images.astype(np.float32) / 255.0)
test_images = (test_images.astype(np.float32) / 255.0)

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
               epochs=EPOCH, validation_data=(X_val, Y_val),callbacks=[es])
    
    models_dir = pathlib.Path(KERAS_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    model.save(f"{KERAS_DIR}/{KERAS_NAME}.h5")
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
    batch_size = BATCHSIZE

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(batch_size)

    epochs = EPOCH
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                x_batch_train_resize = tf.image.resize(x_batch_train, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
                
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
    model.save(f"{KERAS_DIR}/{KERAS_NAME}_scratch.h5")
    return model

if __name__ == "__main__":
    model = ResNet18(include_top=True, include_rescaling=False, classes=CLASSES, input_shape=INPUT_SHAPE)
    train_model(model)
    train_model_scratch(model)