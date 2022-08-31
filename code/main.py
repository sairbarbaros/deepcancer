import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np

path = "brain_tumor_dataset/"
training_set = tf.keras.preprocessing.image_dataset_from_directory(path,
                                             shuffle=True,
                                             batch_size=64,
                                             image_size=(160,160),
                                             validation_split=0.1,
                                             subset='training',
                                             seed=34)
test_set = tf.keras.preprocessing.image_dataset_from_directory(path,
                                             shuffle=True,
                                             batch_size=64,
                                             image_size=(160,160),
                                             validation_split=0.16,
                                             subset='validation',
                                             seed=34)


def data_augmenter():
    """ augment your data for making your data less prone to overfitting.
    :returns: data augmentation model
    """

    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"))
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.34))
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomCrop(160,160))
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomContrast(0.7, 1.2))
    
    return data_augmentation

def tumor_identifier(image_shape=(160,160), data_augmentation=data_augmenter()):
    """ create a model for your tumor identification project. you can also use other applications of keras.
    :param image_shape: shape of the input images
    :type image_shape: tuple
    :param data
    """

    input_shape = image_shape + (3,)

    base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape) 
    
    x = data_augmentation(inputs) 
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    
    x = base_model(x, training=False) 
    
    x = tf.keras.layers.GlobalMaxPooling2D()(x) 
    x = tf.keras.layers.Dropout(0.34)(x)
        
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    

    return model

brain_model = tumor_identifier((160,160), data_augmenter())

brain_model = tumor_identifier((160,160), data_augmenter())

brain_model.summary()

brain_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,)

history = brain_model.fit(
        training_set,
        validation_data=test_set,
        epochs=300, 
        shuffle=True, 
        verbose=True,
        callbacks=[early_stopping])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, label='training acc')
plt.plot(epochs, val_acc, label='validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy-%')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()

train_result = brain_model.evaluate(training_set)
val_result = brain_model.evaluate(test_set)
