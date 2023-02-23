import DataPreprocessing as ds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
import os



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            # print('Reached 99% accuracy, so stop training')
            self.model.stop_training = True


def split(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle=True)

    return X_train, X_test, X_val, y_train, y_test, y_val


def CNN(SIZE):
    pretrained_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
    #
    # Freezing the weights of layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # printing model summary
    pretrained_model.summary()

    # Choose `mixed_7` as the last layer of your base model
    last_layer = pretrained_model.get_layer('mixed7')
    # print(last_layer.output_shape)
    last_layer_output = last_layer.output

    # Adding DNN to the prefetched model
    DNN = tf.keras.layers.Flatten()(last_layer_output)
    DNN = tf.keras.layers.Dense(512, activation='relu')(DNN)
    DNN = tf.keras.layers.Dropout(0.2)(DNN)
    DNN = tf.keras.layers.Dense(3, activation='softmax')(DNN)

    model = tf.keras.Model(pretrained_model.input, DNN)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    path = 'Dataset/Data for test/'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, path)
    SIZE = 400
    epoch = 20
    batchsize = 4
    X, Y = ds.Read_Dataset(filepath, SIZE)
    print(X)
    print(X.shape)
    print(Y)
    print(Y.shape)

    X_train, X_test, X_val, y_train, y_test, y_val = split(X, Y)


    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0, rotation_range=40, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

    train_set = train_generator.flow(X_train, y_train, batch_size=batchsize)

    val_set = train_generator.flow(X_val, y_val, batch_size=batchsize)

    test_set = test_generator.flow(X_test, y_test, batch_size=batchsize)

    model = CNN(SIZE)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, min_delta=0.005, patience=10)

    model.fit(train_set, epochs=epoch, validation_data=val_set, callbacks=[myCallback(), earlystopping])

    # fetch the val accuracy
    _, val_accuracy = model.evaluate(val_set, verbose=1)

    # fetch the testing accuracy
    _, test_accuracy = model.evaluate(test_set, verbose=1)

    # Predict the test dataset
    y_predicted = model.predict(test_set)
    y_predicted = np.argmax(y_predicted, axis=1)


    # Calculate the scores for each fold
    precision = precision_score(y_test, y_predicted, average='macro')

    recall = recall_score(y_test, y_predicted, average='macro')

    f1score = f1_score(y_test, y_predicted, average='macro')

    conf_mat = confusion_matrix(y_test, y_predicted)

    # Print the scores for each fold
    print("validation Accuracy = {}".format(val_accuracy * 100))
    print("Testing Accuracy = {}".format(test_accuracy * 100))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 Score = {}".format(f1score))
    print("Confusion Matrix = {}".format(conf_mat))

    model.save_weights('./../api/final_model_w.h5')


if __name__ == '__main__':
    main()