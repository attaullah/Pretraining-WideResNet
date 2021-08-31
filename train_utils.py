import os
import numpy as np
from utils import shallow_clf_accuracy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from wrn import wide_residual_network
from Triplet import triplet_loss


def get_model(arch, size, channels=3, num_classes=1000, lr=1e-3, loss_type="cross-entropy", es=64, dropout_rate=0.2):
    """
    This function creates and compiles Wide Residual Network model.

    :param arch: Name of WRN-d-w, d: being depth and w: width of model
    :param size: size, i.e., width and height of input image
    :param channels: number of channels in image
    :param num_classes: default set to 1000 for ImageNet
    :param lr: learning rate
    :param loss_type: either cross-entropy or triplet. default is cross-entropy
    :param es: embeddings size
    :param dropout_rate: dropout rate
    :return: tf.keras model compiled according to given loss_type
    """
    input_shape = (size, size, channels)
    dw = arch.split('-')[1:]  # WRN-d-w, d:  depth and w: width
    conv_base = wide_residual_network(depth=int(dw[0]), width=int(dw[1]), input_shape=input_shape)
    model = models.Sequential()
    model.add(conv_base)
    # add additional layers
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    model.add(layers.Dense(es, activation=None, name="fc1"))
    metrics = []
    if loss_type == "triplet":
        model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2-normalisation"))  # l2-normalisation
        loss = triplet_loss
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name="fc_out"))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['acc']

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.build((None, size, size, channels))
    model.summary()
    return model, conv_base


def get_accuracy_from_embeddings(model, ds_train, ds_test, labelling="lda"):
    """
    Computes accuracy using  embeddings.
    :param model: tf.keras model
    :param ds_train: training dataset object
    :param ds_test: test dataset object
    :param labelling: classifier used for labelling
    :return: accuracy
    """
    # get labels of train and test examples
    y = np.concatenate([y for x, y in ds_train], axis=0)
    y_test = np.concatenate([y for x, y in ds_test], axis=0)
    # get embeddings
    train_embeddings = model.predict(ds_train)
    test_embeddings = model.predict(ds_test)

    _, accuracy = shallow_clf_accuracy(train_embeddings, y, test_embeddings, y_test, labelling)
    return accuracy


def get_accuracy(model, ds_train, ds_test, loss_type="", labelling="lda"):
    """
    Calculates either supervised accuracy or shallow classifier accuracy from dataset objects
    :param model: t.keras model object
    :param ds_train: training dataset object
    :param ds_test: test dataset object
    :param loss_type: either cross-entropy or triplet
    :param labelling: labelling algorithm either KNN or LDA for triplet loss
    :return: supervised or shallow classifier accuracy based on
    """
    if loss_type == "triplet":
        accuracy = get_accuracy_from_embeddings(model, ds_train, ds_test, labelling)
    else:
        accuracy = model.evaluate(ds_test, verbose=0)[1]

    return np.round(100.*accuracy, 2)


def start_training(model, train_ds, epochs=50, batch_size=128):
    """

    :param model: tf.keras model
    :param train_ds: training data set object
    :param epochs: number of epochs
    :param batch_size: size of mini-batch
    """
    model.fit(train_ds, epochs=epochs, batch_size=batch_size)


def dump_weights(model, base_model,  save_str="", include_top=False):

    """
    Save model weights.
    :param base_model: base model without additional layers
    :param model: tf.keras model
    :param save_str: name of output weights file
    :param include_top: include the last layer of dense layer
    """
    if include_top:
        model.save_weights(save_str+"-top-weights.h5")
    else:
        base_model.save_weights(save_str+"-weights.h5")
