#!/usr/bin python3.6
import os
import sys
import argparse
import pdb
import h5py

import tensorflow as tf
import numpy as np
from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import checking_tool
import process_data
import model_zoo


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_sca_model(model_file):
    check_file_exists(model_file)
    try:
        model = load_model(model_file)
    except:
        raise ValueError("Error: can't load Keras model file {}".format(model_file))
    return model


def load_training_data(opts):
    '''data loading function'''
    target_byte = opts.target_byte
    network_type == opts.network_type
    # start_idx and end_idx will be include in the dataset now
    # start_idx, end_idx = opts.start_idx, opts.end_idx

    data_path = opts.input
    whole_pack = np.load(data_path)
    shifted = opts.shifted

    attack_window = opts.attack_window
    if attack_window:
        tmp = attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data_shifted(whole_pack, target_byte, network_type, shifted, attack_window)
    else:
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data(whole_pack, target_byte, network_type, attack_window)
    if opts.max_trace_num:
        traces = traces[:opts.max_trace_num]
        labels = labels[:opts.max_trace_num]
    print('training with {:d} traces'.format(opts.max_trace_num))
    return traces, labels, inp_shape


# Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, verbose=False):
    # check modelDir existence
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    log_dir = "logs/train_id_fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpointer = ModelCheckpoint(save_file_name, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    callbacks = [checkpointer, earlyStopper]

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_profiling = process_data.sanity_check(input_layer_shape, X_profiling)

    clsNum = len(set(Y_profiling))
    Y_profiling = to_categorical(Y_profiling, clsNum)
    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_split=0.1, batch_size=batch_size,
                        verbose=verbose, epochs=epochs,
                        shuffle=True, callbacks=callbacks)

    print('model save to path: {}'.format(save_file_name))
    return history


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-nt', '--network_type', choices={'mlp', 'cnn', 'cnn2', 'wang', 'hw_model'}, help='')
    parser.add_argument('-s', '--shifted', type=int, default=0, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-mtn', '--max_trace_num', type=int, default=0, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # get the params
    network_type = opts.network_type
    resDir = opts.output
    verbose = opts.verbose
    target_byte = opts.target_byte

    # make resDir and modelDir
    modelDir = os.path.join(resDir, '{}_dir'.format(network_type))
    os.makedirs(modelDir, exist_ok=True)

    dataset_name = os.path.basename(os.path.dirname(opts.input))
    model_save_file = os.path.join(modelDir, 'best_model_{}_dataset_{}_targetbyte_{}.hdf5'.format(network_type, dataset_name, target_byte))

    # set all the params
    epochs = 100
    batch_size = 100

    # get the data and model
    #load traces
    X_profiling, Y_profiling, input_shape = load_training_data(opts)
    print('trace data shape is: ', X_profiling.shape)

    inp, embedding = model_zoo.cnn_best2(input_shape, emb_size=256, classification=False)
    out = Dense(256, activation='softmax', name='predictions')(embedding)
    best_model = Model(inp, out, name='cnn_best2')
    optimizer = RMSprop(lr=0.00001)
    best_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    best_model.summary()

    if 'hw_model' == network_type:
        print('now train dnn model for HW leakage model...')
    else:
        print('now train dnn model for ID leakage model...')
    history = train_model(X_profiling, Y_profiling, best_model, model_save_file, epochs, batch_size, verbose)
