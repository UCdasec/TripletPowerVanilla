#!/usr/bin/python3
import os
import sys
import argparse
import pdb
import h5py

import tensorflow as tf
import numpy as np
from collections import defaultdict
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tools.checking_tool as checking_tool
import tools.process_data as process_data
import tools.model_zoo as model_zoo


# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx-min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = plaintext_list[p][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            # AES_Sbox[plaintext ^ i]
            tmp_label = process_data.aes_internal(plaintext, i)
            proba = predictions[p][tmp_label]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                '''
                min_proba = 0.000000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba**2)
                '''

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)


def full_ranks(model, dataset, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = key[target_byte]
    # Check for overflow
    if max_trace_idx > dataset.shape[0]:
        raise ValueError("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]
    # Sanity check
    if input_layer_shape[1] != dataset.shape[1]:
        raise ValueError("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(dataset[0, :])))

    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        print('# This is a MLP')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
    elif len(input_layer_shape) == 3:
        print('# This is a CNN: reshape the data')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        raise ValueError("Error: model input shape length %d is not expected ..." % len(input_layer_shape))

    # Predict our probabilities
    predictions = model.predict(input_data, batch_size=200, verbose=0)

    index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], plaintext_attack[t-rank_step:t], real_key, t-rank_step, t, key_bytes_proba, target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def get_the_labels(textins, key, target_byte):
    labels = []
    for i in range(textins.shape[0]):
        text_i = textins[i]
        label = process_data.aes_internal(text_i[target_byte], key[target_byte])
        labels.append(label)

    labels = np.array(labels)
    return labels


def load_data(opts):
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    target_byte = opts.target_byte
    network_type = opts.network_type
    val_data_path = opts.input
    shifted = opts.shifted

    if not os.path.isfile(val_data_path):
        raise ValueError('file did not find: {}'.format(val_data_path))
    clsNum = 9 if 'hw_model' == network_type else 256
    val_data_whole_pack = np.load(val_data_path)

    attack_window = opts.attack_window
    if attack_window:
        tmp = opts.attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data_shifted(val_data_whole_pack, target_byte, network_type, shifted, attack_window)
    else:
        traces, labels, text_in, key, inp_shape = process_data.process_raw_data(val_data_whole_pack, target_byte, network_type, attack_window)

    labels = to_categorical(labels, clsNum)
    return traces, labels, text_in, key, inp_shape


def plot_figure(x, y, model_file_name, dataset_name, fig_save_name, testType):
    plt.title('Performance of ' + model_file_name + ' against ' + dataset_name + ' testType ' + testType)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # checking model file existence
    model_file = opts.model_file

    # Load model
    model = checking_tool.load_best_model(model_file)
    model.summary()

    # Load profiling and attack data and metadata from the ASCAD database
    # val_traces, val_label, val_textin, key
    X_attack, Y_attack, plaintext_attack, key, inp_shape = load_data(opts)

    '''
    print('shuffle the data and then calculate the rank')
    X_attack, Y_attack, plaintext_attack = loadData.shuffleData(X_attack, Y_attack, plaintext_attack)
    print('shuffle data has been done!')
    '''

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_attack = process_data.sanity_check(input_layer_shape, X_attack)

    # run the accuracy test
    score, acc = model.evaluate(Reshaped_X_attack, Y_attack, verbose=opts.verbose)
    print('test acc is: {:f}'.format(acc))

    # We test the rank over traces of the Attack dataset, with a step of 10 traces
    print('start computing rank value...')
    min_trace_idx = 0
    max_trace_idx = 2000
    rank_step = 1
    target_byte = opts.target_byte
    ranks = full_ranks(model, X_attack, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step)

    # We plot the results
    # f_ranks[i] = [t, real_key_rank]
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]

    dataset_name = os.path.basename(opts.input)
    os.makedirs(opts.output, exist_ok=True)
    testType = os.path.basename(opts.input).split('.')[0]
    testType = testType.split('_')[1]
    device_name = os.path.basename(os.path.dirname(opts.input))
    fig_save_dir = os.path.join(opts.output, device_name, opts.network_type)
    os.makedirs(fig_save_dir, exist_ok=True)
    fig_save_name = os.path.join(fig_save_dir, str(dataset_name) + '_rank_performance_byte_{}_{}.png'.format(target_byte, testType))
    print('figure save to file: {}'.format(fig_save_name))
    model_file_name = os.path.basename(model_file).split('.')[0]

    # def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plot_figure(x, y, model_file_name, dataset_name, fig_save_name, testType)

    test_summary_path = 'test_summary.txt'
    contents = '{} --- {} --- {:f}\n\n'.format(opts.input, opts.network_type, acc)
    with open(test_summary_path, 'a') as f:
        f.write(contents)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_file', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-nt', '--network_type', default='cnn2', choices={'mlp', 'cnn', 'cnn2', 'wang'}, help='')
    parser.add_argument('-s', '--shifted', type=int, default=10, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
