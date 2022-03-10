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


'''
In this hamming weight test, we probability need 2 or more traces to get the real label of a trace
how we gonna do this?
Our goal is to recover the key byte
1. accuracy
    let's make it simple, the accuracy is according to the HW[sbox]

2. ranking curve
    each time we run a message, we will enumerate the all possible key byte,
    calculate its hw value and then divide its probability with the members correspond
    to its hw value and add it up, in this way we can get the ranking curve

    example
'''


# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte, HW, hw_mapping):
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
            tmp_label = HW[process_data.aes_internal(plaintext, i)]
            tmp_hw_list = hw_mapping[tmp_label]
            proba = predictions[p][tmp_label]
            if proba != 0:
                proba_log_share = np.log(proba) / len(tmp_hw_list)
                for elem in tmp_hw_list:
                    key_bytes_proba[elem] += proba_log_share
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                min_proba_log_share =  np.log(min_proba**2) / len(tmp_hw_list)
                for elem in tmp_hw_list:
                    key_bytes_proba[elem] += min_proba_log_share
                '''
                min_proba = 0.000000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba**2)
                '''

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = process_data.calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


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
    hw_mapping = create_hw_label_mapping()
    HW = process_data.get_HW()

    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], plaintext_attack[t-rank_step:t], real_key, t-rank_step, t, key_bytes_proba, target_byte, HW, hw_mapping)
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
    shifted = opts.shifted
    network_type = 'hw_model'

    val_data_path = opts.input
    if not os.path.isfile(val_data_path):
        raise ValueError('file did not find: {}'.format(val_data_path))

    val_data_whole_pack = np.load(val_data_path)
    # note here we are using hw model so the label have only 9 classes

    attack_window = opts.attack_window
    if opts.attack_window:
        tmp = opts.attack_window.split('_')
        start_idx, end_idx = int(tmp[0]), int(tmp[1])
        attack_window = [start_idx, end_idx]

    if shifted:
        print('data will be shifted in range: ', [0, shifted])
        val_traces, val_labels, val_textin, key, inp_shape = process_data.process_raw_data_shifted(val_data_whole_pack, target_byte, network_type, shifted, attack_window)
    else:
        val_traces, val_labels, val_textin, key, inp_shape = process_data.process_raw_data(val_data_whole_pack, target_byte, network_type, attack_window)

    clsNum = len(set(val_labels))
    assert (9 == clsNum)
    val_labels = to_categorical(val_labels, clsNum)
    return val_traces, val_labels, val_textin, key, inp_shape


def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plt.title('Performance of ' + model_file_name + ' against ' + dataset_name)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_name)
    plt.show(block=False)
    plt.figure()


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # set params
    model_file = opts.model_file
    target_byte = opts.target_byte

    # checking && Load model
    model = checking_tool.load_best_model(model_file)

    # Load attack data val_traces, val_label, val_textin, key
    X_attack, Y_attack, plaintext_attack, key, inp_shape = load_data(opts)

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
    max_trace_idx = 5000
    rank_step = 10
    ranks = full_ranks(model, X_attack, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step)

    # We plot the results
    # f_ranks[i] = [t, real_key_rank], t for traces and real_key_rank for rank value
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]

    # make the output path
    dataset_name = os.path.basename(opts.input)
    os.makedirs(opts.output, exist_ok=True)
    device_name = os.path.basename(os.path.dirname(opts.input))
    fig_save_dir = os.path.join(opts.output, device_name, 'hw_model')
    os.makedirs(fig_save_dir, exist_ok=True)
    fig_save_name = os.path.join(fig_save_dir, str(dataset_name) + '_rank_performance_byte_{}.png'.format(target_byte))
    print('figure save to file: {}'.format(fig_save_name))
    model_file_name = os.path.basename(model_file).split('.')[0]

    # def plot_figure(x, y, model_file_name, dataset_name, fig_save_name):
    plot_figure(x, y, model_file_name, dataset_name, fig_save_name)

    test_summary_path = 'test_summary.txt'
    contents = '{} --- {} --- {:f}\n\n'.format(opts.input, 'hw_model', acc)
    with open(test_summary_path, 'a') as f:
        f.write(contents)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_file', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-s', '--shifted', type=int, default=10, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
