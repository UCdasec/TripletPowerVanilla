#!/usr/bin/python3.6
import pdb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential


def generate_params():
    params = {
                'padding': 'same',
                'act': 'elu',

                'conv1': 64,
                'k_size1': 2,   # kernel size
                'p_size1': 2,

                'conv2': 128,
                'k_size2': 2,   # kernel size
                'p_size2': 2,

                'conv3': 256,
                'k_size3': 2,   # kernel size
                'p_size3': 2,

                'conv4': 512,
                'k_size4': 2,   # kernel size
                'p_size4': 2,

                'conv5': 512,
                'k_size5': 2,   # kernel size
                'p_size5': 2,

                'dense1': 4096,
                'dense2': 4096,
                'dense_act': 'elu',

                'optimizer': 'adam',

                'batch_size': 100,
                'epochs': 100
            }
    return params


def create_power_model(input_shape, emb_size, classification=True):
    '''In wang's paper, the identify model and the hamming weight model
    is the same except the last layer, output layer size of identify model
    is 256 and output layer size of hamming weight model is 9'''
    params = generate_params()
    if 256 == emb_size:
        print('Creating identical power model...')
    elif 9 == emb_size:
        print('Creating hamming weight power model...')
    else:
        raise ValueError('class number should be either 256 or 9!')

    inp = Input(shape=input_shape)
    for i in range(1, 6):
        if 1 == i:
            x = Conv1D(filters=params['conv{}'.format(i)],
                       kernel_size=params['k_size{}'.format(i)],
                       activation=params['act'],
                       padding=params['padding'],
                       kernel_initializer='glorot_normal')(inp)
        else:
            x = Conv1D(filters=params['conv{}'.format(i)],
                       kernel_size=params['k_size{}'.format(i)],
                       activation=params['act'],
                       padding=params['padding'],
                       kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size=params['p_size{}'.format(i)])(x)

    x = GlobalMaxPool1D()(x)

    x = Dense(params['dense1'], activation=params['dense_act'], kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    x = Dense(params['dense2'], activation=params['dense_act'], kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    if classification:
        x = Dense(emb_size, activation='softmax')(x)
        model = Model(inp, x, name='power_model')
        print('Compiling the model...')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        print('[log] --- finish construct the power trace model')
        return model
    else:
        embeddings = x
        return inp, embeddings


# MLP Best model (6 layers of 200 units)
def mlp_best(input_shape=200, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    x = Dense(input_shape, input_dim=1400, activation='relu')(inp)
    for i in range(4):
        x = Dense(input_shape, activation='relu')(x)
    if classification:
        x = Dense(emb_size, activation='softmax')(x)
        optimizer = RMSprop(lr=0.00001)
        model = Model(inp, x, name='mlp')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the mlp model')
        return model
    else:
        embeddings = x
        return inp, embeddings


# CNN Best model
def cnn_best(input_shape, emb_size=256, classification=True):
    # From VGG16 design
    # input_shape = (700,1)
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn model')
        return model
    else:
        embeddings = x
        return inp, embeddings


### CNN Best model
def cnn_best2(input_shape, emb_size=256, classification=True):
    # From VGG16 design
    # input_shape = (1400,1)
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best2')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        embeddings = x
        return inp, embeddings


def create_hamming_weight_model(input_shape):
    ''' create the hamming weight power model, it is from Wang's thesis '''
    inp = Input(shape=input_shape)

    # block 1
    x = Conv1D(filters=64, kernel_size=11, padding='same',
               activation='selu', name='block1_conv')(inp)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    # block 2
    x = Conv1D(filters=128, kernel_size=11, padding='same',
               activation='selu', name='block2_conv')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    # block 3
    x = Conv1D(filters=256, kernel_size=11, padding='same',
               activation='selu', name='block3_conv')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    # block 4
    x = Conv1D(filters=512, kernel_size=11, padding='same',
               activation='selu', name='block4_conv')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    # block 5
    x = Conv1D(filters=512, kernel_size=11, padding='same',
               activation='selu', name='block5_conv')(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten(name='flatten')(x)

    dense_1 = Dense(4096, activation='relu', name='fc1')(x)
    dense_2 = Dense(4096, activation='relu', name='fc2')(dense_1)
    x = BatchNormalization()(x)
    out = Dense(9, activation='softmax')(dense_2)

    # Create model
    model = Model(inp, out, name='hw_model')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print('[log] --- finish construct the cnn2 model')
    return model


def test():
    inp_shape = (95, 1)
    emb_size = 256
    classification = True

    # test the mlp model
    best_model = mlp_best(emb_size, classification)
    best_model.summary()

    # test the cnn model
    best_model = cnn_best(inp_shape, emb_size, classification)
    best_model.summary()

    # test the cnn2 model
    best_model = cnn_best2(inp_shape, emb_size, classification)
    best_model.summary()

    # test the power_model
    model = create_power_model(inp_shape, emb_size, classification)
    model.summary()

    # test the hamming weight model
    model = create_hamming_weight_model(input_shape=inp_shape)
    model.summary()


if __name__ == '__main__':
    test()
