#!/usr/bin/env python3

"""
This Python File contains various Functions
to create the network and train the network.
"""
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import (LSTM, Input, Dense, Activation, Embedding,
                          Reshape, concatenate, Multiply, Lambda,
                          TimeDistributed, Permute, RepeatVector,
                          Dropout)

from keras.backend import sum as k_sum
from keras.models import Model, Sequential
from os import path

def generate_lstm_model(n_notes, n_durs, embed=100, rnn_units=256, dense_units=256, scale='both'):
    """
    This Function creates the model for the Neural Network.
    Since this project is using a LSTM model, Recurrent Units are needed.
    Besides this the model will use a mechanic called 'Attention',
    which is commonly used in a translation domain to predict a word
    based on a last word.
    """

    # Network take 2 Inputs: 1 for duration + 1 for notes
    # Attention needs no predetermined input length
    note_in = Input(shape=(None, ), name='note_input')
    dur_in = Input(shape=(None, ), name='dur_input')

    # Emedding Layer: translates mapped notes in vectors
    note_embedding = Embedding(n_notes, embed, name='note_embedd')(note_in)
    dur_embedding = Embedding(n_durs, embed, name='dur_embed')(dur_in)

    # Concat Layer: Aggregate both vectors as a new input for the recurrent layer
    concat_layer = concatenate([note_embedding, dur_embedding],
                               name='concat_layer')

    # 2 LSTM Layers as recurrent part: Every layer can send every hidden state to the next layer
    model = LSTM(rnn_units, return_sequences=True, name='First_LSTM')(concat_layer)
    model = LSTM(rnn_units, return_sequences=True, name='Second_LSTM')(model)

    # Dropout Layer: Fight overfitting
    model = Dropout(0.3)(model)

    # Building the Attention Mechanism

    # Adaption function as a Dense Layer is reshaped to a vector with a shape ( 1, length )
    adapt_func = Dense(1, activation='tanh', name='Adapt_layer')(model)
    adapt_func = Reshape([-1], name='Rm_1_vec')(adapt_func)
    # Apply Softmax for calculating weights
    sm_activation = Activation('softmax', name='Softmax_Act')(adapt_func)

    # Calculate the weighted sum of the hidden states
    # RepeatVector Layer: Copys the weights rnn_units-times and creates
    # A matrix with a shape ( rnn_units, length )
    # Matrix is transposed with a Permute Layer to a shape ( length, rnn_units )
    sm_activation_rep = Permute([2, 1])(RepeatVector(rnn_units)(sm_activation))
    # Matrix is elementwise multiplid with the hidden states of the last LSTM layer
    # with a shape of ( length, rnn_units )
    hws = Multiply(name='Multiple_RNN_Weights')([model, sm_activation_rep])
    # Lambda layer: Calculate Sum of the length-Dimension => Generate context Vector with rnn_units-length
    hws = Lambda(lambda model_inp:
                 k_sum(model_inp, axis=1),
                 output_shape=(rnn_units, ),
                 name='Context_Vector')(hws)

    # First Output - Notepitch
    note_out = Dense(n_notes,
                     activation='softmax',
                     name='note_height')(hws)

    # Second Output - Duration
    dur_out = Dense(n_durs,
                    activation='softmax',
                    name='duration')(hws)

    # Combine the created input layers + output layers in one model
    final_model = Model([note_in, dur_in],
                        [note_out, dur_out],
                        name=f'Jazz_LSTM_{scale}')

    # Categorical Crossentropy since its a classification problem with One-Hot-Coded Training data
    # Adam as optimizer since adam works pretty well
    final_model.compile(loss=['categorical_crossentropy',
                              'categorical_crossentropy'],
                        optimizer='adam')

    return final_model

def train(inputs, outputs, model, folder, both=True, verbose=0, bs=32, ep=100, checkpoints=True, patience=5):
    """
    Wrapper function for building a training environment for the model.
    inputs/outputs and model are the obligatory parameters for the training.
    Folder determines the used scale and is equivalent to the folder name in the weights folder.
    Per default both is true and corresponds to a folder.
    Verbose can be set on 1 to get the training output from the keras function.
    The training consists of a mini batch gradient, where the batch size can be configured with
    the parameter bs.
    Ep determines the amount of training epoches.
    For consistency and a more comfortable training 2 checkpoints are implemented.
    For fighting overfitting early stopping is implemented (regulated with patience).
    Since the model is implented with keras the format for saving the structured data
    is h5.
    """

    folder = 'both' if both else folder
    folder_path = f'weights/{folder}/'
    if checkpoints:
        first_checkpoint = ModelCheckpoint(path.join(f'weights/{folder}/', 'weights-improvement-{epoch:02d}-{loss:.4f}.h5'),
                                           monitor='loss',
                                           verbose=verbose,
                                           save_best_only=True,
                                           mode='min')

    second_checkpoint = ModelCheckpoint(path.join(f'weights/{folder}/', 'weights.h5'),
                                        monitor='loss',
                                        verbose=verbose,
                                        save_best_only=True,
                                        mode='min')

    early_stop = EarlyStopping(monitor='loss',
                               restore_best_weights=True,
                               patience=patience)

    callbacks = [first_checkpoint, second_checkpoint, early_stop] if checkpoints else [early_stop, second_checkpoint]

    model.save_weights(folder_path, 'weights.h5')
    model.fit(inputs, outputs,
              verbose=verbose,
              epochs=ep, batch_size=bs,
              validation_split=0.3,
              shuffle=True,
              callbacks=callbacks)
