#!/usr/bin/env python3
"""
This Python File contains Functions to extract informations from midi files and porcess those informations to train a neural network.
"""

from os import path
from music21 import note, converter
from numpy import reshape
from keras.utils import to_categorical
from glob import glob
from pickle import dump

def read_midi_data(scale, both=True, folder='data'):
    """
    Function for reading the Training data.
    The data must be in the Folder 'data' as sequential midi files.
    On top of that the midi files can just contain melodical sequences.

    There are 3 Options:
    > alterated: Will load training data written with a alterated or diminished scale for the dominant
    > diatonic: Will load training data written with just diatonic and small chromatic approaches
    > both: Will load alterated and diatonic training data
    """

    # Get all midi files from 'data' folder
    midi_data = glob(path.join(f'{folder}/{scale}', '*.mid'))
    # music21.converter: Tool for loading music files like midi
    midi_conv = converter

    if both:
        other_scale = 'alterated' if scale == 'diatonic' else 'diatonic'
        other_files = glob(path.join(f'{folder}/{other_scale}', '*.mid'))
        # Add aditional midi files if desired
        midi_data += other_files

    return midi_data, midi_conv

def extract_notes_and_duration(scale='diatonic', both=True, length=17, show=True, folder='data', save_data=True, send_names=False):
    """
    This Function will extract the notes from the training data in midi format.
    The scaling material of the training data can be adjusted with the parameters.
    On default all training data will be taken.
    Addionally the sequential length of the result can be configured.
    Since the projects goal is to generate Jazz Licks the length should never exeed the default length.
    The default length is based on a classical Jazz Lick with a chain of eights and
    a full note in the 3. measure (tonic)
    """

    # Saving extracted notes and durations in its order for every midi file
    notes = []
    durs = []
    midi_names = []

    # Get the midi data and the music 21 converter
    midi_data, midi_conv = read_midi_data(scale, both, folder=folder)

    # Loop through every midi file
    for midi_file in midi_data:
        if show:
            print(f'Fetch data from : {midi_file}')
        if send_names:
            midi_names.append(midi_file)

        # Parse the corresponding midi file per stringname - a score will be returned
        lick_score = midi_conv.parse(midi_file)

        # Needed? - Test if it makes an impact
        # Delimiter to seperate the informations of each midi file
        # Its important to keep track of every Lick and its musical elements
        notes += length * ['START']
        durs += length * [0]

        # Loop through the music score of each midi file
        for score_element in lick_score.flat:
            # Check if the element is a note
            # If so: Append the pitch + duration of the note element
            if type(score_element) == note.Note:
                notes.append(str(score_element.nameWithOctave))
                durs.append(score_element.duration.quarterLength)
            # Check if the element is a rest
            # If so: Append the rest + duration
            elif type(score_element) == note.Rest:
                notes.append(str(score_element.name))
                durs.append(score_element.duration.quarterLength)

    store_folder = scale if not both else 'both'
    
    if save_data:
        # Save binaries for later
        with open(path.join(f'stored/notes/{store_folder}'), 'wb') as store:
            dump(notes, store)

        with open(path.join(f'stored/durs/{store_folder}'), 'wb') as store:
            dump(durs, store)

    # Return extracted Notes + Durations (+ midi_names)
    if send_names:
        return notes, durs, midi_names
    return notes, durs

def generate_sequence(notes, durs, note_to_int, dur_to_int, scale, length=17):
    """
    This Function will format the previous formated data (notes and durations)
    in a receivable format for the network (sequential data).
    The Function will return a Tuple (inputs, outputs),
    where inputs consist of 2 Arrays (note + duration) with a shape
    of ( m , length ) since the length regulates the sequence length.
    The outputs naturally contains 2 Arrays (note + duration) with a shape
    of ( m , x ) since the outputs are One-Hot-Coded.
    The number m will take the size note-vector-length - configured length,
    as a result the sequential input will be multiplied with a sliding window.
    """

    # Collect Data about notes and durations
    unique_notes = list(note_to_int.keys())
    unique_durs = list(dur_to_int.keys())
    size_notes = len(unique_notes)
    size_durs = len(unique_durs)

    # Data Structure for Inputs/Outputs for Notes and Durations
    inputs_note, outputs_note = [], []
    inputs_durs, outputs_durs = [], []

    # Iterate with a sliding window over all notes
    for num in range(len(notes) - length):

        # Take a Subsequence of notes with the configured length
        note_seq_in = notes[num:num + length]
        # Fetch Note from note vector at position num + length
        note_seq_out = notes[num + length]

        # Convert the input sequence in a sequence of numbers + append
        inputs_note.append([note_to_int[symbol] for symbol in note_seq_in])
        # Convert the output Note into a number + append
        outputs_note.append(note_to_int[note_seq_out])

        # Take a Subsequence of durations with the configured length
        dur_seq_in = durs[num:num + length]
        # Fetch Duration from the duration vector at position num + length
        dur_seq_out = durs[num + length]
        # Convert the input sequence in a sequence of numbers + append
        inputs_durs.append([dur_to_int[symbol] for symbol in dur_seq_in])
        # Convert the output Note into a number + append
        outputs_durs.append(dur_to_int[dur_seq_out])

    # Save the current length of Arrays in Inputs
    # The Number will determine m of the shape (m , length)
    pattern_length = len(inputs_note)

    # Reshaping the input vectors without changing the data within
    # Important for preventing errors caused by different shapes when training the network
    inputs_note = reshape(inputs_note, (pattern_length, length))
    inputs_durs = reshape(inputs_durs, (pattern_length, length))
    inputs = [inputs_note, inputs_durs]

    # One Hot Coding the outpute notes and durations
    # The total size of classes corresponds to the amount of unique notes and unique durations
    outputs_note = to_categorical(outputs_note, num_classes=size_notes)
    outputs_durs = to_categorical(outputs_durs, num_classes=size_durs)
    outputs = [outputs_note, outputs_durs]

    with open(path.join(f'stored/inputs/{scale}'), 'wb') as store:
        dump(inputs, store)

    return inputs, outputs

def build_note_dict(notes, durs):
    """
    This Function creates 2 Dictionarys based on the 2 Input Vectors:
    > 1: Mapping note names to a numerical value
    > 2: Mapping Durations to a numerical value
    These Dictionarys will be further used to create Training data
    for the network, because numbers are needed as a representation for the notes.
    """

    # Get unique Notes (+ pitch) and Durations
    unique_notes = sorted(list(set(notes)))
    unique_durs = sorted(list(set(durs)))

    dur_to_int, note_to_int = {}, {}

    # Extract unique notes/durations and assign a numerical value
    for num in range(len(unique_notes)):
        note = unique_notes[num]
        note_to_int[note] = num

    for num in range(len(unique_durs)):
        dur = unique_durs[num]
        dur_to_int[dur] = num

    return note_to_int, dur_to_int

