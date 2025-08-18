#!/usr/bin/env python3
"""
This Python File contains Functions to generate Jazz Licks
dependent on the predictions of a previously trained network.
"""

from pickle import load
from utils.jazz_lstm import generate_lstm_model
from utils.midi_tools import build_note_dict
from numpy import reshape, argmax, append, log, exp, array
from numpy.random import randint, choice
from music21 import stream, instrument, duration as m21_dur, note as m21_note

def get_notes_and_durs(scale):
    """
    Function to load the Generated Note Vectors from the previously
    stored binaries (pickle dump).
    The Function will return a tuple of vectors which contains the note-sequence
    and the duration-sequence.
    """
    note_vec, dur_vec = None, None
    with open(f'stored/notes/{scale}', 'rb') as binaries:
        note_vec = load(binaries)

    with open(f'stored/durs/{scale}', 'rb') as binaries:
        dur_vec = load(binaries)

    return note_vec, dur_vec

def reverse_dict(dic):
    """
    Small helper Function to reverse a Dictionary.
    Important for the back-mapping of Notes/Durations-number-symbol to
    the Note/Duration.
    """
    return {val : key for key, val in dic.items()}

def get_informations(scale='both'):
    """
    Function to reproduce all important informations of a scale.
    The Function will return a tuple of Vectors which contains:
    1. The Vector of Duration/Notes
    2. Unique Duration/Notes
    3. Number of Unique Durations/Notes
    4. Dictionary Duration/Note to Integer
    5. Dictionary Integer to Duration/Note
    6. Inputs for the network (From previous pickle.dump)
    """
    # Note/Duration Vectors
    note_vec, dur_vec = get_notes_and_durs(scale)

    # Unique notes + Amount of unique notes
    unique_notes = set(note_vec)
    n_notes = len(unique_notes)

    # Unique durs + Amount of uniqte durs
    unique_durs = set(dur_vec)
    n_durs = len(unique_durs)

    # Dictionary to map Note/Dur to an integer val
    note_to_int, dur_to_int = build_note_dict(unique_notes, unique_durs)
    # Dictioarny to map integer val back to Note/Dur
    int_to_dur, int_to_note = reverse_dict(dur_to_int), reverse_dict(note_to_int)

    # Inputs for Network
    with open(f'stored/inputs/{scale}', 'rb') as binaries:
        inputs = load(binaries)

    # Put all information in a Tuple
    notes_informations = (note_vec, unique_notes, n_notes, note_to_int, int_to_note, inputs[0])
    durs_informations = (dur_vec, unique_durs, n_durs, dur_to_int, int_to_dur, inputs[1])

    return notes_informations, durs_informations

def set_randomize_val(output, rand_val):
    """
    Function to generate a more or less randomized output based on the rand_val
    If rand val is set 0: The Network will generate an output which corresponds
    to the most likely Melodic or Rhythmic pattern.
    If not: A data distribution will be simulated, which will be further used as
    a base to draw elements from.
    The more the rand_val is set to 1, the more randomized the output sequence will be.
    """
    # Get the most likely output
    if rand_val == 0:
        return argmax(output)
    # Get a randomized output based on the rand_val
    else:
        # Generate a probabilistic data distribution
        output = log(output) / rand_val
        output_exp = exp(output)
        output = output_exp / sum(output_exp)
        # Based on probability p: Draw more or less
        # randomized elements from the sequence
        return choice(len(output), p=output)

def generate_notes_durs(model, note_informations, durs_informations, 
                        length=17, additional_notes=17, note_rand=0.55,
                        dur_rand=0.1):
    """
    Function to generate a sequence which corresponds to a new generated lick.
    The sequence can be further translated in midi format.

    At first the length will determine a sequence of tokens which is later used to
    generate the new notes/durations. The network will pick a random note for the first prediction.
    Any further note will be predicted based on the previously note with the attention mechanism.
    
    Optional Parameters are:
    
    1: Length Determines the length of the sequence - Should be set on 17 which 
    is a good length for a Jazz lick based on 8ths (Training data)
    
    2: Additional Notes: Parameter to fill the Real Notes in - Should be set to 17
    Note: Any value above 17 wont have an impact since the training data is based on
    inputs with the length of 17. Any value below 17 will result in a sequence with less notes.
    
    3: Randomized val for Notes: takes an input from 0 to 1,
    the more the value is to 1 the more randomized the duration pattern will be.
        
    4: Randomized val for Durations: takes an input from 0 to 1, 
    the more the value is to 1 the more randomized the duration pattern will be.
    Note: The duration wont have a big variance so a big number wont affect the sequence 
    that much.
    """
        
        
    # Fecth Note and Duration information
    (note_vec, unique_notes, n_notes, 
     note_to_int, int_to_note, note_input) = note_informations
    
    (dur_vec, unique_durs, n_durs, 
     dur_to_int, int_to_dur, dur_input) = durs_informations
    
    # START and 0 Token to build the sequences with the desired length
    notes = ['START'] * length
    durations = [0] * length
    
    # Fill output sequence with Tokens to determine the output length
    pred_output = [[note, dur] for note, dur in zip(notes, durations)]
    # Input sequence for notes based on START Tokens - determines the length
    note_input = [note_to_int[note] for note in notes]
    # Input sequence for duration based on 0 Tokens - determines the length
    dur_input = [dur_to_int[dur] for dur in durations]
    
    for idx in range(additional_notes):
        
        # Put the input information in a right format
        pred_input = [array([note_input]),
                      array([dur_input])]
        
        # Let the network predict new notes + durations based on the input
        pred_notes, pred_durs = model.predict(pred_input, verbose=0)
        
        # Get a more or less randomized prediction based on the predicted network output 
        randomized_note_val = set_randomize_val(pred_notes[0], note_rand)
        randomized_dur_val = set_randomize_val(pred_durs[0], dur_rand)
        
        # Map the predicted note/duration back to a note/duration symbol
        generated_note = int_to_note[randomized_note_val]
        generated_dur = int_to_dur[randomized_dur_val]
        
        # Append the generated note/druation to the output list
        pred_output.append([generated_note, generated_dur])
        
        # Add the prediction value to the input so the network will make the next prediction
        # Based on the previous prediction
        note_input.append(randomized_note_val)
        dur_input.append(randomized_dur_val)
        
        # Note input length should not exceed length
        if len(note_input) > length:
            note_input = note_input[1:]
            dur_input = dur_input[1:]
        
        # Break if the generated note is a Token
        if generated_note == 'START':
            break
        
    return pred_output

def generate_midi_seq(output, scale, idx):
    """
    Function to translate the output of a sequence to a midi sequence.
    The Parameter Scale will save the midi file to the corresponding folder,
    while the parameter idx will give the midi file a number in the file name.
    """

    # Stream Object
    midi_stream = stream.Stream()

    # Loop for every List in output since output contains lists of note/dur
    for music_pattern in output:

        # Fetch Note and Duration in variables
        note, dur = music_pattern

        # Skip every Token
        if note == 'START' or dur == 0:
            continue

        # Case Note is a rest
        elif note == 'rest':
            # Generate Rest instance
            new_note = m21_note.Rest()
            # Add the corresponding duration to the rest
            new_note.duration = m21_dur.Duration(dur)
            # Convert in piano midi
            new_note.storedInstrument = instrument.Piano()
            # Add to the stream
            midi_stream.append(new_note)

        # Case Note is a note
        else:
            # Generate Note instance
            new_note = m21_note.Note(note)
            # Add the corresponding duration to the Note
            new_note.duration = m21_dur.Duration(dur)
            # Convert in piano midi
            new_note.storedInstrument = instrument.Piano()
            # Add to the stream
            midi_stream.append(new_note)

    # Write the new generated Lick sequence as midi file
    midi_stream.write('midi', fp=f'generated_midi/{scale}/Generated_Lick_{scale}_{idx+1}.mid')

def generate_n_licks(n, jazz_model, notes_informations, durs_informations, 
                     scale='both', note_rand=0.55, dur_rand=0.1, 
                     length=17, additional=17):
    """
    Function to generate automatically n Licks at once in midi format.
    Scale will determine the saving folder for the Licks.
    Note that the note/informations will determine the scale of the generated lick.
    So the Generated lick needs the information for diatonic notes/durs to generate diatonic licks.
    """
    
    # Loop for every generated lick
    for lick_num in range(n):
        # Produce the n-th sequence
        output = generate_notes_durs(jazz_model, notes_informations, durs_informations, 
                                     note_rand=note_rand, dur_rand=dur_rand, length=length,
                                     additional_notes=additional)
        # Write the n-th Lick
        generate_midi_seq(output, scale, lick_num)
