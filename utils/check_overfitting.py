#!/usr/bin/env python3

from utils.evaluate import *
from utils.midi_tools import extract_notes_and_duration, build_note_dict

def get_lick(folder_name, folder, show=False, scale='both', both=True, train_data=False):
    if train_data:
        note, dur, names = extract_notes_and_duration(scale=scale, both=both, show=show, send_names=True)
    else:
        note, dur, names = extract_notes_and_duration(scale=f'{scale}/{folder_name}', show=show, both=both, length=17, folder=folder, save_data=False, send_names=True)
 
    lick_note = extract_lick_elements(note)
    return lick_note, names

def overfitting_rate(epoch_folder_name, scale='both', show=False, both=True):
    """
    Calculates an overfitting score: All Licks and TrainingData in a certain scale folder will be compared. A high score shows low overfitting, while a low score shows a lot overfitted licks (0 = every lick is overfitted and 1 = no lick is overfitted).
    Besides this, this function will return the names of the viable licks (not overfitted).
    """
    # Fetch all Licks in zipped format (Note_sequence, Duration_sequence)
    generated_licks, gen_names = get_lick(epoch_folder_name, 'generated_midi', show=show, scale=scale)
    training_data, train_names = get_lick(scale, 'data', show=show, scale=scale, both=both, train_data=True)
    viable_licks = [idx for idx in range(len(generated_licks))]
    gen_names = [name.split('/')[-1] for name in gen_names]
    # To Test if the modell overfitted and copied training data all licks 
    # are compared to each other in terms of note sequence 
    # because the duration will be similar
    # Get all training data sequences
    for original_lick in training_data:
        # Get all generated licks
        for idx, generated_lick in enumerate(generated_licks):
            # Compare each original seq to each generated seq
            if original_lick == generated_lick:
                if idx not in viable_licks:
                    continue
                viable_licks.remove(idx)
                
    
    if show:
        print(f"Epoch 180 got {(len(viable_licks) / len(generated_licks)) * 100} percent of non overfitted results")
    gen_names = [gen_names[index] for index in viable_licks]
    return viable_licks, (len(viable_licks) / len(generated_licks)), gen_names

def plot_overfitting_rate(overfitting_scores, scale):
    """
    Shows the percantage of overfitted licks per epoch
    """
    fig, ax = plt.subplots()
    sns.lineplot(overfitting_scores)
    sns.set()
    plt.title(f'Percantage of new Licks with increasing Epochs ({scale})')
    plt.ylabel('Percantage')
    plt.xlabel('Epochs')
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels([5, 35, 70, 80, 90, 180])
    plt.savefig(f'imgs/overfitting_scores_{scale}.png' , dpi=250)
    plt.show()

def collect_overfitting_and_names(test_folder, scale='both'):
    """
    Collects Data about the percentage of overfitted licks and
    Names of not overfitted licks.
    """
    overfitting_scores = []
    viable_names = []
    for folder in test_folder:
        lick_idx, score, name = overfitting_rate(folder, scale=scale, show=False)
        viable_names.append(name)
        overfitting_scores.append(score)
    return overfitting_scores, viable_names
