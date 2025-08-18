#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def extract_lick_elements(licks):
    """
    Function which creates a nested list of licks with cutting all Tokens
    """
    vectors, vector = [], []
    for idx, element in enumerate(licks):
            if element != "START" and element != 0:
                vector.append(element)
            # Indicates if the Lick is terminated
            elif (licks[idx + 1] == "START" or licks[idx + 1] == 0) and vector:
                vectors.append(vector)
                vector = []
    return vectors

def transform_licks(licks, transformer):
    """
    Mapps a Lick into numbers which is essential for visualizing the information.
    The Function returns a nested list with its licks and an overall array for data distribution.
    """
    licks_transformed = []
    for lick in licks:
        lick_transformed = []
        for element in lick:
            lick_transformed.append(transformer[element])
        licks_transformed.append(lick_transformed)

    overall_transformed = [element for sublist in licks_transformed for element in sublist]
    return licks_transformed, overall_transformed

def comparing_boxplot(overall_arrays, epochs, title, filename):
    """
    This Function creates 2 Boxplots which shows the data distribution of the overall-array,
    which helps to show similarity and differences.
    """
    sns.set()
    n = len(overall_arrays)
    fig, axes = plt.subplots(1, n, sharey=True, figsize=(6, 6))
    fig.tight_layout()
    for index, array in enumerate(overall_arrays):
        sns.boxplot(array, ax=axes[index])
    fig.suptitle(title, y=1.05)
    for index, epoch in enumerate(epochs):
        if index == 0:
            axes[0].title.set_text("Training Data")
        else:
            axes[index].title.set_text(f"Ep={epoch}")
        axes[index].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.show()
    fig.savefig(f'imgs/{filename}.png', dpi=250)

def show_p_val(test_array_note, test_array_dur, overall_notes, overall_durs, epoch):
    """
    Function to calculate pval with t-test.
    The Result shows how much Pitch and Duration differ between generated Licks and the training data.
    if the p-val is below 0.05 the distribution of pitches/durations is significant different compared to the training data.
    Goal is a NON-Different (no significant difference) result to show that the generated Licks are similar to the training data.
    """
    t, p_pitch = stats.ttest_ind(overall_notes, test_array_note)
    t, p_dur = stats.ttest_ind(overall_durs, test_array_dur)
    alpha = 0.05
    print(f'\nEpoch = {epoch}\nP Value for Durations: {p_pitch:.2} Significant: {p_pitch < alpha}\nP Value for Pitch: {p_dur:.2} Significant: {p_dur < alpha}\n')

def show_p_vals(test_array_note, test_array_dur, overall_notes, overall_durs, epochs):
    """
    Function to apply the show-p-val method on Multiple results
    """
    for index, epoch in enumerate(epochs):
        show_p_val(test_array_note[index], test_array_dur[index], 
                   overall_notes, overall_durs, epoch)

def plot_data_distribution(overall, overall_gen, title, epochs, filename):
    """
    Function to make Subplot of all Data Distribution (Density)
    for generated Licks at certain epochs
    """
    sns.set()
    fig, axes = plt.subplots(3, 2, sharey=True, figsize=(10, 10))
    fig.tight_layout()
    outer_index = 0
    for i in range(3):
        for j in range(2):
            epoch = epochs[outer_index]
            array = overall_gen[outer_index]
            sns.histplot(overall, kde=True, ax=axes[i, j])
            sns.histplot(array, kde=True, ax=axes[i, j])
            axes[i, j].title.set_text(f"Ep={epoch}")
            outer_index += 1
    fig.suptitle(title, y=1.05)
    fig.savefig(f'imgs/{filename}.png', dpi=250)
    plt.show()
