# Generative Jazz Licks  
* GDKI WiSe 22/23  

## Short Description  
As part of the final examination for the module *Foundations of Artificial Intelligence*, this project implements a recurrent neural network that is used to create generative jazz music.  
The generative jazz music is produced in the form of so-called *jazz licks*. A jazz lick refers to a short phrase over a cadential harmonic scheme. Such jazz licks are commonly used for improvisation and can appear in many different forms. Within the scope of this project, three-bar II-V-I jazz licks in major are generated.  
The project addresses the central question of whether a neural network is capable of generating new, acceptable-sounding jazz licks. These are validated using various analyses and metrics.  

## Installation  
To use the repository, a functioning Jupyter environment is required.  
Furthermore, several Python modules are needed, which can be installed with the help of the *requirements.txt* file and pip:  
```
pip install -r requirements.txt
```
In addition, a current version of the software *Musescore3* is required to play the MIDI files (https://musescore.com/).  

## Execution  
* The project is executed sequentially across three different notebooks.  
* In total, three different models are trained, each based on different jazz scales and tonal material (diatonic, altered, and both combined). This results in three times the amount of stored data and processes.  
* The notebooks serve the following purposes:  
  * **1_LSTM_generated_weights**: Used to read and process the training data (MIDI format) into a suitable format for the network. The networks are then trained with the training data, and the weights are saved. The training process involves various hyperparameters that can be set as desired.  
  * **2_LSTM_generate_lick**: Used to generate new jazz licks. The previously trained weights are used to make predictions. Similarly, various hyperparameters can be set to control the generation/prediction process.  
  * **3_Model_Evaluation_and_Validation**: Used solely to compare the generated jazz licks with the training data. The goal is to evaluate the generalization ability and pattern recognition of the network models. In addition to a Turing-test-like approach, statistical and musicological analyses are conducted.  

## File Overview  
* The following files and folders belong to the project:  
  **Folders**:  
  * **Sample_Audios**: Contains the audio files used for the Turing-test-like approach. The audio files are created with Musescore (from the MIDI files) and supplemented with suitable harmonic accompaniment and a ternary rhythm interpretation (swing). A total of 10 audio files are randomly selected (5 original licks + 5 generated licks).  
  * **data**: Contains all digitized training data – filtered into altered and diatonic.  
  * **generated_midi**: Contains all generated licks (from *2_LSTM_generate_lick*), organized in folders by the number of epochs. All newly generated licks are stored in the corresponding scale folder (altered, diatonic, or both).  
  * **imgs**: Contains images of the architecture or from the evaluation/validation.  
  * **stored**: Contains information in binary format (from *1_LSTM_generated_weights*) for efficient data transfer between notebooks. The binary files are stored in scale-specific folders.  
  * **weights**: Stores checkpoints (if the optional parameter is set) and the trained weights from *1_LSTM_generated_weights*, which are then used to generate licks in *2_LSTM_generate_lick*.  
  * **documents**: Contains the documentation, PowerPoint presentation, and the poster in PDF format.  

* **utils** contains outsourced Python code with implemented functions for handling and programming the actual tasks. These outsourced functions provide interfaces and help maintain the organization of the notebooks:  
  * **evaluate.py**: Contains methods for generating graphics for validation.  
  * **jazz_lstm.py**: Contains the network architecture.  
  * **midi_generation.py**: Contains functions for generating new jazz licks in MIDI format.  
  * **midi_tools.py**: Contains functions for loading and transforming the training data.  
  * **check_overfitting.py**: Checks whether a generated jazz lick was simply copied from the training data by comparing each note sequence from the generated folder with every note sequence from the training data (ignoring rhythm). Suitable visualizations show the proportion of overfitted licks and list the names of valid licks (not overfitted).  

* **Notebooks**: See *Execution*.  

## Notes  
* The finished audio files are supplemented with a minimal Musescore-generated accompaniment (Cm9–F7–Bbmaj9) and the jazz-typical rhythmic interpretation, *swing*, in order to embed the licks in a realistic sound environment.  
* The Python files are thoroughly documented and include a docstring so that explanations can quickly be displayed within the notebook.  
* The notebooks contain additional information about the goals of each notebook.  
* The documentation language of the notebooks/Python files is English, since, in accordance with many conventions, English variable names are used, ensuring linguistic consistency of the Python code.  
