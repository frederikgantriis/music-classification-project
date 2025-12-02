# Music Classification

For this project, I have used the GTZAN dataset.

This dataset consist of the following 10 classes:

TODO: INSERT DISTRIBUTION OF THE CLASSES

0. blues
1. classical
1. country
1. disco
1. hiphop
1. jazz
1. metal
1. pop
1. reggae
1. rock

Furhtermore, the following images show a representation of the waveform, spectrogram and mel-spectrogram of the blues class:

![blues](visualise/plots/images/raw-data/genres_original/blues/blues.00000.wav-mel_spectrogram.png)
![blues](visualise/plots/images/raw-data/genres_original/blues/blues.00000.wav-spectrogram.png)
![blues](visualise/plots/images/raw-data/genres_original/blues/blues.00000.wav-waveform.png)
All other images can be found in [this folder](visualise/plots/images/raw-data/genres_original/)

## The pipeline

For this project, I have created a pipeline using Data Version Control (DVC).

This ensures that data is processed in isolated stages, that ensures control and easy maintenance.

Furthermore, DVC allows for experimentation process to be mostly automated.

This means, that I can tune hypeparameters by queueing different experiments.
Then, I can run the queue, where it will automatically run the entire queue, and save the results and illustrations.

The pipeline consist of the following steps:

- Prepare: This step is a simple script that fetches the dataset, unzip and move it to a raw-data folder.
- Transform: Here, the data is split into train and test, converted into mel-spectrograms, and split into different sequences.
- Train: In this step, the model is quite simply trained on the training dataset from previous steps.
- Evaluate: Lastly, the trained model is evaluated, and the metrics (Accuracy, Precision and F1) are logged along with a confusion matrix.

## The models

The Models evaluated in this project are:

- LSTM
- GRU
- BiLSTM
- BiGRU
- TCN
- MLP (variation)

All of these models have been trained over 20 epochs, where the model uses the `adam` optimiser and `sparse_categorical_crossentropy` as the loss-function. These were chosen, since they were the standard.

Furthermore, the models have been chosen based on the project description, as well as suggestions from research and literature.

## Problems during development

Firstly, something has to be said about the development of this project.

I have very limited experience with working with audio-files. This means, that large parts of the time spent on the project, was spent on managing and preparing the audio-files to enable training.

Furthermore, after doing a bunch of experiments, I noticed that the accuracy still was around 0.1%, which is the same accuracy one would get by guessing that all music are classical music.

After observing this, I tried to focus on the training step of the model. Here, the loss went from around 2.5 to a final loss of around 0.4-0.8. This was true from all the models. I then tried to introduce a 3% dropout, which made the loss completely steady around 2.5, while the precision weren't becoming any better. This was not what I expected.

Furthermore, the models didn't get any better by making the sequences longer, which I would expect the model to, since it has more data to predict what genre a given audiofile is. It can be hard to classify if a single hit on a snare-drum is rock, pop or anything else, but I would expect it to classify it properly after hearing a couple of bars.

After, this observation, I inspected the transformation step. Here, a major error was noticed. Limited by my knowledge of working with audio-files, I had - when extracting audio-files and its labels - made an error that meant that the audio-file didn't match the label. This meant, that labels where shuffled and the models would then not have any chance of classifying images properly.

After fixing this error, accuracy when from 0.1% to 0.65%.

This problem means that the time for final testing were quite limited, and there is room for improvement. However, with this knowledge, let us move on to the evaluation.

## Evaluation

Firstly, let's take a look at a confusion matrix for the best performing model:

![confusion](eval/plots/images/confusion_matrix.png)

It should be noted, that the order correspond to the order given at the start of this file.

In general, the models are quite good at classifying classical music. A reason for this, might be that classical music is the only category in this dataset where there is no clear tempo or sense of beat.

If we look at the mel-spectrogram for disco for instance, there seems to be a clear sense of beat with space in between:

![disco](visualise/plots/images/raw-data/genres_original/disco/disco.00000.wav-mel_spectrogram.png)

If we then look at classical music, the same sense of beat can be hard to identify:

![classical](visualise/plots/images/raw-data/genres_original/classical/classical.00000.wav-mel_spectrogram.png)
There doesn't seem to be any kind of beat, or repeated patterns that are obvious from the disco example.

Furthermore, it seems that disco is often misclassified as pop, reggae and rock. From a musical standpoint, it makes sense that disco is closer to pop, reggae and rock than jazz, classical and metal. This misclassification could be due to inconsistencies in the dataset. A model can only be as good as the data it is trained on, and different people might classify some tracks as disco while others would classify it as pop. In addition, the classes are mainly focused on popular music. For instance, only one category is selected for jazz. In this category, there can be a large variety in music from the 1920's to the music from today.
