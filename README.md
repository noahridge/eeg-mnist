# EEG MNIST 
This is an investigational project using PyTorch to develop a neural network that predicts digits from EEG (Electroencephalogram) data based on the MindBigData dataset.

In the MindBigData 2015 experiment participants were shown digits 0-9 while brain activity was monitored via a variety of off-the-shelf commercial EEG systems. The dataset consists of 2-second snippets of EEG signals captured from each of the EEG channels. Multiple EEG systems were used having between 1 and 14 channels. 

More information about the experiment can be found here 

https://www.researchgate.net/publication/281817951_MindBigData_the_MNIST_of_Brain_Digits_v101

A couple of approaches were used to develop a neural network that could predict the digit shown from the EEG data. 

One approach was to use a 1d convolutional neural network trained directly on the multi-channel time-series data from each data capture event. However, this approach was shown to have very low accuracy, likely due to variations in the signals between channels. 

The second approach is to use a time-frequency analysis method, converting the EEG signals to spectrograms and then training a 2d convolutional neural network on these signals. Hopefully, this will better capture the event-related oscillations typical with brain signals. 

This is currently still a work in progress. 

