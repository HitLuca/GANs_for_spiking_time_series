# Generating spiking time series with Generative Adversarial Networks: an application on banking transactions

## Description
Master thesis project for my Master in Artificial Intelligence at UvA (University of Amsterdam).

This project aims at generating spiking time series patterns using Generative Adversarial Networks, and compare the results against own baselines.

## Project structure

The ```implementation``` folder contains the generative models used, along with the scripts for metrics calculation.

The ```datasets``` folder contains the datasets used (at the moment only the [Berka dataset](https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm)


Results are saved into the ```outputs``` folders on each sub-model directory.

```utils.py``` contains meta-variables used at training time.

## Getting started
This project is intended to be self-contained, with the only exception being the dataset that is downloaded automatically.

Before starting, run the ```setup.py``` script, that will automatically download and parse the dataset, creating ready-to-use .npy files.


### Prerequisites
Necessary libraries are indicated in the ```requirements.txt```, to install them run

```pip install -r requirements.txt```

### Running the algorithms
As the Java server is now integrated in the project, there is no need to start it separately.

The ```experiment_runner.py``` file contains all the code necessary to train the predictive models on each log file and evaluate them with each inference method.

### Training the models
Just choose your model of choice and run the ```train_name_of_model.py``` script. Variables that can be changed are located in ```train_name_of_model.py``` and ```utils.py``` scripts.

The outputs of the model are stored in the corresponding ```outputs``` folder.

### Evaluating the models
Copy the output folder of the various models into the ```comparison_datasets``` folder, and then run the ```evaluate_datasets.py``` scripts

