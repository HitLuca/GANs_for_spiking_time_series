# Generating spiking time series with Generative Adversarial Networks: an application on banking transactions

## Description
Master thesis project for my Master in Artificial Intelligence at UvA (University of Amsterdam).

This project aims at generating spiking time series patterns using Generative Adversarial Networks, and compare the results against own baselines.

## Project structure

The ```master_thesis/generative_models``` folder contains the generative models used, along with the scripts for metrics calculation in ```master_thesis/comparison_metrics```.

The ```datasets``` folder contains the datasets used (at the moment only the [Berka dataset](https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm)


Results are saved into the ```outputs``` folders, divided into subfolders for each model.

```utils.py``` contains meta-variables used at training time.

## Getting started
This project is intended to be self-contained, with the only exception being the dataset that is downloaded automatically.

Before starting, run the ```setup.py``` script, that will automatically download and parse the dataset, creating ready-to-use .npy files.


### Prerequisites
Necessary libraries are indicated in the ```requirements.txt```, to install them run

```pip install -r requirements.txt```

### Training the models
As the Java server is now integrated in the project, there is no need to start it separately.

The ```train_model.py``` file contains all the code necessary to train a generative model on the berka dataset. The model type is passed as an argument.

The outputs of the model are stored in the ```outputs/model_name``` folder.

### Comparing different models
First, move the output folder for each model in the ```comparison_metrics/comparison_datasets``` folder, then run the ```evaluate_datasets.py``` script in the ```comparison_metrics``` folder.