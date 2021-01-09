# Bayesian Inference for Semi-Supervised Learning with Normalizing Flows
This repository contains PyTorch implementation of the experiments from our paper

[Bayesian Inference for Semi-Supervised Learning with Normalizing Flows](https://github.com/hoangphat1908/bayes-flowgmm/blob/public/bayes-flowgmm.pdf)

by Phat Nguyen and Apoorv Vikram Singh.

# Introduction

Normalizing flows work by transforming a latent distribution through an invertible neural network for an approach to generative modelling. Izmailov et al. (2019) proposed FlowGMM, an approach to generative semi-supervised learning using normalizing flows by modelling the data in latent space as a Gaussian mixture model.

This project builds on top of their work by performing Bayesian Inference for the FlowGMM model using techniques suggested by Wilson & Izmailov (2020). We show results on real-world text data-sets: AG News and Yahoo Answers. The results show significant improvement compared to the FlowGMM model, showing that Bayesian model averaging can give much better performance when compared to point estimates.

We also introduce Flow Ensembles, a new method to combine FlowGMM models from latent space distribution, which further enhances test accuracy from traditional Deep Ensembles methods.

# Installation
To run the scripts you will need to clone the repo and install it locally. You can use the commands below.
```bash
git clone https://github.com/hoangphat1908/bayes-flowgmm.git
cd bayes-flowgmm
pip install -e .
```
## Dependencies
We have the following dependencies for FlowGMM that must be installed prior to install to FlowGMM
* Python 3.7+
* [PyTorch](http://pytorch.org/) version 1.0.1+
* [torchvision](https://github.com/pytorch/vision/) version 0.2.1+
* [tensorboardX](https://github.com/lanpa/tensorboardX)

We provide the scripts and example commands to reproduce the experiments from the paper.
# Tabular Datasets

The experiments are shown in [this ipython notebook](https://github.com/hoangphat1908/bayes-flowgmm/blob/public/experiments/bayesian/bayesian.ipynb). The notebook includes temperature scaling method and performance visualization of the Flow Ensembles model for FlowGMM

The tabular datasets will be download and preprocessed automatically the first time they are needed. Using the commands below you can reproduce the performance from the table.

|| AGNEWS | YAHOO |
|---------|------|------|
|FlowGMM |   82.6  | 59.0 |
|Deep Ensembles-3|   83.3  | 62.2 |
|Flow Ensembles-3 |   **83.8**  | **63.2** |

## Text Classification
Train multiple **FlowGMM** models on AG-News (200 labeled examples) and compare results between Deep Ensembles and Flow Ensembles:
```bash
python run_ensembles.py --dataset=AG_News --labeled=200 --num_models=3 --num_epochs=100 --test_epochs=5 --lr=3e-4 --unlab_weight=.6
```
Train multiple **FlowGMM** models on YAHOO Answers (800 labeled examples) and compare results between Deep Ensembles and Flow Ensembles:
```bash
python run_ensembles.py --dataset=YAHOO --labeled=800 --num_models=3 --num_epochs=200 --test_epochs=10 --lr=3e-4 --unlab_weight=.2
```

# References for Code Base

* This repo was originally forked from the [Flow Gaussian Mixture Model (FlowGMM) GitHub repo](https://github.com/izmailovpavel/flowgmm).
* Code for SWAG is ported from [Bayesian Deep Learning and a Probabilistic Perspective of Generalization GitHub repo](https://github.com/izmailovpavel/understandingbdl).
