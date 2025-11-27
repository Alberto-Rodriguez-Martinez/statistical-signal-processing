# Statistical Signal Processing – Interactive Jupyter Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alberto-Rodriguez-Martinez/statistical-signal-processing)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Alberto-Rodriguez-Martinez/statistical-signal-processing/HEAD)

This repository provides a collection of Jupyter notebooks designed to support the course Statistical Signal Processing for undergraduate and graduate engineering students. The notebooks complement the theoretical lectures and offer an interactive environment in which students can explore statistical concepts, implement algorithms, and analyse real or simulated signals.

## Objectives
* Provide clear, concise theoretical explanations integrated directly into the notebooks.
* Enable hands-on experimentation with estimators, detectors, and classifiers.
* Reinforce understanding through simulations and visualisation of statistical phenomena.
* Offer reusable code and examples for further study or project work.

## How to Use These Notebooks
* Each notebook is self-contained and can be run locally or on cloud-based Jupyter environments.
* Cells marked as interactive allow modification of parameters to observe real-time effects on estimators or decision rules.
* Some sections include optional exercises that encourage deeper exploration of the topics.
* Quick student instructions and workflows are in README_simple_workflow.md (see that file for Binder/Colab/local instructions).

The material is organised according to the structure of the course:

## 1. Introduction

Probability, random variables, and stochastic processes
This section revisits the probabilistic foundations required for the course. The notebooks illustrate core concepts through simple simulations, enabling students to visualise distributions, correlations, stationarity, and key properties of random processes.

## 2. Estimation Theory
### 2.1. Introduction, estimator properties, and the Cramér–Rao bound

Concepts such as bias, variance, efficiency, and consistency are introduced, together with practical computations of the CRB for different estimation problems.

### 2.2. Classical Estimation (ML, MoM, LS)

The notebooks present the Maximum Likelihood (ML), Method of Moments (MoM), and Least Squares (LS) estimators, using numerical examples that allow students to experiment with sample size, noise levels, and model assumptions.

### 2.3. Bayesian Estimation (MAP, MMSE, LMMSE)

Bayesian estimators are analysed and compared with their classical counterparts. Interactive cells let students modify priors, likelihoods, and model parameters to understand how assumptions affect performance.

## 3. Detection Theory
### 3.1. Binary Detection (Neyman–Pearson, Minimum a Priori Error)

The notebooks implement hypothesis-testing strategies, Receiver Operating Characteristic (ROC) curves, threshold selection, and performance evaluation under different noise conditions.

### 3.2. Waveform Detection

Continuous-time detection problems are illustrated with practical waveform examples, energy detectors, matched filters, and SNR-driven performance analyses.

## 4. Classification Theory
### 4.1. Fundamentals of Classification and Machine Learning

The notebooks introduce classification from a signal-processing perspective. Topics include discriminant functions, decision boundaries, linear classifiers, and connections with statistical learning.

