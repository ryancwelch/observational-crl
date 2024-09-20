# Algorithms for Causal Disentanglement from Purely Observational Data

This repository contains an implementation of the necessary algorithms to recover upstream-layer representations of latent causal graphs as formulated in the paper [Identifiability Guarantees for Causal Disentanglement from Purely Observational Data](https://openreview.net/forum?id=M20p6tq9Hq&noteId=0KDq5QK0Sw). The algorithms are capable of recovering both upstream-layer represensentations of latent variables and single-layer representations of exogenous noise terms in non-linear additivive Gaussian noise models transformed via linear mixing. The method is intended to perform with high levels of accurately with the use of reliable second order score estimators.

## Abstract 
> Causal disentanglement aims to learn about latent causal factors behind data, holding the promise to augment existing representation learning methods in terms of interpretability and extrapolation. Recent advances establish identifiability results assuming that interventions on (single) latent factors are available; however, it remains debatable whether such assumptions are reasonable due to the inherent nature of intervening on latent variables. Accordingly, we reconsider the fundamentals and ask what can be learned using just observational data. <br>
> We provide a precise characterization of latent factors that can be identified in nonlinear causal models with additive Gaussian noise and linear mixing, without any interventions or graphical restrictions. In particular, we show that the causal variables can be identified up to a _layer_-wise transformation and that further disentanglement is not possible. We transform these theoretical results into a practical algorithm consisting of solving a quadratic program over the score estimation of the observed data. We provide simulation results to support our theoretical guarantees and demonstrate that our algorithm can derive meaningful causal representations from purely observational data.

 ## Instructions for Setting Up the Environment

 To create a new conda environment, open terminal and run this command:
 
```
 conda create --name ocrl python=3.12.2
```
Install PyTorch following their [official instructions](https://pytorch.org/get-started/locally/). We recommend using the following versions or torch and torchvision:
```
torch==2.2.2
torchvision==0.17.2
```
Install Gurobi, which is the state-of-the-art optimization solver for quadratic programs, using their [official instuctions](https://www.gurobi.com/).

Install the remaining requirements using the following command:
```
 pip install -r requirements.txt
```

 ## Running Experiments

Refer to the notebooks folder for experiments of recoving upstream-layer representations of latent variables and single-layer representations of exogenous noise terms in synthethic graphs. ```Score Oracle Simulations.ipynb ``` presents the experiments using perfect score estimation, where the code listed reproduces the results from the specified paper. ```Score Estimation Simulations.ipynb ``` presents experiments using data driven score estimation methods. These notebooks provide templates to easily experiment with different DAG types, number of training examples, and score estimators.

To use these algorithms on your data, refer to the models folder, where ```causal_peeler.py``` contains the methods to learn upstream-layer representations, and ```noise_estimator.py``` contains the methods to learn single-layer representations of exogenous noise variables.


