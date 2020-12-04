# GNN_DOVE
<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/GNN--DOVE-v2.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>  

GNN-Dove is a computational tool using graph neural network that can evaluate the quality of docking protein-complexes.  

Copyright (C) 2020 Xiao Wang, Sean T Flannery, Daisuke Kihara, and Purdue University. 

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

Contact: Daisuke Kihara (dkihara@purdue.edu)


## Citation:
Xiao Wang, Sean T Flannery & Daisuke Kihara. Protein Docking Model Evaluation by Graph Neural Networks(2020).


## Introduction
Physical interactions of proteins are one of the key factors to determine many important cellular processes. Thus, it’s crucial to determine the structure of proteins for understanding molecular mechanisms of protein complexes. However, experimental approaches still take considerable amount of time and cost to determine the structures. To complement experimental approaches, various computational methods have been proposed to predict the structures of protein complexes. One of the challenges is to identify near-native structures from a large pool of predicted structures. We developed a deep learning-based approach named Graph Neural Network-based DOcking decoy Evaluation (GNN-DOVE). To evaluate a protein docking model, GNN-DOVE will extract the interface area and transform it to graph structures. The chemical properties of atoms and the distance information will be kept as the node and edge information in the graph. GNN-DOVE is trained and validated on docking models in the DockGround database. The extraordinary performance on testing set verified that the GNN model can learn more representative features for accurate predictions. 

## Overall Protocol
```
(1) Extract the interface region of protein-complex;
(2) Construct two graphs with/wihout intermolecular interactions based on interface region;
(3) Apply GNN with attention mechanism to process two input graphs;
(4) Output the evaluation score for input protein-complex.
```
<p align="center">
  <img src="figure/protocal.jpeg" alt="protocol" width="80%">
</p> 

## Network Architecture

<p align="center">
  <img src="figure/network.png" alt="network" width="80%">
</p> 
The illustration of graph neural network (GNN) with attention and gate-augmented mechanism (GAT)

## Pre-required software
Python 3 : https://www.python.org/downloads/    
rdkit: https://www.rdkit.org/docs/Install.html

## Installation  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:kiharalab/GNN_DOVE.git && cd GNN_DOVE
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip3 install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip3 install torch==1.7.0
pip3 install numpy==1.18.1
pip3 install scipy==1.4.1
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n GNN_DOVE python=3.6.10
conda activate GNN_DOVE
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate GNN_DOVE
conda deactivate(If you want to exit) 
```



