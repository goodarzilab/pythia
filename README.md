# Pythia model

## Introduction

Pythia is a neural network capable of encoding the context-free grammar of RNA structure.

A dilated convolutional operation can detect base pairing of two nucleotides at a fixed distance.
A neural network layer capable of learning structure, therefore,
must detect base pairing of nucleotides at a range of distances,
as opposed to a single dilation width.

Through custom design of fixed weights within the weight tensor
of a convolutional layer, we developed an efficient means for
simultaneous detection of base pairing at a range of dilation
widths while allowing convolutional filters to capture asymmetrical bulges as well.

Our fixed dilated convolutional layer can capture base pairing of
nucleotides within a defined range of distances, allowing for
asymmetrical bulges of defined lengths to occur within the RNA structure as well.
The fixed dilated convolutional layer, therefore, learns the context-free
grammar of RNA structure.
Combined with other convolutional and dense layers, Pythia can learn minimum
free energy of RNA folding, complex RNA structures such as pseudo-knots, and learn binding preferences of RBPs


## Examples

Please see our notebooks ![here](https://github.com/goodarzilab/pythia/tree/master/src/rbp/notebooks).


## Requirements

Run the following commands to install the dependencies:


```
conda create -n pytorch -c pytorch pytorch=1.4.0
pip install apex
pip install adabound
conda install -c bioconda ushuffle
conda install -c conda-forge transformers
conda install -c anaconda scikit-learn
conda install -c anaconda seaborn
```


If using a GPU, make sure `nvidia-smi` returns a meaningful output such as:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:2F:00.0 Off |                    0 |
| N/A   53C    P0    28W / 250W |      0MiB / 12198MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


To run the notebooks, also add:


```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pytorch
jupyter notebook
```


