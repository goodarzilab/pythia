# Requirements

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


