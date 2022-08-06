# Training and evaluating performance of Pythia, DeepSEA, and DNABERT


This repository contains scripts for training and evaluating the performance of Pythia, DeepSEA, and DNABERT.

Datasets containing positive and negative sequences for binding of each RBP can be downloaded from https://doi.org/10.5281/zenodo.6969819


## Training

Example training:

```
usage: train.py [-h] [--inputsize INPUTSIZE] [--optimizer OPTIMIZER] [--lr LR]
                [--kernel-size KERNEL_SIZE] [--conv-width CONV_WIDTH]
                [--trainable] [--dp DP] [--dil-start DIL_START]
                [--dil-end DIL_END] [--bulge-size BULGE_SIZE] [--useRegConv]
                [--binarize-fd]
                [--model-name {Pythia,DeepSEA,DNABERT,DeepBind}]
                [--disable-conv-dp]
                joblibpath outdir

Train and evaluate a model on a single-rbp joblib-made dataset (dataPrep
folder)

positional arguments:
  joblibpath            Path to input joblib file
  outdir                Path to output directory for reporting and plotting
                        model training performance

optional arguments:
  -h, --help            show this help message and exit
  --inputsize INPUTSIZE
                        Length of sequences to model. This must be the same as
                        what provided in joblib files
  --optimizer OPTIMIZER
                        Optimizer
  --lr LR               Learning rate
  --kernel-size KERNEL_SIZE
                        Kernel size
  --conv-width CONV_WIDTH
                        Convolutional width (filter size)
  --trainable           Will use trainable dilated layers
  --dp DP               Dropout
  --dil-start DIL_START
  --dil-end DIL_END
  --bulge-size BULGE_SIZE
  --useRegConv          Will use an arm of fixed dilated convolutional network
  --binarize-fd         If specified, fixed dilated layer will be binarized
  --model-name {Pythia,DeepSEA,DNABERT,DeepBind}
                        Model name (Pythia, DeepSEA, DeepBind, or DNABERT
  --disable-conv-dp     Specify to disable dropout in convolutions
JLPATH=AGO2_BoundaryFile_postitive_negative_seqs.joblib
OUTDIR=$PWD/trainedModel/AGO2
mkdir -p $OUTDIR
python train.py $JLPATH $OUTDIR --inputsize --binarize-fd --model-name Pythia
```

## Evaluation

Once the model is trained, you can save the path to the trained model into "MODELPATH" and run predict.py:


```
usage: predict.py [-h] [--inputsize INPUTSIZE] [--kernel-size KERNEL_SIZE]
                  [--conv-width CONV_WIDTH] [--trainable] [--dp DP]
                  [--dil-start DIL_START] [--dil-end DIL_END]
                  [--bulge-size BULGE_SIZE] [--useRegConv] [--binarize-fd]
                  [--model-name {Pythia,DeepSEA,DNABERT,DeepBind}]
                  [--disable-conv-dp]
                  joblibpath modelpath outpath

Use a model on a single-rbp joblib-made dataset (dataPrep folder)

positional arguments:
  joblibpath            Path to input joblib file
  modelpath             Path to trained and saved model
  outpath               Path to output tsv.gz for saving the predictions

optional arguments:
  -h, --help            show this help message and exit
  --inputsize INPUTSIZE
                        Length of sequences to model. This must be the same as
                        what provided in joblib files
  --kernel-size KERNEL_SIZE
                        Kernel size
  --conv-width CONV_WIDTH
                        Convolutional width (filter size)
  --trainable           Will use trainable dilated layers
  --dp DP               Dropout
  --dil-start DIL_START
  --dil-end DIL_END
  --bulge-size BULGE_SIZE
  --useRegConv          Will use an arm of fixed dilated convolutional network
  --binarize-fd         If specified, fixed dilated layer will be binarized
  --model-name {Pythia,DeepSEA,DNABERT,DeepBind}
                        Model name (Pythia, DeepSEA, DeepBind, or DNABERT
  --disable-conv-dp     Specify to disable dropout in convolutions
  --fasta               If specified, will assume joblibpath is a .fasta file
JLPATH=AGO2_BoundaryFile_postitive_negative_seqs.joblib
OUTDIR=$PWD/trainedModel/AGO2
OUTPATH=$OUTDIR/Pythia_Sequences_binding_and_prediction.tsv.gz
python predict_rbp.py $JLPATH $MODELPATH $OUTPATH --inputsize 256 --model-name Pythia
```


## Predicting on new FASTA files

The current script accepts data in the format of joblib.
The joblib object is a dictionary, containing a key named "Input", containing a character numpy array of the sequence, and a key names "Response", containing a character array of the same length as "Input" with values either as "Bound" or "Unbound".
On the provided datasets, the function split_dict will use random seed of 42 to split the data into Training, Tuning, and Validation sets. The predict script will only make predictions for the subset Containing "Validation" data.

If you want to feed your own fasta file and make predictions, you can provide a FASTA file instead of joblib and activate the option --fasta.
