from argparse import ArgumentParser
# from apex import amp
from datetime import datetime
import joblib
import numpy as np
import os
import pandas as pd
from sklearn import metrics
import torch
from torch import nn
from train import load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt_level = 'O1'


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def split_dict(joblibpath):
    curdict = joblib.load(joblibpath)
    np.random.seed(42)
    shuffled_idxs = np.arange(curdict["Input"].shape[0])
    np.random.shuffle(shuffled_idxs)
    ratios = [0.64, 0.16, 0.2]
    ratio_names = ["Training", "Tuning", "Validation"]
    split_vals = [int(each * shuffled_idxs.shape[0]) for each
                  in ratios]
    split_vals[2] = shuffled_idxs.shape[0] - (
        split_vals[0] + split_vals[1])
    newdict = {}
    curidx = 0
    lastidx = 0
    for i in range(len(ratio_names)):
        ratio_name = ratio_names[i]
        curidx += split_vals[i]
        print("Splitting {} into {}-{}".format(
                ratio_name, lastidx, curidx))
        addict = {}
        use_idxs = shuffled_idxs[lastidx:curidx]
        for key, arr in curdict.items():
            addict[key] = arr[use_idxs]
        newdict[ratio_name] = addict
        lastidx = curidx
    return newdict


def get_n_params(model):
    pp = 0
    pp2 = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        if p.requires_grad:
            pp += nn
        else:
            pp2 += nn
    print("{} trainable and {} non-trainable".format(pp, pp2))
    return pp, pp2


def get_logpath(dictpaths):
    from datetime import datetime
    curtime = str(datetime.now())
    curtime = curtime.split(
        ":")[0].replace(" ", "_")
    logpath = os.path.join(
        dictpaths["logdir"],
        curtime + "modelTraining.log")
    return logpath


def get_seq(curseq):
    dict_seq = {"A": 0, "C": 1,
                "G": 2, "U": 3}
    seqlen = len(curseq[0])
    train1 = np.zeros(
        (curseq.shape[0], 4, len(curseq[0])))
    for i in range(curseq.shape[0]):
        for j in range(seqlen):
            if curseq[i][j] != "N":
                k = dict_seq[curseq[i][j]]
                train1[i, k, j] = 1
    train1 = torch.from_numpy(train1).to(device)
    train1 = train1.type(torch.float32)
    return train1


def get_label(response):
    resp_b = torch.from_numpy(
        response == "Bound").long().to(device)
    return resp_b


def process_minibatch(tensordict_train, idx_batch, MINIBATCH):
    idx_st = idx_batch * MINIBATCH
    idx_end = (idx_batch + 1) * MINIBATCH
    if idx_end > tensordict_train["Input"].shape[0]:
        idx_end = tensordict_train["Input"].shape[0]
    try:
        train1 = get_seq(tensordict_train["Input"][idx_st:idx_end])
    except Exception:
        print("Error at index {}, minibatch {}, max dim {}".format(
                idx_batch, MINIBATCH,
                tensordict_train["Input"].shape[0]))
        raise ValueError("Index error; see logs")
    resp_b = get_label(tensordict_train["Response"][idx_st:idx_end])
    return train1, resp_b, None


def assess_performance(net, tensordict, logpath, epoch, MINIBATCH=128):
    datanames = ["Training", "Tuning"]
    variables = ["Time", "Epoch", "BCE.Train",
                 "BCE.Tune", "AP.Train", "AP.Tune",
                 "Acc.Train", "Acc.Tune"]
    if epoch == 0:
        outlink = open(logpath, "w")
        print("Making {}".format(logpath))
        outlink.write("\t".join(variables) + "\n")
        outlink.close()
    dictvals = {}
    for dataname in datanames:
        print("Assessing {}".format(dataname))
        tensordict_train = tensordict[dataname]
        pred_b = np.zeros((tensordict_train["Input"].shape[0]))
        resp_b = tensordict_train["Response"] == "Bound"
        TOT_IDX = int(tensordict_train["Input"].shape[0] / MINIBATCH)
        if TOT_IDX > 100:
            TOT_IDX = int(TOT_IDX / 10)
        for idx_batch in range(TOT_IDX):
            train1, _, _ = process_minibatch(
                tensordict_train, idx_batch, MINIBATCH)
            pred_m = net(train1)
            idx_st = idx_batch * MINIBATCH
            idx_end = min([idx_st + MINIBATCH, resp_b.shape[0]])
            pred_m_soft = nn.functional.softmax(pred_m)
            pred_b[idx_st:idx_end] = pred_m_soft.detach().cpu().numpy()[:, 1]
            del train1, pred_m
            torch.cuda.empty_cache()
        bce = metrics.log_loss(
            resp_b[:idx_end], pred_b[:idx_end])
        ap = metrics.average_precision_score(
            resp_b[:idx_end],
            pred_b[:idx_end])
        acc = metrics.accuracy_score(
            resp_b[:idx_end],
            pred_b[:idx_end] > 0.5)
        dictvals["BCE.{}".format(dataname)] = bce
        dictvals["AP.{}".format(dataname)] = ap
        dictvals["Acc.{}".format(dataname)] = acc
        if dataname == "Training":
            train_df = pd.DataFrame(
                {"Response": resp_b[:idx_end],
                 "Pred.Response": sigmoid(pred_b[:idx_end])})
    tune_df = pd.DataFrame(
        {"Response": tensordict_train["Response"][:idx_end],
         "Pred.Response": sigmoid(pred_b[:idx_end])})
    tune_df.to_csv(
        logpath.replace(".log", ".tsv.gz"), sep="\t",
        index=None, compression="gzip")
    with open(logpath, "a+") as loglink:
        curvals = [str(datetime.now()).split(".")[0],
                   str(epoch)]
        for variable in variables[2:]:
            adname = variable.replace("Train", "Training")
            adname = adname.replace("Tune", "Tuning")
            tune_df[variable] = round(dictvals[adname], 4)
            train_df[variable] = round(dictvals[adname], 4)
            curvals.append(str(round(dictvals[adname], 4)))
        loglink.write("\t".join(curvals) + "\n")
    return tune_df, train_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_1st_output(train1, resp_mfe, tensordict_train,
                    MINIBATCH, net, outdir):
    seqs = tensordict_train["Input"][:MINIBATCH]
    out1 = net.conv1_dil(
        torch.nn.functional.pad(train1, (19, 20)))
    out2 = out1.reshape(out1.shape[0], -1)
    df = pd.DataFrame(
        {"Seq": seqs,
         "MFEs": resp_mfe.cpu().detach().numpy().reshape(-1)})
    out2_ar = out2.cpu().detach().numpy()
    for j in range(out2_ar.shape[1]):
        df["Var.{}".format(j)] = out2_ar[:, j]
    df.to_csv(
        os.path.join(outdir, "Model_output.tsv.gz"),
        sep="\t", compression="gzip")
    # For one of the sequences, visualize convolution output
    best_idx = np.where(df["MFEs"] == max(df["MFEs"]))[0]
    out3 = out1[best_idx].detach().cpu().numpy()[0]
    seq_ar = train1[best_idx].detach().cpu().numpy()[0]
    newdf = pd.DataFrame(
        np.transpose(seq_ar))
    newdf.columns = ["A", "C", "G", "U"]
    newdf["Seq"] = np.array(list(str(list(seqs[best_idx])[0])))
    for j in range(out3.shape[0]):
        newdf["Var.{}".format(j)] = out3[j, :]
    newdf.to_csv(
        os.path.join(outdir, "Model_output_1seq.tsv.gz"),
        sep="\t", compression="gzip")


def load_fasta(fastapath):
    fastadf = pd.read_csv(fastapath, header=None)
    num_lines = int(fastadf.shape[0] / 2)
    response_ar = np.array(["Bound"] * num_lines)
    idx_use = np.arange(1, num_lines + 2, step=2)[:num_lines]
    seq_ar = np.array(fastadf.iloc[idx_use, 0])
    outdict = {
        "Validation": {"Response": response_ar,
                       "Input": seq_ar}}
    return outdict


def apply_model(joblibpath, net, outpath, MINIBATCH=128,
                model_name="Pythia", fasta=False):
    if model_name == "DNABERT":
        MINIBATCH = 16
    if fasta:
        tensordict, tensordict_valid = load_fasta(joblibpath)
    else:
        tensordict = split_dict(joblibpath)
        tensordict_valid = tensordict["Validation"]
    pred_b = np.zeros((tensordict_valid["Input"].shape[0]))
    resp_b = tensordict_valid["Response"] == "Bound"
    TOT_IDX = int(tensordict_valid["Input"].shape[0] / MINIBATCH) + 1
    for idx_batch in range(TOT_IDX):
        idx_st = idx_batch * MINIBATCH
        if idx_st >= resp_b.shape[0]:
            break
        idx_end = min([idx_st + MINIBATCH, resp_b.shape[0]])
        train1, _, _ = process_minibatch(
            tensordict_valid, idx_batch, MINIBATCH)
        pred_m = net(train1)
        pred_m_soft = nn.functional.softmax(pred_m)
        pred_b[idx_st:idx_end] = pred_m_soft.detach().cpu().numpy()[:, 1]
        del train1, pred_m
        torch.cuda.empty_cache()
    # binarize_fd may produce nan
    pred_b[np.isnan(pred_b)] = 0
    bce = metrics.log_loss(
        resp_b[:idx_end], pred_b[:idx_end])
    ap = metrics.average_precision_score(
        resp_b[:idx_end],
        pred_b[:idx_end])
    acc = metrics.accuracy_score(
        resp_b[:idx_end],
        pred_b[:idx_end] > 0.5)
    tune_df = pd.DataFrame(
        {"Response": tensordict_valid["Response"][:idx_end],
         "Pred.Response": pred_b[:idx_end]})
    for each_key, each_val in tensordict_valid.items():
        tune_df[each_key] = each_val[:idx_end]
    tune_df["BCE"] = bce
    tune_df["Average.Precision"] = ap
    tune_df["Accuracy"] = acc
    print(outpath)
    tune_df.to_csv(
        outpath,
        sep="\t", compression="gzip", index=None)


def save_model(net, optimizer, dictpaths):
    curpaths = dictpaths["chkpaths"] +\
        [dictpaths["modelpath"]]
    for modelpath in curpaths:
        checkpoint = {
            'model': net.state_dict()
            # 'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict()
        }
        torch.save(
            checkpoint,
            modelpath)


def plot_model(tune_df, logpath, adname="Tune"):
    import seaborn as sns
    logdf = pd.read_csv(logpath, sep="\t")
    losses = list(logdf["BCE.Train"]) +\
        list(logdf["BCE.Tune"]) +\
        list(logdf["AP.Train"]) +\
        list(logdf["AP.Tune"])
    datasets = (list(["Training"] * logdf.shape[0]) +
                list(["Tuning"]) * logdf.shape[0]) * 2
    losstypes = list(
        ["BCE binding"] * logdf.shape[0] * 2) +\
        list(["Average precision"] * logdf.shape[0] * 2)
    newdict = {
        "Epoch": list(logdf["Epoch"]) * 4,
        "Loss": losses,
        "Dataset": datasets,
        "Loss type": losstypes}
    newdf = pd.DataFrame(newdict)
    sns_plot = sns.relplot(
        data=newdf,
        x="Epoch", y="Loss",
        kind="line",
        col="Loss type",
        hue="Dataset",
        facet_kws=dict(sharey=False),
        style="Dataset")
    imgpath = logpath.replace(
        ".log", "{}_trainingPref.png".format(adname))
    pdfpath = logpath.replace(
        ".log", "{}_trainingPref.pdf".format(adname))
    sns_plot.savefig(imgpath)
    sns_plot.savefig(pdfpath)


def main(joblibpath, outpath, inputsize, modelpath,
         useRegConv=False, kernel_size=16,
         conv_width=256, dp=0.5, trainable=False,
         dil_start=2, dil_end=48, bulge_size=2,
         binarize_fd=False, model_name="Pythia",
         disable_conv_dp=False, fasta=False):
    print(outpath)
    if os.path.exists(outpath):
        raise ValueError("Output exists")
    outdir = os.path.dirname(outpath)
    print(outdir)
    net, _ = load_model(
        inputsize, outdir, "Adam", 0.001, [modelpath],
        useRegConv, kernel_size, conv_width, dp, trainable,
        dil_start, dil_end, bulge_size, binarize_fd, model_name=model_name,
        disable_conv_dp=disable_conv_dp)
    apply_model(
        joblibpath, net, outpath, model_name=model_name,
        fasta=fasta)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Use a model "
        "on a single-rbp joblib-made dataset (dataPrep folder)")
    parser.add_argument(
        "joblibpath",
        help="Path to input joblib file")
    parser.add_argument(
        "modelpath",
        help="Path to trained and saved model")
    parser.add_argument(
        "outpath",
        help="Path to output tsv.gz "
        "for saving the predictions")
    parser.add_argument(
        "--inputsize",
        type=int,
        default=256,
        help="Length of sequences to model. "
        "This must be the same as what provided "
        "in joblib files")
    parser.add_argument(
        "--kernel-size",
        default=16,
        type=int,
        help="Kernel size")
    parser.add_argument(
        "--conv-width",
        default=64,
        type=int,
        help="Convolutional width (filter size)")
    parser.add_argument(
        "--trainable",
        action="store_true",
        help="Will use trainable dilated layers")
    parser.add_argument(
        "--dp",
        default=0.5,
        type=float,
        help="Dropout")
    parser.add_argument(
        "--dil-start",
        type=int,
        default=2)
    parser.add_argument(
        "--dil-end",
        type=int,
        default=48)
    parser.add_argument(
        "--bulge-size",
        type=int,
        default=2)
    parser.add_argument(
        "--useRegConv",
        action="store_true",
        help="Will use an arm of fixed "
        "dilated convolutional network")
    parser.add_argument(
        "--binarize-fd",
        action="store_true",
        help="If specified, fixed dilated layer "
        "will be binarized")
    parser.add_argument(
        "--model-name",
        choices=["Pythia", "DeepSEA", "DNABERT", "DeepBind"],
        default="Pythia",
        help="Model name (Pythia, DeepSEA, DeepBind, or DNABERT")
    parser.add_argument(
        "--disable-conv-dp",
        action="store_true",
        help="Specify to disable dropout in convolutions")
    parser.add_argument(
        "--fasta",
        action="store_true",
        help="If specified, will assume the provided joblibpath is .fasta")
    args = parser.parse_args()
    print(args)
    main(args.joblibpath, args.outpath, args.inputsize,
         args.modelpath,
         args.useRegConv, args.kernel_size,
         args.conv_width, args.dp, args.trainable,
         args.dil_start, args.dil_end, args.bulge_size,
         args.binarize_fd, args.model_name,
         args.disable_conv_dp, fasta=args.fasta)
