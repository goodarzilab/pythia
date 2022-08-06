from argparse import ArgumentParser
from apex import amp
from datetime import datetime
import joblib
from model import PythiaModel
from dnabert import DnaBert
from model_deepsea import DeepSEA
import numpy as np
import os
import pandas as pd
from sklearn import metrics
import torch
from torch import nn
from utils import load_model_from_file
from utils import get_optimizer
from utils import get_model_params
from utils import compile_paths
from utils import regularize_loss_deepsea


device = torch.device("cuda:0")
opt_level = 'O1'


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def split_dict(joblibpath, max_inputs=1e9):
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
        if len(use_idxs) > max_inputs:
            use_idxs = use_idxs[:max_inputs]
        for key, arr in curdict.items():
            addict[key] = arr[use_idxs]
        newdict[ratio_name] = addict
        lastidx = curidx
    return newdict


def load_model(inputsize, outdir, optimizer, lr, chkpaths,
               useRegConv=False, kernel_size=8, conv_width=256,
               dp=0.1, trainable=False, dil_start=5,
               dil_end=24, bulge_size=2, binarize_fd=False,
               model_name="Pythia", disable_conv_dp=False):
    torch.manual_seed(42)
    if model_name == "Pythia":
        net = PythiaModel(
            inputsize=inputsize,
            num_rbps=1,
            kernel_size=kernel_size,
            use_fixedConv=np.logical_not(useRegConv),
            dil_start=dil_start, dil_end=dil_end,
            bulge_size=bulge_size,
            dp=dp, trainable=trainable,
            binarize_fd=binarize_fd,
            disable_conv_dp=disable_conv_dp)
    elif model_name == "DeepSEA":
        net = DeepSEA(inputsize=inputsize)
    elif model_name == "DeepBind":
        from model_rbp import DeepBind
        net = DeepBind()
    elif model_name == "DNABERT":
        from dnabert import get_defaults
        list_args = get_defaults()
        net = DnaBert(list_args[0], list_args[1], list_args[2])
        print(net)
        lr = lr / 10
    else:
        print("Unrecognized model name {}".format(model_name))
        raise ValueError("Unrecognized model name")
    net.to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("{} has {} learnable parameters".format(model_name, n_params))
    optimizer = get_optimizer(
        optimizer, net, lr)
    net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    for eachpath in chkpaths:
        if os.path.exists(eachpath):
            net, optimizer = load_model_from_file(eachpath, net, optimizer)
            print("Loaded from {}".format(eachpath))
    if torch.cuda.device_count() > 1:
        print("Will use {} GPUs!".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)
    return net, optimizer


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
    train1 = get_seq(tensordict_train["Input"][idx_st:idx_end])
    resp_mfe = torch.from_numpy(
        tensordict_train["MFEs"][idx_st:idx_end]).to(device)
    resp_b = get_label(tensordict_train["Response"][idx_st:idx_end])
    resp_mfe = abs(resp_mfe.reshape(-1, 1))
    resp_mfe = resp_mfe.type(torch.float32)
    return train1, resp_b, resp_mfe


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


def train_model(joblibpath, dictpaths, net, optimizer,
                MINIBATCH=128, MAXEPOCHS=100,
                model_name="Pythia"):
    tensordict = split_dict(joblibpath)
    if model_name == "DNABERT":
        print("Setting batch for DNABERT to 8")
        MINIBATCH = 24
        MAXEPOCHS = 20
        tensordict = split_dict(joblibpath, max_inputs=96000)
    if torch.cuda.device_count() > 1:
        MINIBATCH = MINIBATCH * torch.cuda.device_count()
    tensordict_train = tensordict["Training"]
    torch.manual_seed(42)
    IMB = torch.tensor([.1, 1]).float()
    crit_bce = nn.CrossEntropyLoss(IMB).to(device)
    logpath = get_logpath(dictpaths)
    TOT_IDX = int(tensordict_train["Input"].shape[0] / MINIBATCH)
    _ = assess_performance(net, tensordict, logpath, 0, MINIBATCH)
    best_loss = 1500
    for epoch in range(MAXEPOCHS):
        loss_ce = 0
        for idx_batch in range(TOT_IDX):
            train1, resp_b, _ = process_minibatch(
                tensordict_train, idx_batch, MINIBATCH)
            optimizer.zero_grad()
            pred_m = net(train1)
            if torch.max(pred_m) == 0:
                print(pred_m)
            adce = crit_bce(pred_m, resp_b.long())
            if model_name == "DeepSEA":
                net, adce = regularize_loss_deepsea(
                    net, adce, l1=1e-8, l2=5e-7, l3=0.9)
            with amp.scale_loss(adce, optimizer) as loss:
                loss.backward()
            if torch.isnan(adce):
                print("{} {}".format(idx_batch, adce))
                raise ValueError("NA loss")
            torch.nn.utils.clip_grad_norm_(
               net.parameters(), max_norm=2, norm_type=2)
            optimizer.step()
            loss_ce += adce.item()
            del train1, resp_b, pred_m
            torch.cuda.empty_cache()
        loss_ce = loss_ce / idx_batch
        printstr = "Epoch {}: Loss: {}".format(
            epoch, loss_ce)
        print(printstr)
        tune_df, train_df = assess_performance(
            net, tensordict, logpath, epoch + 1,
            MINIBATCH)
        newloss = list(tune_df["BCE.Tune"])[0]
        if newloss < best_loss:
            best_loss = newloss
            checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(
                checkpoint,
                dictpaths["bestpath"])
        save_model(net, optimizer, dictpaths)
        if epoch % 10 == 0:
            plot_model(tune_df, logpath, "Tune")
            plot_model(train_df, logpath, "Train")


def save_model(net, optimizer, dictpaths):
    curpaths = dictpaths["chkpaths"] +\
        [dictpaths["modelpath"]]
    for modelpath in curpaths:
        checkpoint = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
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


def main(joblibpath, outdir, inputsize=256, optim_name="Adam",
         lr=0.01, useRegConv=False,
         kernel_size=16, conv_width=64, dp=0.5, trainable=False,
         dil_start=2, dil_end=48, bulge_size=2, binarize_fd=True,
         model_name="Pythia", disable_conv_dp=False):
    modelparams = get_model_params(
        inputsize, optim_name, lr, kernel_size, conv_width, dp,
        trainable, dil_start, dil_end, bulge_size, binarize_fd,
        model_name, disable_conv_dp)
    dictpaths = compile_paths(outdir, modelparams,
                              useRegConv=useRegConv)
    chkpaths = dictpaths["chkpaths"] +\
        [dictpaths["bestpath"], dictpaths["modelpath"]]
    net, optimizer = load_model(
        inputsize, outdir, optim_name, lr, chkpaths,
        useRegConv, kernel_size, conv_width, dp, trainable,
        dil_start, dil_end, bulge_size, binarize_fd, model_name,
        disable_conv_dp)
    train_model(
        joblibpath, dictpaths, net, optimizer,
        model_name=model_name)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train and evaluate a model "
        "on a single-rbp joblib-made dataset (dataPrep folder)")
    parser.add_argument(
        "joblibpath",
        help="Path to input joblib file")
    parser.add_argument(
        "outdir",
        help="Path to output directory for reporting "
        " and plotting model training performance")
    parser.add_argument(
        "--inputsize",
        type=int,
        default=50,
        help="Length of sequences to model. "
        "This must be the same as what provided "
        "in joblib files")
    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="Optimizer")
    parser.add_argument(
        "--lr",
        default=0.004,
        type=float,
        help="Learning rate")
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
    args = parser.parse_args()
    print(args)
    main(args.joblibpath, args.outdir, args.inputsize,
         args.optimizer, args.lr,
         args.useRegConv, args.kernel_size,
         args.conv_width, args.dp, args.trainable,
         args.dil_start, args.dil_end, args.bulge_size,
         args.binarize_fd, args.model_name,
         args.disable_conv_dp)
