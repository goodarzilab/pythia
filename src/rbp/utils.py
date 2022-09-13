import torch
from torch import optim
import adabound
# from apex import amp
from collections import OrderedDict
import os


def regularize_loss_deepsea(net, loss, l1=1e-8, l2=5e-7, l3=0.9):
    # l3
    torch.nn.utils.clip_grad_norm_(
        net.Conv1.parameters(), l3)
    torch.nn.utils.clip_grad_norm_(
        net.Conv2.parameters(), l3)
    torch.nn.utils.clip_grad_norm_(
        net.Conv3.parameters(), l3)
    torch.nn.utils.clip_grad_norm_(
        net.Linear1.parameters(), l3)
    torch.nn.utils.clip_grad_norm_(
        net.Linear2.parameters(), l3)
    # l2
    l0_params = torch.cat(
        [x.view(-1) for x in net.Conv1.parameters()])
    l1_params = torch.cat(
        [x.view(-1) for x in net.Conv2.parameters()])
    l2_params = torch.cat(
        [x.view(-1) for x in net.Conv3.parameters()])
    l3_params = torch.cat(
        [x.view(-1) for x in net.Linear1.parameters()])
    l4_params = torch.cat(
        [x.view(-1) for x in net.Linear1.parameters()])
    l1_l0 = l1 * torch.norm(l0_params, 1)
    l1_l1 = l1 * torch.norm(l1_params, 1)
    l1_l2 = l1 * torch.norm(l2_params, 1)
    l1_l3 = l1 * torch.norm(l3_params, 1)
    l2_l0 = l2 * torch.norm(l0_params, 2)
    l2_l1 = l2 * torch.norm(l1_params, 2)
    l2_l2 = l2 * torch.norm(l2_params, 2)
    l2_l3 = l2 * torch.norm(l3_params, 2)
    l1_l4 = l1 * torch.norm(l4_params, 1)
    l2_l4 = l2 * torch.norm(l4_params, 2)
    loss = loss + l1_l0 + l1_l1 + l1_l2 +\
        l1_l3 + l2_l0 + l2_l1 + l2_l2 + l2_l3 +\
        l1_l4 + l2_l4
    return net, loss


def compile_paths(outdir, modelparams, loss_scalers=[1, 1], useRegConv=False):
    adname = "pythiaModel"
    for key, value in modelparams.items():
        adname += "_{}-{}".format(
            key, value)
    adname += "lossScale_{}_{}".format(
        loss_scalers[0], loss_scalers[1])
    if useRegConv:
        adname += "_noFixedConv"
    logdir = os.path.join(outdir, adname)
    os.makedirs(logdir, exist_ok=True)
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = "NA"
    chkdir = os.path.join(
        "/checkpoint/mkarimza",
        job_id)
    if not os.path.exists(chkdir):
        chkdir = os.path.join(logdir, "checkpoints")
        os.makedirs(chkdir, exist_ok=True)
    chkpaths = [
        os.path.join(chkdir, "{}_{}.pt".format(adname, each))
        for each in [0, 1]]
    modelpath = os.path.join(
        logdir, adname + "_latest.pt")
    bestpath = os.path.join(
        logdir, adname + "_best.pt")
    dictpaths = {
        "adname": adname,
        "chkpaths": chkpaths, "logdir": logdir,
        "modelpath": modelpath,
        "bestpath": bestpath}
    return dictpaths


def get_model_params(inputsize, optimizer, lr, kernel_size=8,
                     conv_width=256, dp=0.1, trainable=False,
                     dil_start=5, dil_end=24, bulge_size=2,
                     binarize_fd=False, model_name="Pythia",
                     disable_conv_dp=False):
    paramdict = {
        "optimizer": optimizer,
        "inputsize": inputsize,
        "lr": lr,
        "dp": dp,
        "conv_width": conv_width,
        "kernel": kernel_size,
        "dilationTraining": trainable,
        "dil_start": dil_start,
        "dil_end": dil_end,
        "bulge_size": bulge_size,
        "binarize_fd": binarize_fd,
        "model_name": model_name,
        "DisConvDP": disable_conv_dp}
    return paramdict


def get_optimizer(optname, net, lr):
    if optname == "Adabound":
        optimizer = adabound.AdaBound(
            net.parameters(), lr=lr, final_lr=0.1)
    elif optname == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optname == "Adagrad":
        optimizer = optim.Adagrad(
            net.parameters(), lr=lr*10)
    elif optname == "Adam":
        optimizer = optim.Adam(
            net.parameters(), lr=lr)
    else:
        raise ValueError("optimizer name not recognized")
    return optimizer


def load_model_from_file(chkpath, net, optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(chkpath)
        print("Successfully loaded {}".format(chkpath))
        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # amp.load_state_dict(checkpoint['amp'])
        print("Successfully loaded the model")
    else:
        net = torch.load(chkpath)
    return net, optimizer
