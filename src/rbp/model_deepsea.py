import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# from https://github.com/PuYuQian/PyDeepSEA/blob/master/DeepSEA_train.py
class DeepSEA(nn.Module):
    def __init__(self, kernel_size=8,
                 out_channels=[320, 480, 960], inputsize=256):
        super(DeepSEA, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.Conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=self.out_channels[0],
            kernel_size=self.kernel_size)
        self.Conv2 = nn.Conv1d(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size)
        self.Conv3 = nn.Conv1d(
            in_channels=self.out_channels[1],
            out_channels=self.out_channels[2],
            kernel_size=self.kernel_size)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.lin_size = self.get_lin_size(inputsize)
        self.Linear1 = nn.Linear(self.lin_size, 925)
        self.Linear2 = nn.Linear(925, 2)

    def get_lin_size(self, inputsize):
        x = torch.rand(4, 4, inputsize)
        out = self.Maxpool(
            self.Conv2(self.Maxpool(self.Conv1(x))))
        out = self.Conv3(out)
        outdim = int(out.shape[1] * out.shape[2])
        print("Linear size: {}".format(outdim))
        return outdim

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = x.reshape(input.shape[0], self.lin_size)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_xy():
    np.random.seed(42)
    x = np.zeros((1024, 4, 256))
    y = np.zeros((1024, 1))
    for i in range(1024):
        temp_seq = np.random.choice(
            np.array([0, 1, 2, 3]),
            256, replace=True)
        ad_hot = to_one_hot(temp_seq, 4)
        y[i] = np.sum(ad_hot[:, 2]) * np.sum(ad_hot[:, 3]) / 80
        x[i] = ad_hot.T
    return x, y


def train_model(model_name="DeepSEA"):
    from torch.nn import MSELoss
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using gpu")
    else:
        device = torch.device("cpu")
    x, y = get_xy()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)
    if model_name == "DeepSEA":
        net = DeepSEA()
    elif model_name == "DNABERT":
        from dnabert import DnaBert
        from dnabert import get_defaults
        list_args = get_defaults()
        net = DnaBert(list_args[0], list_args[1], list_args[2])
    elif model_name == "Pythia":
        from model import PythiaModel
        net = PythiaModel(
            inputsize=256, dil_start=2, dil_end=48, binarize_fd=True)
    else:
        print("Only DeepSEA and DNABERT and Pythia is acceptable")
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model has {} learnable parameters".format(n_params))
    net = net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.001,
        weight_decay=0.01, eps=1e-4)
    crit_mse = MSELoss().to(device)
    MINIBATCH = 16
    TOT_IDX = int(x.shape[0] / MINIBATCH) + 1
    for epoch in range(10):
        loss_mse = 0
        for idx_batch in range(TOT_IDX):
            idx_st = idx_batch * MINIBATCH
            idx_end = min([idx_st + MINIBATCH, x.shape[0]])
            if idx_st >= x.shape[0]:
                break
            train1 = x[idx_st:idx_end].to(device)
            resp = y[idx_st:idx_end].to(device).float()
            pred = net(train1)
            admse = crit_mse(pred, resp)
            optimizer.zero_grad()
            admse.backward()
            optimizer.step()
            loss_mse += admse.item()
            del train1, resp, pred
        avg_loss = loss_mse / idx_batch
        # scheduler.step(avg_loss)
        # cur_lr = round(scheduler.get_last_lr()[0], 6)
        print("Loss at epoch {} is {}".format(
                epoch, avg_loss))
    del net
    torch.cuda.empty_cache()


if __name__ == "__main__":
    modelnames = ["DeepSEA", "Pythia", "DNABERT"]
    for modelname in modelnames:
        print("Training {}".format(modelname))
        train_model(modelname)
