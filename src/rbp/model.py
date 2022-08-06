import numpy as np
import torch
import torch.nn as nn


class fixedDilatedConv(nn.Module):
    def __init__(self, in_channel=4, dil_start=5,
                 dil_end=24,
                 trainable=False, bulge_size=2,
                 binarize_fd=False):
        super(fixedDilatedConv, self).__init__()
        self.binarize_fd = binarize_fd
        self.in_channel = in_channel
        self.out_channel = int(
            (dil_end - dil_start + 1) * 6 * bulge_size * 3)
        self.out_channel = self.out_channel - (
            (dil_end - dil_start + 1) * 6)
        self.dil_start = dil_start
        self.dil_end = dil_end
        self.bulge_size = bulge_size
        self.kernel_size = int(
            dil_end - dil_start + 1) * 2
        self.mid = int(self.kernel_size / 2)
        weights = torch.Tensor(
            torch.zeros(
                self.out_channel,
                self.in_channel,
                self.kernel_size))
        weights = self.inject_weights(weights)
        self.weights = torch.nn.Parameter(weights)
        if not trainable:
            self.weights.requires_grad = False

    def inject_weights(self, weights):
        A, C, G, U = 0, 1, 2, 3
        weight_codes = [
            (A, U, 1), (U, A, 1.25), (C, G, 1.5),
            (G, C, 1.75), (G, U, .5), (U, G, .75)]
        if self.binarize_fd:
            weight_codes = [
                (A, U, 1), (U, A, 1), (C, G, 1),
                (G, C, 1), (G, U, 1), (U, G, 1)]
        self.weight_names = []
        kernel_idx = 0
        for nuc_tuple in weight_codes:
            nuc1, nuc2, adweight = nuc_tuple
            for i in range(self.dil_start, self.dil_end + 1):
                part_1 = round(i / 2)
                part_2 = i - part_1 + 1
                for each_bulge in range(1, self.bulge_size + 1):
                    if each_bulge == 1:
                        curbulges = [-each_bulge, 0, each_bulge]
                    else:
                        curbulges = [-each_bulge, each_bulge]
                    for bulge in curbulges:
                        idx_1 = np.array(
                            [int(self.kernel_size / 2) - part_1,
                             int(self.kernel_size / 2) + part_2])
                        if bulge > 0:
                            idx_1[idx_1 > self.mid] += bulge
                        elif bulge < 0:
                            idx_1[idx_1 < self.mid] -= bulge
                        idx_1 = idx_1[idx_1 < self.kernel_size]
                        idx_1 = idx_1[idx_1 > 0]
                        idx_0 = np.setdiff1d(
                            np.arange(self.kernel_size), idx_1)
                        weights[kernel_idx, :, idx_0] = 0
                        weights[kernel_idx, nuc1, idx_1[0]] = adweight
                        weights[kernel_idx, nuc2, idx_1[-1]] = adweight
                        adname = "{}_{} dil {} bulge {}".format(
                            nuc1, nuc2, i, bulge)
                        self.weight_names.append(adname)
                        kernel_idx += 1
        return weights[:kernel_idx, :, :]

    def forward(self, x):
        out = torch.nn.functional.conv1d(
            x, self.weights)
        if self.binarize_fd:
            out[out < torch.max(out)] = 0
            out[out > 0] = 1
        return out


class stackedDilatedConv(nn.Module):
    def __init__(self, dil_start=5, dil_end=24,
                 kernel_size=16, width=30,
                 inputsize=50):
        super(stackedDilatedConv, self).__init__()
        # self.padding = padding
        self.dil_start = dil_start
        self.dil_end = dil_end
        self.kernel_size = kernel_size
        self.width = width
        self.inputsize = inputsize
        self.padding = 0
        self.padding = self.get_lout(
            dil_end)
        if self.padding < 0:
            self.padding = int(abs(self.padding) + (inputsize * 2))
        self.pad1 = int(self.padding / 2)
        self.pad2 = self.padding - self.pad1
        conv_id = 0
        for i in range(dil_start, dil_end + 1):
            adlayer = nn.Conv1d(
                4, self.width,
                kernel_size=self.kernel_size,
                dilation=i)
            setattr(self, "Dilation.{}".format(conv_id), adlayer)
            conv_id += 1
        self.num_layers = conv_id
        self.width_out = int(self.num_layers * self.width)
        self.kernel_out = self.get_lout(self.dil_start)

    def get_lout(self, dil):
        lout = self.inputsize + 2 * self.padding -\
            dil * (self.kernel_size - 1)
        return lout

    def forward(self, x):
        x_padded = nn.functional.pad(
            x, (self.pad1, self.pad2))
        out = torch.zeros(
            x.shape[0],
            self.width_out,
            self.kernel_out).to(x.device)
        for i in range(self.num_layers):
            width_st = int(i * self.width)
            width_end = width_st + self.width
            adname = "Dilation.{}".format(i)
            temp = getattr(self, adname)(x_padded)
            dim_diff = int(
                (self.kernel_out - temp.shape[-1]) / 2)
            dim_diff_2 = int(self.kernel_out - dim_diff - temp.shape[-1])
            out[:, width_st:width_end, ] = nn.functional.pad(
                temp, (dim_diff, dim_diff_2))
        return out


class ConvArm1(nn.Module):
    def __init__(self, inputsize=50, kernels=[16, 10, 5],
                 widths=[64, 64, 64],
                 pools=[2, 2, 4], dropout=0.5,
                 disable_conv_dp=False):
        super(ConvArm1, self).__init__()
        self.inputsize = inputsize
        self.kernels = kernels
        self.pools = pools
        self.widths = widths
        self.disable_conv_dp = disable_conv_dp
        list1 = [
            nn.Conv1d(
                self.inputsize,
                widths[0], kernel_size=self.kernels[0]),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU()]
        list2 = [
            nn.Conv1d(
                widths[0],
                widths[1], kernel_size=self.kernels[1]),
            nn.BatchNorm1d(widths[1]),
            nn.ReLU()]
        list3 = [
            nn.Conv1d(
                widths[1],
                widths[2], kernel_size=self.kernels[2]),
            nn.BatchNorm1d(widths[2]),
            nn.ReLU()]
        if self.disable_conv_dp:
            list1.append(nn.Dropout(p=dropout))
            list2.append(nn.Dropout(p=dropout))
            list3.append(nn.Dropout(p=dropout))
        list1.append(nn.MaxPool1d(pools[0]))
        list2.append(nn.MaxPool1d(pools[1]))
        list3.append(nn.MaxPool1d(pools[2]))
        self.a1conv1 = nn.Sequential(*list1)
        self.a1conv2 = nn.Sequential(*list2)
        self.a1conv3 = nn.Sequential(*list3)

    def forward(self, x):
        out = self.a1conv1(x)
        out = self.a1conv2(out)
        out = self.a1conv3(out)
        return out


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


class DeepBind(nn.Module):
    def __init__(self, num_motifs=16, motif_len=24,
                 lin_dim=32, dp=0.5):
        super(DeepBind, self).__init__()
        self.channel_out = num_motifs
        self.kernel_size = motif_len
        self.lin_dim = lin_dim
        self.dp = dp
        self.conv = nn.Sequential(
            nn.Conv1d(
                4, self.channel_out,
                kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.lin_dim))
        self.feedForward = nn.Sequential(
            nn.Linear(int(self.lin_dim * self.channel_out),
                      self.lin_dim),
            nn.ReLU(),
            nn.Dropout(self.dp),
            nn.Linear(self.lin_dim, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x_padded = nn.functional.pad(
            x, (self.kernel_size, self.kernel_size))
        out = self.conv(x_padded)
        out = out.reshape(out.shape[0], -1)
        out = self.feedForward(out)
        return out


class PythiaModel(nn.Module):
    def __init__(self, init_channels=64, kernel_size=16,
                 num_rbps=1, inputsize=50,
                 use_fixedConv=True, dp=0.25, trainable=False,
                 dil_start=5, dil_end=24, bulge_size=2,
                 binarize_fd=False, disable_conv_dp=False):
        super(PythiaModel, self).__init__()
        self.disable_conv_dp = disable_conv_dp
        self.use_fixedConv = use_fixedConv
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.num_rbps = num_rbps
        self.inputsize = inputsize
        self.out_channel_fd = int(
            (dil_end - dil_start + 1) * 6 * bulge_size * 3)
        # Dilating layers
        if self.use_fixedConv:
            if trainable:
                self.conv1_dil = fixedDilatedConv(
                    dil_start=dil_start, dil_end=dil_end,
                    bulge_size=bulge_size, trainable=True,
                    binarize_fd=binarize_fd)
            else:
                self.conv1_dil = fixedDilatedConv(
                    dil_start=dil_start, dil_end=dil_end,
                    bulge_size=bulge_size, trainable=False,
                    binarize_fd=binarize_fd)
        else:
            out_channel = int(
                (dil_end - dil_start + 1) * 6 * bulge_size * 3)
            self.dil_dim = out_channel - ((dil_end - dil_start + 1) * 6)
            self.conv1_reg = stackedDilatedConv(
                inputsize=self.inputsize,
                dil_start=dil_start, dil_end=dil_end)
        self.pool_dil = nn.AdaptiveMaxPool1d(60)
        self.process_dil = nn.Sequential(
            nn.BatchNorm1d(60),
            nn.ReLU())
        # arm 1 dilations
        self.arm1_dil = ConvArm1(
            60,
            kernels=[16, 10, 5],
            pools=[2, 2, 2],
            dropout=dp,
            disable_conv_dp=self.disable_conv_dp)
        self.arm2_reg = ConvArm1(
            4,
            widths=[128, 64, 64],
            kernels=[12, 7, 5],
            pools=[2, 2, 2],
            dropout=dp,
            disable_conv_dp=self.disable_conv_dp)
        self.lin_dim = self.get_lindim()
        self.dense_1 = nn.Linear(
            self.lin_dim,
            64)
        self.dense_relu_1 = nn.ReLU()
        self.dense_2 = nn.Linear(
            64, 32)
        self.dense_relu_2 = nn.ReLU()
        self.dense_3 = nn.Linear(
            32, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def get_lindim(self):
        x = torch.rand(1, 4, self.inputsize)
        out = self.forward_dil(x)
        return out.shape[1]

    def get_bn_dim(self):
        x = torch.rand(
            1, 4, self.inputsize)
        x_padded = nn.functional.pad(
            x, (25, 24))
        if self.use_fixedConv:
            out = self.conv1_dil(x_padded)
        else:
            out = self.conv1_reg(x_padded)
        return out.shape[2]

    def forward_dil(self, x):
        x_padded = nn.functional.pad(
            x, (x.shape[2], x.shape[2]))
        if self.use_fixedConv:
            out = self.conv1_dil(x_padded)
        else:
            out = self.conv1_reg(x_padded)
        out = self.pool_dil(out)
        out = out.reshape(
            out.shape[0], out.shape[2], -1)
        out = self.process_dil(out)
        out1 = self.arm1_dil(out)
        out2 = self.arm2_reg(x)
        out_merged = torch.cat(
            (out1, out2), 2)
        out = out_merged.view(
            out_merged.shape[0], -1)
        return out

    def forward(self, x):
        out = self.forward_dil(x)
        out_mfe = self.dense_1(out)
        out_mfe = self.dense_relu_1(out_mfe)
        out_mfe = self.dense_2(out_mfe)
        out_mfe = self.dense_relu_2(out_mfe)
        out_mfe = self.dense_3(out_mfe)
        return out_mfe


def conv1_dim(lin, kernel, pad=1, dil=1, stride=1):
    lout = (
        ((lin + 2 * pad - dil * (kernel - 1) - 1) / stride) + 1)
    return lout


def test_model(inputsize, num_rbps=1, use_fixedConv=True,
               kernel_size=16, trainable=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using gpu")
    else:
        device = torch.device("cpu")
    print("i {} n {} fixed {} k {} trainable {}".format(
            inputsize, num_rbps, use_fixedConv, kernel_size,
            trainable))
    net = PythiaModel(
        inputsize=inputsize, num_rbps=num_rbps,
        kernel_size=kernel_size,
        use_fixedConv=use_fixedConv,
        trainable=trainable)
    net.to(device)
    print(net)
    in_tensor = torch.rand(
        (10, 4, inputsize)).to(device)
    out_mfe = net(in_tensor)
    print("Input: {}\nOutput: {}".format(
            in_tensor.shape, out_mfe.shape))


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_x():
    np.random.seed(42)
    x = np.zeros((32, 4, 50))
    for i in range(32):
        temp_seq = np.random.choice(
            np.array([0, 1, 2, 3]),
            50, replace=True)
        ad_hot = to_one_hot(temp_seq, 4)
        x[i] = ad_hot.reshape(4, -1)
    return x


def test_fixed():
    np.random.seed(42)
    addict = {"A": 0, "C": 1, "G": 2, "U": 3}
    str_seq = np.random.choice(
        np.array(["A", "C", "G", "U"]),
        50, replace=True)
    str_seq[15:32] = list("ACUUAGGUAACCUAGAA")
    temp_seq = np.array(
        [addict[each] for each in str_seq])
    ad_hot = to_one_hot(temp_seq, 4)
    x = torch.from_numpy(ad_hot.T).reshape(1, 4, -1)
    x = torch.zeros(1, 4, 50)
    for i in range(50):
        x[0, :, i] = torch.from_numpy(ad_hot[i])
    self = fixedDilatedConv(dil_start=2, dil_end=21)
    dict_data = {"Seq": str_seq[:40]}
    for i in np.arange(self.weights.shape[0]):
        adname = self.weight_names[i]
        cur_weight = self.weights[i]
        res_val = torch.mul(cur_weight, x[0, :, :40])
        dict_data[adname] = sum(res_val).detach().numpy()
    import pandas as pd
    outdf = pd.DataFrame(dict_data)
    outdf.to_csv("Example_data_stemloop.tsv", sep="\t")


def test_fixed_2():
    np.random.seed(42)
    addict = {"A": 0, "C": 1, "G": 2, "U": 3}
    str_seq = "ACGCUUACAAAGUAAGCGU"
    filters = ["NNNNCNNNGNNNN",
               "NNNANNNNNUNNN",
               "NNUNNNNNNNANN",
               "NUNNNNNNNNNAN",
               "CNNNNNNNNNNNG"]
    temp_seq = np.array(
        [addict[each] for each in str_seq])
    ad_hot = to_one_hot(temp_seq, 4)
    x = torch.from_numpy(ad_hot.T).reshape(1, 4, -1)
    filters_temp = torch.zeros(5, 4, 19)
    i = 0
    for each_f in filters:
        j = 0
        for each_nuc in list(each_f):
            if each_nuc != "N":
                idx_nuc = addict[each_nuc]
                filters_temp[i, idx_nuc, j] = 1
            j += 1
        i += 1
    x_padded = nn.functional.pad(x, (9, 9))
    outval = torch.nn.functional.conv1d(x_padded.float(), filters_temp.float())
    print(outval[0].T)


if __name__ == "__main__":
    test_model(50, use_fixedConv=False, kernel_size=16)
    test_model(50, use_fixedConv=True, kernel_size=16)
    print("Script passed basic tests")
