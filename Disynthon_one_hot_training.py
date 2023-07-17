import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(0)
import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


OHE = OneHotEncoder()

hidden_num, drop_rate, learning_rate = 128, 0.5, 0.001
L1_rate, n_epochs = 10, 50
dfpath = "example.csv"
outpath = ""
groupname = "py1355"


def disynthon_train(dfpath, hidden_num, drop_rate, learning_rate, groupname, L1_rate, n_epochs, outpath):
    class OnehotDEL(BaseEstimator, TransformerMixin): 
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
            pass

        def fit(self, X, *args, **kwargs):
            return self

        def transform(self, X, length=2):  # X is an iterable object, like a list, a series
            X = X.reset_index(drop=True)
            X["CodeA"] = X["CodeA"] - 1
            X["CodeB"] = X["CodeB"] + self.a - 1
            X["CodeC"] = X["CodeC"] + self.a + self.b - 1
            A_index = X["CodeA"].reset_index().values
            A_index = A_index[A_index[:, 1] != -1]
            B_index = X["CodeB"].reset_index().values
            B_index = B_index[B_index[:, 1] != 259]
            C_index = X["CodeC"].reset_index().values
            C_index = C_index[C_index[:, 1] != 260 + 259]
            values = [1] * length * len(X)
            ABC_index = np.concatenate([A_index, B_index, C_index], axis=0).transpose()

            i2 = torch.LongTensor(ABC_index)
            v2 = torch.FloatTensor(values)
            shape2 = (len(X), (self.a + self.b + self.c))
            t2 = torch.sparse_coo_tensor(i2, v2, torch.Size(shape2))
            return t2

    a = 260
    b = 260
    c = 451
    num_pip = OnehotDEL(a, b, c)

    post_str = groupname
    pre_str = "Pre"

    sms_df = pd.read_csv(dfpath)
    print(sms_df.columns)
    sms_df.tail()
    w1 = np.sum(sms_df[pre_str]) / np.sum(sms_df[post_str])
    print(w1)
    smile_file = sms_df.copy()
    t2 = num_pip.transform(smile_file).to_dense()

    def calcu_EF(post, pre, file):
        # file：df；post,pre:col names
        import numpy as np
        sum_post = np.sum(file[post])
        a = (file[pre] + 0.375) * sum_post
        sum_pre = np.sum(file[pre])
        b = (file[post] + 0.375) * sum_pre
        w = sum_post / sum_pre
        values = b / a
        return values

    from sklearn.model_selection import train_test_split

    smile_file["EF_nll"] = calcu_EF(post_str, pre_str, smile_file)
    smile_file = smile_file.reset_index(drop=True)
    print(smile_file.head(), smile_file.shape)
    train_set, test_set = train_test_split(smile_file, test_size=0.2, random_state=42)
    smile_file_train = train_set.copy()
    test_file = test_set.copy()
    train_list = t2[train_set.index, :]
    train_k1 = (train_set[post_str].values).reshape(-1, 1)
    train_k2 = (train_set[pre_str].values).reshape(-1, 1)
    train_label = (train_set["EF_nll"].values).reshape(-1, 1)
    test_list = t2[test_set.index, :]
    test_k1 = (test_set[post_str].values).reshape(-1, 1)
    test_k2 = (test_set[pre_str].values).reshape(-1, 1)
    test_label = (test_set["EF_nll"].values).reshape(-1, 1)
    from torch.utils.data import DataLoader, Dataset

    size = 256
    from numpy.core.numeric import False_

    class train_dataset(Dataset):
        def __init__(self):
            self.Data = train_list
            self.Post = np.asarray(train_k1)
            self.Pre = np.asarray(train_k2)
            self.Label = np.asarray(train_label)

        def __getitem__(self, index):
            txt = self.Data[index]
            post = torch.tensor(self.Post[index]).float()
            pre = torch.tensor(self.Pre[index]).float()
            label = torch.tensor(self.Label[index]).float()
            return txt, post, pre, label

        def __len__(self):
            return len(self.Data)

    class test_dataset(Dataset):
        def __init__(self):
            self.Data = test_list
            self.Post = np.asarray(test_k1)
            self.Pre = np.asarray(test_k2)
            self.Label = np.asarray(test_label)

        def __getitem__(self, index):
            txt = self.Data[index]
            post = torch.tensor(self.Post[index]).float()
            pre = torch.tensor(self.Pre[index]).float()
            label = torch.tensor(self.Label[index]).float()
            return txt, post, pre, label

        def __len__(self):
            return len(self.Data)

    train_data = train_dataset()
    train_data_loader = DataLoader(train_data, batch_size=size, shuffle=True)
    test_data = test_dataset()
    test_data_loader = DataLoader(test_data, batch_size=size, shuffle=False)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Sequential(torch.nn.Linear(a + b + c, hidden_num, ), nn.BatchNorm1d(hidden_num))
            self.fc2 = torch.nn.Sequential(
                torch.nn.Dropout(drop_rate),
                torch.nn.Linear(hidden_num, 1), )
            # 输出数字的话就应该（1024，1），底下用relu

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

    def init_weights(m):
        # if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.fc1[0].weight)
        m.fc1[0].bias.data.fill_(0.)
        torch.nn.init.kaiming_uniform_(m.fc2[1].weight)
        m.fc2[1].bias.data.fill_(0.1)

    # 1024，256 to 256,1,new two layers model

    model = Net()

    init_weights(model)
    targetstr = "ALLDisynthons"
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    datestr = stamp[0:8]
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(f"{outpath}/runs/{groupname}{datestr}{targetstr}z2_{L1_rate}reg")
    print(groupname, L1_rate, targetstr, datestr)
    import torch

    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    MSE = nn.MSELoss()

    def loss_fn(out, post, pre, w=w1):
        R_d = out / w1
        zstat = 2 * (torch.sqrt(post + 3 / 8) - torch.sqrt((pre + 3 / 8) * R_d))
        zstat = zstat / torch.sqrt(1 + R_d)
        loss_z = torch.pow(zstat, 2) / 2
        loss_L1 = L1_rate * torch.abs(out)
        loss = loss_z + loss_L1
        return loss

    import time

    start = time.time()

    for epoch in range(n_epochs):
        training_loss = 0.0
        valid_loss = 0.0
        mse_diff = 0
        mse_diff2 = 0
        model = model.to(device)
        model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            input, post, pre, target = batch
            input, post, pre, target = input.to(device), post.to(device), pre.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, post, pre)
            diff = MSE(output, target)
            torch_sum = torch.sum(loss, dim=0)
            (torch_sum).backward()
            optimizer.step()
            training_loss += torch_sum
            mse_diff += diff.sum()
        training_loss /= len(train_data_loader.dataset)
        mse_diff /= len(train_data_loader)
        print(f"epoch={epoch + 1}")
        writer.add_scalar("MAPloss/train", training_loss, epoch)
        writer.add_scalar("MSEloss/train", mse_diff, epoch)
        model.eval()
        for batch in test_data_loader:
            input, post, pre, target = batch
            input, post, pre, target = input.to(device), post.to(device), pre.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, post, pre)
            diff = MSE(output, target)
            torch_sum = torch.sum(loss, dim=0)
            valid_loss += torch_sum
            mse_diff2 += diff.sum()
        valid_loss /= len(test_data_loader.dataset)
        mse_diff2 /= len(test_data_loader)
        writer.add_scalar("MAPloss/test", valid_loss, epoch)
        writer.add_scalar("MSEloss/test", mse_diff2, epoch)


    end = time.time()
    print(f"{len(smile_file)}lines_cost_{str(end - start)}")
    mkdir(f"{outpath}/models")
    path = f"{outpath}/models/{groupname}{datestr}{targetstr}{L1_rate}step={n_epochs}.pt"
    torch.save(model.state_dict(), path)
    writer.flush()
    writer.close()
