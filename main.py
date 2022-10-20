import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from customized_dataset import MyDataset, KFold
from models.dann import DANN
from tqdm.auto import tqdm

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'


loss_regression = torch.nn.MSELoss()
loss_domain = torch.nn.NLLLoss()


def train(s_train_loader, t_train_loader, model, optimizer, batch_size, epoch, epochs):
    s_t_loader = tqdm(enumerate(s_train_loader), total=len(s_train_loader))
    t_t_loader = tqdm(enumerate(t_train_loader), total=len(t_train_loader))
    len_dataloader = min(len(s_train_loader), len(t_t_loader))

    for (i, (xs, ys)), (_, (xt, yt)) in zip(s_t_loader, t_t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        domain_y = torch.zeros(batch_size)
        if torch.cuda.is_available():
            xs, ys = xs.cuda(), ys.cuda()
            domain_y = domain_y.cuda()
        optimizer.zero_grad()
        regression_pred, domain_pred = model(xs, alpha)
        err_s_label = loss_regression(regression_pred, ys)
        err_s_domain = loss_domain(domain_pred, domain_y)



def main(argv):
    gdsc_ic50_dataset = \
        MyDataset.from_ccl_dd_ic50(torch.load(GDSC_TENSOR_PATH + 'CCL.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'DD.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'IC50.pt'))
    ccle_domain_dataset = \
        MyDataset.from_ccl_dd_domain(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
                                     torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
                                     1)
    ccle_ic50_dataset_test = \
        MyDataset.from_ccl_dd_ic50(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'))

    k_fold = 5
    batch_size = 20
    lr = 2e-3
    epochs = 100

    gdsc_ic50_fold = KFold(gdsc_ic50_dataset, k_fold, 1)
    ccle_domain_fold = KFold(ccle_domain_dataset, k_fold, 1)
    model = DANN(gdsc_ic50_dataset.get_n_feature(), 0.5, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for k in range(k_fold):
        gdsc_tr_loader, gdsc_v_loader = \
            DataLoader(gdsc_ic50_fold.get_next_train_validation(), batch_size=batch_size, shuffle=False)
        ccle_tr_loader, ccle_v_loader = \
            DataLoader(ccle_domain_fold.get_next_train_validation(), batch_size=batch_size, shuffle=False)
        for epoch in range(epochs):
            train(gdsc_tr_loader, ccle_tr_loader, model, optimizer, batch_size, epoch, epochs)



if __name__ == "__main__":
    main(sys.argv[1:])
