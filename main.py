import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from customized_dataset import MyDataset, KFold
from models.dann import DANN
from tqdm.auto import tqdm

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'
WEIGHTS_PATH = './data/weights/'

loss_regression = torch.nn.MSELoss()
loss_domain = torch.nn.NLLLoss()


def train(s_train_loader, t_train_loader, model, optimizer, batch_size, epoch, epochs):
    s_t_loader = tqdm(enumerate(s_train_loader), total=len(s_train_loader))
    t_t_loader = tqdm(enumerate(t_train_loader), total=len(t_train_loader))
    len_dataloader = min(len(s_train_loader), len(t_t_loader))
    loss_total_fin, loss_t_domain_fin, mse_fin = 0, 0, 0

    for (i, (xs, ys)), (_, (xt, dyt)) in zip(s_t_loader, t_t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        dys = torch.zeros(batch_size)
        if torch.cuda.is_available():
            xs, ys, dys, xt, dyt = xs.cuda(), ys.cuda(), dys.cuda(), xt.cuda(), dyt.cuda()
        optimizer.zero_grad()

        regression_pred, domain_pred = model(xs, alpha)
        loss_s_label = loss_regression(regression_pred, ys)
        loss_s_domain = loss_domain(domain_pred, dys)

        _, domain_pred = model(xt, alpha)
        loss_t_domain = loss_domain(domain_pred, dyt)

        loss = loss_s_label + loss_s_domain + loss_t_domain
        loss.backward()
        optimizer.step()

        loss_total_fin += loss
        loss_t_domain_fin += loss_t_domain
        mse_fin += loss_s_label

    loss_total_fin = loss_total_fin / len_dataloader
    loss_t_domain_fin = loss_t_domain_fin / len_dataloader
    mse_fin = mse_fin / len_dataloader
    print('EPOCH {} TRAINING SET RESULTS: Average total loss: {:.4f} Average target domain loss: {:.4f}  '
          'Average source mse: {:.4f}'.format(epoch, loss_total_fin, loss_t_domain_fin, mse_fin))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        tmp0, tmp1 = gdsc_ic50_fold.get_next_train_validation()
        gdsc_tr_loader, gdsc_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=batch_size, shuffle=False)
        tmp0, tmp1 = ccle_domain_fold.get_next_train_validation()
        ccle_tr_loader, ccle_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=batch_size, shuffle=False)
        for epoch in range(1, epochs + 1):
            train(gdsc_tr_loader, ccle_tr_loader, model, optimizer, batch_size, epoch, epochs)
            torch.save(model.state_dict(),
                       WEIGHTS_PATH + 'DANN_TRAIN_FD{}_BS{}_LR{}_EP{}.pt'.format(k + 1, batch_size, lr, epoch))


if __name__ == "__main__":
    main(sys.argv[1:])
