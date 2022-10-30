import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from customized_dataset import MyDatasetSep as MyDataset
from customized_dataset import KFoldSep as KFold
from logger import Logger
from models.dann import DANNConv2d as DANN
from tqdm.auto import tqdm
from datetime import date
import os
import wandb

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'
WEIGHTS_PATH = './data/weights/'
LOGGER_PATH = './results/'

loss_regression = torch.nn.MSELoss()
loss_domain = torch.nn.NLLLoss()

use_wandb = False
use_local_logger = True


def train(s_train_loader, t_train_loader, model, optimizer, batch_size, epoch, epochs,
          s_x_mm_tuple, t_x_mm_tuple, logger, is_parallel):
    s_t_loader = tqdm(enumerate(s_train_loader), total=len(s_train_loader))
    t_t_loader = tqdm(enumerate(t_train_loader), total=len(t_train_loader))
    len_dataloader = min(len(s_t_loader), len(t_t_loader))
    loss_total_fin, loss_t_domain_fin, mse_fin = 0, 0, 0

    for (i, (x1s, x2s, ys)), (_, (x1t, x2t, dyt)) in zip(s_t_loader, t_t_loader):
        s_len = x1s.shape[0]
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # TODO
        dys = torch.zeros((s_len, 2))
        dys[:, 0] = 1
        xs, xt = torch.zeros((s_len, x2s.shape[1], x1s.shape[1])), torch.zeros((s_len, x2s.shape[1], x1s.shape[1]))
        for j in range(s_len):
            xs[j] = torch.matmul(x2s[j].view(-1, 1), x1s[j].view(1, -1))
            xt[j] = torch.matmul(x2t[j].view(-1, 1), x1t[j].view(1, -1))

        if torch.cuda.is_available():
            dys, dyt = dys.to(torch.int64), dyt.to(torch.int64)
            xs, ys, dys, xt, dyt = xs.to(0), ys.to(0), dys.to(0), xt.to(0), dyt.to(0)
        model.zero_grad()

        xs = (xs - s_x_mm_tuple[0]) / (s_x_mm_tuple[1] - s_x_mm_tuple[0]) * (1 - 0) + 0
        xt = (xt - t_x_mm_tuple[0]) / (t_x_mm_tuple[1] - t_x_mm_tuple[0]) * (1 - 0) + 0

        regression_pred, domain_pred = model(xs, alpha)
        loss_s_label = loss_regression(regression_pred, ys.view(-1, 1))
        loss_s_domain = loss_domain(domain_pred, dys[:, 1])

        _, domain_pred = model(xt, alpha)
        loss_t_domain = loss_domain(domain_pred, dyt[:, 1])

        loss = loss_s_label + loss_s_domain + loss_t_domain

        if is_parallel > 1:
            loss.mean().backward()
            loss_total_fin += loss.mean()
            loss_t_domain_fin += loss_t_domain.mean()
            mse_fin += loss_s_label.mean()
        else:
            loss.backward()
            loss_total_fin += loss
            loss_t_domain_fin += loss_t_domain
            mse_fin += loss_s_label

        optimizer.step()

    loss_total_fin = loss_total_fin / len_dataloader
    loss_t_domain_fin = loss_t_domain_fin / len_dataloader
    mse_fin = mse_fin / len_dataloader
    print('EPOCH {} TRAINING SET RESULTS: Average total loss: {:.4f} Average target domain loss: {:.4f}  '
          'Average source regression loss: {:.4f}'.format(epoch, loss_total_fin, loss_t_domain_fin, mse_fin))

    if use_local_logger and logger is not None:
        logger.log({'epoch': epoch,
                    'train_total_loss': loss_total_fin,
                    'train_source_reg_loss': mse_fin,
                    'train_target_domain_loss': loss_t_domain_fin})

    if use_wandb:
        wandb.log({"train_total_loss": loss_total_fin,
                   "train_source_reg_loss": mse_fin,
                   "train_target_domain_loss": loss_t_domain_fin})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test(s_test_loader, t_test_loader, model, epoch, epochs, s_x_mm_tuple, t_x_mm_tuple, logger, is_parallel):
    s_loader = tqdm(enumerate(s_test_loader), total=len(s_test_loader))
    t_loader = tqdm(enumerate(t_test_loader), total=len(t_test_loader))
    len_dataloader = min(len(s_loader), len(t_loader))
    mse_s_fin, mse_t_fin = 0, 0

    for (i, (xs, ys)), (_, (xt, yt)) in zip(s_loader, t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        if torch.cuda.is_available():
            xs, ys, xt, yt = xs.to(0), ys.to(0), xt.to(0), yt.to(0)

        xs = (xs - s_x_mm_tuple[0]) / (s_x_mm_tuple[1] - s_x_mm_tuple[0]) * (1 - 0) + 0
        xt = (xt - t_x_mm_tuple[0]) / (t_x_mm_tuple[1] - t_x_mm_tuple[0]) * (1 - 0) + 0

        regression_pred, _ = model(xs, alpha)
        # mse_s = loss_regression(regression_pred * (y_mm_tuple_tr[1] - y_mm_tuple_tr[0]) + y_mm_tuple_tr[0],
        #                         ys.view(-1, 1))
        mse_s = loss_regression(regression_pred, ys.view(-1, 1))

        regression_pred2, _ = model(xt, alpha)
        # mse_t = loss_regression(regression_pred2 * (y_mm_tuple_tr[1] - y_mm_tuple_tr[0]) + y_mm_tuple_tr[0],
        #                         yt.view(-1, 1))
        mse_t = loss_regression(regression_pred2, yt.view(-1, 1))

        if is_parallel > 1:
            mse_s_fin += mse_s.mean()
            mse_t_fin += mse_t.mean()
        else:
            mse_s_fin += mse_s
            mse_t_fin += mse_t

    mse_s_fin = mse_s_fin / len_dataloader
    mse_t_fin = mse_t_fin / len_dataloader

    if use_local_logger and logger is not None:
        logger.log({'epoch': epoch,
                    'test_source_mse': mse_s_fin,
                    'test_target_mse': mse_t_fin})

    if use_wandb:
        wandb.log({"test_source_mse": mse_s_fin,
                   "test_target_mse": mse_t_fin})

    print('EPOCH {} TESTING RESULTS: Average source mse: {:.4f} Average target mse: {:.4f}'
          .format(epoch, mse_s_fin, mse_t_fin))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv):
    info = 'ConvXNormAcrossSetSepYNoNorm'
    k_fold = 5
    batch_size = 10
    lr = 1e-3
    epochs = 20
    is_parallel = 0
    if torch.cuda.is_available():
        is_parallel = torch.cuda.device_count()

    # Variables hold folder names
    run_curr = 1
    for path in os.listdir(WEIGHTS_PATH):
        # If current path is a dir
        if os.path.isdir(os.path.join(WEIGHTS_PATH, path)):
            run_curr += 1
    dir_weights = '{}RUN{}_{}_{}/'.format(WEIGHTS_PATH, run_curr, date.today(), info)
    dir_plots = '{}RUN{}_{}_{}/plots/'.format(LOGGER_PATH, run_curr, date.today(), info)
    dir_values = '{}RUN{}_{}_{}/values/'.format(LOGGER_PATH, run_curr, date.today(), info)

    print('Number of GPU(s) used: {} \nStart training \nRUN {} \nDATE {} \nINFORMATION {} \nLEARNING RATE {} \nBATCH SIZE {} \nEPOCHS {}'
          .format(is_parallel, run_curr, date.today(), info, lr, batch_size, epochs))

    gdsc_ic50_dataset = \
        MyDataset.from_ccl_dd_ic50(torch.load(GDSC_TENSOR_PATH + 'CCL.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'DD.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'IC50.pt'))
    ccle_domain_dataset = \
        MyDataset.from_ccl_dd_domain(torch.load(CCLE_TENSOR_PATH + 'CCL.pt'),
                                     torch.load(CCLE_TENSOR_PATH + 'DD.pt'),
                                     1)
    ccle_ic50_dataset_test = \
        MyDataset.from_ccl_dd_ic50(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'), frac=0.2)

    tr_s_x_mm_tuple = (gdsc_ic50_dataset.get_min1tmin2_max1tmax2())
    tr_t_x_mm_tuple = (ccle_domain_dataset.get_min1tmin2_max1tmax2())

    gdsc_ic50_fold = KFold(gdsc_ic50_dataset, k_fold, 1)
    ccle_domain_fold = KFold(ccle_domain_dataset, k_fold, 1)

    for k in range(k_fold):

        train_logger, test_logger = None, None

        if use_local_logger:
            train_logger = Logger(['epoch',
                                   'train_total_loss',
                                   'train_source_reg_loss',
                                   'train_target_domain_loss'])
            test_logger = Logger(['epoch',
                                  'test_source_mse',
                                  'test_target_mse'])

        if use_wandb:
            wandb.init(project="dann_on_drug_response", entity="xingshen")
            wandb.config = {
                "fold": k,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs
            }

        tmp0, tmp1 = gdsc_ic50_fold.get_next_train_validation()
        gdsc_tr_loader, gdsc_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=1, shuffle=False)

        tmp0, tmp1 = ccle_domain_fold.get_next_train_validation()
        ccle_tr_loader, ccle_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=1, shuffle=False)

        ccle_ic50_test_loader = DataLoader(ccle_ic50_dataset_test, batch_size=1, shuffle=False)

        model = DANN(gdsc_ic50_dataset.get_n_feature(), 0.5, 1)
        use_model = model
        if is_parallel > 1:
            use_model = torch.nn.DataParallel(model, device_ids=[*range(is_parallel)])
        if torch.cuda.is_available():
            use_model = use_model.to(0)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train(gdsc_tr_loader, ccle_tr_loader, use_model, optimizer, batch_size, epoch, epochs,
                  tr_s_x_mm_tuple, tr_t_x_mm_tuple, train_logger, is_parallel)

            if not os.path.exists(dir_weights):
                os.makedirs(dir_weights)
            if is_parallel > 1:
                torch.save(use_model.module.state_dict(),
                           dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}_P.pt'.format(k + 1, batch_size, lr, epoch))
            else:
                torch.save(use_model.state_dict(),
                           dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}.pt'.format(k + 1, batch_size, lr, epoch))

            # Use the scalar in training to do normalization
            test(gdsc_v_loader, ccle_ic50_test_loader, use_model, epoch, epochs, tr_s_x_mm_tuple, tr_t_x_mm_tuple,
                 test_logger, is_parallel)

        if use_local_logger:

            if not os.path.exists(dir_plots):
                os.makedirs(dir_plots)

            if not os.path.exists(dir_values):
                os.makedirs(dir_values)

            train_logger.save_csv(dir_values + 'TRAIN_METRICS_FD{}_BS{}_LR{}_EP{}.csv'.format(k + 1, batch_size, lr, epochs))
            train_logger.save_plot(dir_plots + 'TRAIN_PLOT_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))
            test_logger.save_csv(dir_values + 'TEST_METRICS_FD{}_BS{}_LR{}_EP{}.csv'.format(k + 1, batch_size, lr, epochs))
            test_logger.save_plot(dir_plots + 'TEST_PLOT_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))


if __name__ == "__main__":
    main(sys.argv[1:])
