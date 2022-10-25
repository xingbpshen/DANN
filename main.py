import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from customized_dataset import MyDataset, KFold
from logger import Logger
from models.dann import DANN
from tqdm.auto import tqdm
import wandb

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'
WEIGHTS_PATH = './data/weights/'
LOGGER_PATH = './results/plots/'

loss_regression = torch.nn.MSELoss()
loss_domain = torch.nn.NLLLoss()

use_wandb = False
use_local_logger = True


def train(s_train_loader, t_train_loader, model, optimizer, batch_size, epoch, epochs,
          s_x_mm_tuple, s_y_mm_tuple, t_x_mm_tuple, logger):
    s_t_loader = tqdm(enumerate(s_train_loader), total=len(s_train_loader))
    t_t_loader = tqdm(enumerate(t_train_loader), total=len(t_train_loader))
    len_dataloader = min(len(s_t_loader), len(t_t_loader))
    loss_total_fin, loss_t_domain_fin, mse_fin = 0, 0, 0

    for (i, (xs, ys)), (_, (xt, dyt)) in zip(s_t_loader, t_t_loader):
        s_len = xs.shape[0]
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # TODO
        dys = torch.zeros((s_len, 2))
        dys[:, 0] = 1
        if torch.cuda.is_available():
            dys, dyt = dys.to(torch.int64), dyt.to(torch.int64)
            xs, ys, dys, xt, dyt = xs.cuda(), ys.cuda(), dys.cuda(), xt.cuda(), dyt.cuda()
        optimizer.zero_grad()

        # xs = (xs - s_x_mm_tuple[0]) / (s_x_mm_tuple[1] - s_x_mm_tuple[0]) * (1 - 0) + 0
        xs = xs - xs.min()
        xs = xs / xs.max()
        ys = (ys - s_y_mm_tuple[0]) / (s_y_mm_tuple[1] - s_y_mm_tuple[0]) * (1 - 0) + 0
        regression_pred, domain_pred = model(xs, alpha)
        loss_s_label = loss_regression(regression_pred, ys.view(-1, 1))
        loss_s_domain = loss_domain(domain_pred, dys[:, 1])

        # xt = (xt - t_x_mm_tuple[0]) / (t_x_mm_tuple[1] - t_x_mm_tuple[0]) * (1 - 0) + 0
        xt = xt - xt.min()
        xt = xt / xt.max()
        _, domain_pred = model(xt, alpha)
        loss_t_domain = loss_domain(domain_pred, dyt[:, 1])

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

    # do not log the first epoch due to plotting appearance
    if use_local_logger and epoch != 1 and logger is not None:
        logger.log({'epoch': epoch,
                    'train_total_loss': loss_total_fin,
                    'train_source_mse': mse_fin,
                    'train_target_domain_loss': loss_t_domain_fin})

    if use_wandb:
        wandb.log({"train_total_loss": loss_total_fin,
                   "train_source_mse": mse_fin,
                   "train_target_domain_loss": loss_t_domain_fin})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test(s_test_loader, t_test_loader, model, epoch, epochs, s_y_mm_tuple, t_y_mm_tuple, logger):
    s_loader = tqdm(enumerate(s_test_loader), total=len(s_test_loader))
    t_loader = tqdm(enumerate(t_test_loader), total=len(t_test_loader))
    len_dataloader = min(len(s_loader), len(t_loader))
    mse_s_fin, mse_t_fin = 0, 0

    for (i, (xs, ys)), (_, (xt, yt)) in zip(s_loader, t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        if torch.cuda.is_available():
            xs, ys, xt, yt = xs.cuda(), ys.cuda(), xt.cuda(), yt.cuda()

        xs = xs - xs.min()
        xs = xs / xs.max()
        ys = (ys - s_y_mm_tuple[0]) / (s_y_mm_tuple[1] - s_y_mm_tuple[0]) * (1 - 0) + 0

        xt = xt - xt.min()
        xt = xt / xt.max()
        yt = (yt - t_y_mm_tuple[0]) / (t_y_mm_tuple[1] - t_y_mm_tuple[0]) * (1 - 0) + 0

        regression_pred, _ = model(xs, alpha)
        mse_s = loss_regression(regression_pred, ys.view(-1, 1))

        regression_pred2, _ = model(xt, alpha)
        mse_t = loss_regression(regression_pred2, yt.view(-1, 1))

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
                                   torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'))

    k_fold = 5
    batch_size = 20
    lr = 1e-3
    epochs = 10

    gdsc_ic50_fold = KFold(gdsc_ic50_dataset, k_fold, 1)
    ccle_domain_fold = KFold(ccle_domain_dataset, k_fold, 1)

    model = DANN(gdsc_ic50_dataset.get_n_feature(), 0.5, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for k in range(k_fold):

        train_logger, test_logger = None, None

        if use_local_logger:
            train_logger = Logger(['epoch',
                                   'train_total_loss',
                                   'train_source_mse',
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
        s_x_mm_tuple = (tmp0.x.min(), tmp0.x.max())
        tr_s_y_mm_tuple = (tmp0.y.min(), tmp0.y.max())
        v_s_y_mm_tuple = (tmp1.y.min(), tmp1.y.max())
        gdsc_tr_loader, gdsc_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=1, shuffle=False)

        tmp0, tmp1 = ccle_domain_fold.get_next_train_validation()
        t_x_mm_tuple = (tmp0.x.min(), tmp0.x.max())
        ccle_tr_loader, ccle_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False), \
            DataLoader(tmp1, batch_size=1, shuffle=False)

        ccle_ic50_test_loader = DataLoader(ccle_ic50_dataset_test, batch_size=1, shuffle=False)
        te_t_y_mm_tuple = (ccle_ic50_dataset_test.y.min(), ccle_ic50_dataset_test.y.max())

        for epoch in range(1, epochs + 1):
            train(gdsc_tr_loader, ccle_tr_loader, model, optimizer, batch_size, epoch, epochs,
                  None, tr_s_y_mm_tuple, None, train_logger)
            torch.save(model.state_dict(),
                       WEIGHTS_PATH + 'DANN_TRAIN_FD{}_BS{}_LR{}_EP{}.pt'.format(k + 1, batch_size, lr, epoch))
            test(gdsc_v_loader, ccle_ic50_test_loader, model, epoch, epochs, v_s_y_mm_tuple, te_t_y_mm_tuple,
                 test_logger)

        train_logger.save_plot(LOGGER_PATH + 'PLOT_TRAIN_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))
        test_logger.save_plot(LOGGER_PATH + 'PLOT_TEST_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))


if __name__ == "__main__":
    main(sys.argv[1:])
