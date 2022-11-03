import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from customized_dataset import MyDatasetSep as MyDataset
from customized_dataset import KFoldSep as KFold
from logger import Logger
from models.dann import DANN
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


@torch.no_grad()
def compute_batch_xs_xt(x1s, x2s, x1t, x2t):
    s_len = x1s.shape[0]
    t_len = x1t.shape[0]
    xs, xt = torch.zeros((s_len, x1s.shape[1] + x2s.shape[1])), torch.zeros((t_len, x1s.shape[1] + x2s.shape[1]))
    for j in range(s_len):
        xs[j] = torch.cat((x1s[j], x2s[j]))
    for j in range(t_len):
        xt[j] = torch.cat((x1t[j], x2t[j]))

    return xs, xt


@torch.no_grad()
def normalization(t, prev_min, prev_max, new_min=0, new_max=1):
    return (t - prev_min) / (prev_max - prev_min) * (new_max - new_min) + new_min


@torch.no_grad()
def standardization(t, mean, std):
    return (t - mean) / std


def train(s_train_loader, t_train_loader, model, optimizer, batch_size, epoch, epochs,
          tr_s_x1_mm_tuple, tr_s_x2_mm_tuple, tr_t_x1_mm_tuple, tr_t_x2_mm_tuple, logger, is_parallel):
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
        # xs, xt = compute_batch_xs_xt(normalization(x1s, tr_s_x1_mm_tuple[0], tr_s_x1_mm_tuple[1]),
        #                              normalization(x2s, tr_s_x2_mm_tuple[0], tr_s_x2_mm_tuple[1]),
        #                              normalization(x1t, tr_t_x1_mm_tuple[0], tr_t_x1_mm_tuple[1]),
        #                              normalization(x2t, tr_t_x2_mm_tuple[0], tr_t_x2_mm_tuple[1]))
        xs, xt = compute_batch_xs_xt(standardization(x1s, tr_s_x1_mm_tuple[0], tr_s_x1_mm_tuple[1]),
                                     normalization(x2s, tr_s_x2_mm_tuple[0], tr_s_x2_mm_tuple[1]),
                                     standardization(x1t, tr_t_x1_mm_tuple[0], tr_t_x1_mm_tuple[1]),
                                     normalization(x2t, tr_t_x2_mm_tuple[0], tr_t_x2_mm_tuple[1]))

        dys, dyt = dys.to(torch.int64), dyt.to(torch.int64)

        if torch.cuda.is_available():
            dys, dyt = dys.to(torch.int64), dyt.to(torch.int64)
            xs, ys, dys, xt, dyt = xs.to(0), ys.to(0), dys.to(0), xt.to(0), dyt.to(0)

        model.zero_grad()

        # print('xs', xs.shape)
        # print('xt', xt.shape)

        # print('CUDA MEM ALLOCATED CKPT 1: ', torch.cuda.memory_allocated())
        # print('CUDA MEM RESERVED CKPT 1: ', torch.cuda.memory_reserved())
        # if torch.cuda.is_available():
        #     xs = xs.to(0)
        # print('CUDA MEM ALLOCATED CKPT 2: ', torch.cuda.memory_allocated())
        # print('CUDA MEM RESERVED CKPT 2: ', torch.cuda.memory_reserved())
        regression_pred, domain_pred = model(xs, alpha)
        # print('CUDA MEM ALLOCATED CKPT 3: ', torch.cuda.memory_allocated())
        # print('CUDA MEM RESERVED CKPT 3: ', torch.cuda.memory_reserved())
        # if torch.cuda.is_available():
        #     del xs
        #     torch.cuda.empty_cache()
        # print('CUDA MEM ALLOCATED CKPT 4: ', torch.cuda.memory_allocated())
        # print('CUDA MEM RESERVED CKPT 4: ', torch.cuda.memory_reserved())

        # print('rp', regression_pred.shape)
        # print('dp', domain_pred.shape)
        # print('ys', ys.view(-1, 1).shape)
        # if torch.cuda.is_available():
        #     ys = ys.to(0)
        loss_s_label = loss_regression(regression_pred, ys.view(-1, 1))
        # if torch.cuda.is_available():
        #     del regression_pred
        #     del ys
        #     torch.cuda.empty_cache()

        # if torch.cuda.is_available():
        #     dys = dys.to(0)
        loss_s_domain = loss_domain(domain_pred, dys[:, 1])
        # if torch.cuda.is_available():
        #     del domain_pred
        #     del dys
        #     torch.cuda.empty_cache()
        #
        # print('CUDA MEM ALLOCATED CKPT 3: ', torch.cuda.memory_allocated())
        # print('CUDA MEM RESERVED CKPT 3: ', torch.cuda.memory_reserved())

        # if torch.cuda.is_available():
        #     xt = xt.to(0)
        _, domain_pred = model(xt, alpha)
        # if torch.cuda.is_available():
        #     del xt
        #     torch.cuda.empty_cache()
        #
        # if torch.cuda.is_available():
        #     dyt = dyt.to(0)
        loss_t_domain = loss_domain(domain_pred, dyt[:, 1])
        # if torch.cuda.is_available():
        #     del domain_pred
        #     del dyt
        #     torch.cuda.empty_cache()

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
def test(s_test_loader, t_test_loader, model, epoch, epochs, tr_s_x1_mm_tuple, tr_s_x2_mm_tuple,
         tr_t_x1_mm_tuple, tr_t_x2_mm_tuple, logger, is_parallel):
    s_loader = tqdm(enumerate(s_test_loader), total=len(s_test_loader))
    t_loader = tqdm(enumerate(t_test_loader), total=len(t_test_loader))
    len_dataloader = min(len(s_loader), len(t_loader))
    mse_s_fin, mse_t_fin = 0, 0

    for (i, (x1s, x2s, ys)), (_, (x1t, x2t, yt)) in zip(s_loader, t_loader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # xs, xt = compute_batch_xs_xt(normalization(x1s, tr_s_x1_mm_tuple[0], tr_s_x1_mm_tuple[1]),
        #                              normalization(x2s, tr_s_x2_mm_tuple[0], tr_s_x2_mm_tuple[1]),
        #                              normalization(x1t, tr_t_x1_mm_tuple[0], tr_t_x1_mm_tuple[1]),
        #                              normalization(x2t, tr_t_x2_mm_tuple[0], tr_t_x2_mm_tuple[1]))
        xs, xt = compute_batch_xs_xt(standardization(x1s, tr_s_x1_mm_tuple[0], tr_s_x1_mm_tuple[1]),
                                     normalization(x2s, tr_s_x2_mm_tuple[0], tr_s_x2_mm_tuple[1]),
                                     standardization(x1t, tr_t_x1_mm_tuple[0], tr_t_x1_mm_tuple[1]),
                                     normalization(x2t, tr_t_x2_mm_tuple[0], tr_t_x2_mm_tuple[1]))

        if torch.cuda.is_available():
            xs, ys, xt, yt = xs.to(0), ys.to(0), xt.to(0), yt.to(0)

        regression_pred, _ = model(xs, alpha)
        # if torch.cuda.is_available():
        #     del xs
        #     torch.cuda.empty_cache()

        mse_s = loss_regression(regression_pred, ys.view(-1, 1))
        # if torch.cuda.is_available():
        #     del regression_pred
        #     del ys
        #     torch.cuda.empty_cache()

        regression_pred2, _ = model(xt, alpha)
        # if torch.cuda.is_available():
        #     del xt
        #     torch.cuda.empty_cache()

        mse_t = loss_regression(regression_pred2, yt.view(-1, 1))
        # if torch.cuda.is_available():
        #     del regression_pred2
        #     del yt
        #     torch.cuda.empty_cache()

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
    info = 'CclDdNormAcrossSetCatSepYNoNormMTanh'
    k_fold = 5
    batch_size = 1000
    lr = 1e-3
    epochs = 100
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

    print('Number of GPU(s) used: {} \nRUN {} \nDATE {} \nINFORMATION {} \nLEARNING RATE {} \nBATCH SIZE {} \nEPOCHS {}'
          .format(is_parallel, run_curr, date.today(), info, lr, batch_size, epochs))

    gdsc_ic50_dataset = \
        MyDataset.from_ccl_dd_ic50(torch.load(GDSC_TENSOR_PATH + 'CCL.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'DD.pt'),
                                   torch.load(GDSC_TENSOR_PATH + 'IC50.pt'))
    print('Data loading CKPT 1.')

    ccle_domain_dataset = \
        MyDataset.from_ccl_dd_domain(torch.load(CCLE_TENSOR_PATH + 'CCL.pt'),
                                     torch.load(CCLE_TENSOR_PATH + 'DD.pt'),
                                     1)
    print('Data loading CKPT 2.')

    ccle_ic50_dataset_test = \
        MyDataset.from_ccl_dd_ic50(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
                                   torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'), frac=0.2)

    print('Dataset load complete.')

    # tr_s_x1_mm_tuple = (gdsc_ic50_dataset.get_x1_min_max())
    tr_s_x2_mm_tuple = (gdsc_ic50_dataset.get_x2_min_max())
    # tr_t_x1_mm_tuple = (ccle_domain_dataset.get_x1_min_max())
    tr_t_x2_mm_tuple = (ccle_domain_dataset.get_x2_min_max())
    tr_s_x1_mm_tuple = (gdsc_ic50_dataset.get_x1_mean_std())
    # tr_s_x2_mm_tuple = (gdsc_ic50_dataset.get_x2_mean_std())
    tr_t_x1_mm_tuple = (ccle_domain_dataset.get_x1_mean_std())
    # tr_t_x2_mm_tuple = (ccle_domain_dataset.get_x2_mean_std())

    print('tr_s_x1_mm_tuple', tr_s_x1_mm_tuple)
    print('tr_s_x2_mm_tuple', tr_s_x2_mm_tuple)
    print('tr_t_x1_mm_tuple', tr_t_x1_mm_tuple)
    print('tr_t_x2_mm_tuple', tr_t_x2_mm_tuple)

    print('Data distribution parameters computed.')

    gdsc_ic50_fold = KFold(gdsc_ic50_dataset, k_fold, 1)
    ccle_domain_fold = KFold(ccle_domain_dataset, k_fold, 1)

    print('K-fold split complete.')

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
            DataLoader(tmp0, batch_size=batch_size, shuffle=False, drop_last=True), \
            DataLoader(tmp1, batch_size=1, shuffle=False, drop_last=True)

        tmp0, tmp1 = ccle_domain_fold.get_next_train_validation()
        ccle_tr_loader, ccle_v_loader = \
            DataLoader(tmp0, batch_size=batch_size, shuffle=False, drop_last=True), \
            DataLoader(tmp1, batch_size=1, shuffle=False, drop_last=True)

        ccle_ic50_test_loader = DataLoader(ccle_ic50_dataset_test, batch_size=1, shuffle=False)

        model = DANN(gdsc_ic50_dataset.get_n_feature(), 0.8, 1)
        use_model = model
        if is_parallel > 1:
            use_model = torch.nn.DataParallel(model, device_ids=[*range(is_parallel)])
        if torch.cuda.is_available():
            print('CUDA MEM ALLOCATED before loading the model: ', torch.cuda.memory_allocated())
            print('CUDA MEM RESERVED before loading the model: ', torch.cuda.memory_reserved())
            use_model = use_model.to(0)
            print('CUDA MEM ALLOCATED after loading the model: ', torch.cuda.memory_allocated())
            print('CUDA MEM RESERVED after loading the model: ', torch.cuda.memory_reserved())

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        print('Start training on fold {}.'.format(k))

        for epoch in range(1, epochs + 1):
            train(gdsc_tr_loader, ccle_tr_loader, use_model, optimizer, batch_size, epoch, epochs,
                  tr_s_x1_mm_tuple, tr_s_x2_mm_tuple, tr_t_x1_mm_tuple, tr_t_x2_mm_tuple, train_logger, is_parallel)

            if not os.path.exists(dir_weights):
                os.makedirs(dir_weights)

            if epoch % 10 == 0:
                if is_parallel > 1:
                    torch.save(use_model.module.state_dict(),
                               dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}_P.pt'.format(k + 1, batch_size, lr, epoch))
                else:
                    torch.save(use_model.state_dict(),
                               dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}.pt'.format(k + 1, batch_size, lr, epoch))

            # Use the scalar in training to do normalization
            test(gdsc_v_loader, ccle_ic50_test_loader, use_model, epoch, epochs, tr_s_x1_mm_tuple, tr_s_x2_mm_tuple,
                 tr_t_x1_mm_tuple, tr_t_x2_mm_tuple, test_logger, is_parallel)

        if use_local_logger:

            if not os.path.exists(dir_plots):
                os.makedirs(dir_plots)

            if not os.path.exists(dir_values):
                os.makedirs(dir_values)

            train_logger.save_csv(
                dir_values + 'TRAIN_METRICS_FD{}_BS{}_LR{}_EP{}.csv'.format(k + 1, batch_size, lr, epochs))
            train_logger.save_plot(
                dir_plots + 'TRAIN_PLOT_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))
            test_logger.save_csv(
                dir_values + 'TEST_METRICS_FD{}_BS{}_LR{}_EP{}.csv'.format(k + 1, batch_size, lr, epochs))
            test_logger.save_plot(dir_plots + 'TEST_PLOT_FD{}_BS{}_LR{}_EP{}.jpg'.format(k + 1, batch_size, lr, epochs))


if __name__ == "__main__":
    main(sys.argv[1:])
