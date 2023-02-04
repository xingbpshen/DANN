import argparse
import torch
from torch.utils.data import DataLoader
from customized_dataset import MyDatasetSep as MyDataset
from customized_dataset import KFoldSep as KFold
from logger import Logger
from models.mlp import MLP
from models.mte import MMTE
from tqdm.auto import tqdm
from datetime import date
import os
import wandb

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'
WEIGHTS_PATH = './data/weights/'
LOGGER_PATH = './results/'

loss_regression = torch.nn.MSELoss()
loss_huber = torch.nn.HuberLoss()

use_wandb = False
use_local_logger = True

def train(train_loader, model, optimizer, epoch, logger, is_parallel):
    t_loader = tqdm(enumerate(train_loader), total=len(train_loader))
    len_dataloader = len(t_loader)
    mse_fin = 0

    for i, (x1, x2, y) in t_loader:

        # for s in optimizer.param_groups:
        #     s['lr'] = pow(12, -0.5) * min(pow(i + 1, -0.5), (i + 1) * pow(int(len_dataloader * 0.04), -1.5))

        if torch.cuda.is_available():
            x1, x2, y = x1.to(0), x2.to(0), y.to(0)

        model.zero_grad()

        y_pred = model(x1, x2)

        loss = loss_regression(y_pred, y.reshape(-1, 1))

        if is_parallel > 1:
            loss.mean().backward()
            mse_fin += loss.mean().item()
        else:
            loss.backward()
            mse_fin += loss.item()

        optimizer.step()

    mse_fin = mse_fin / len_dataloader
    print('EPOCH {} TRAINING SET RESULTS: Average regression loss: {:.4f}'.format(epoch, mse_fin))

    if use_local_logger and logger is not None:
        logger.log({'epoch': epoch,
                    'train_mse': mse_fin})

    if use_wandb:
        wandb.log({"train_mse": mse_fin})

    del x1, x2, y

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def test(test_loader, model, epoch, logger, is_parallel):
    test_loader = tqdm(enumerate(test_loader), total=len(test_loader))
    len_dataloader = len(test_loader)
    mse_fin = 0

    for i, (x1, x2, y) in test_loader:

        if torch.cuda.is_available():
            x1, x2, y = x1.to(0), x2.to(0), y.to(0)

        y_pred = model(x1, x2)

        loss = loss_regression(y_pred, y.reshape(-1, 1))

        if is_parallel > 1:
            mse_fin += loss.mean().item()
        else:
            mse_fin += loss.item()

    mse_fin = mse_fin / len_dataloader

    if use_local_logger and logger is not None:
        logger.log({'epoch': epoch,
                    'test_mse': mse_fin})

    if use_wandb:
        wandb.log({"test_mse": mse_fin})

    print('EPOCH {} TESTING RESULTS: Average mse: {:.4f}'.format(epoch, mse_fin))

    del x1, x2, y

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv):
    info = 'mmte_nobatchnorm_respnorm_gdsc'
    k_fold = 5
    batch_size = 48
    lr = 1e-3
    epochs = int(argv.epochs)
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

    ccl = torch.load(GDSC_TENSOR_PATH + 'CCL.pt')   # np
    dd = torch.load(GDSC_TENSOR_PATH + 'DD.pt')
    resp = torch.load(GDSC_TENSOR_PATH + 'IC50.pt')

    def remove_extreme_features(t):
        idx = torch.argmax(t) + 1
        t[:, (idx % t.shape[1]) - 1] = 0

        return t

    dd = remove_extreme_features(dd)
    # _, _, V = torch.pca_lowrank(ccl)
    # ccl = torch.matmul(ccl, V[:, :12000])
    resp = (resp - resp.mean()) / resp.std()

    gdsc_ic50_dataset = \
        MyDataset.from_ccl_dd_ic50(ccl,
                                   dd,
                                   resp)
    # print('Data loading CKPT 1.')
    #
    # ccle_domain_dataset = \
    #     MyDataset.from_ccl_dd_domain(torch.load(CCLE_TENSOR_PATH + 'CCL.pt'),
    #                                  torch.load(CCLE_TENSOR_PATH + 'DD.pt'),
    #                                  1)
    # print('Data loading CKPT 2.')
    #
    # ccle_ic50_dataset_test = \
    #     MyDataset.from_ccl_dd_ic50(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
    #                                torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
    #                                torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'), frac=0.2)

    print('Dataset load complete.')

    gdsc_ic50_fold = KFold(gdsc_ic50_dataset, k_fold, use_portion_frac=1)
    # ccle_domain_fold = KFold(ccle_domain_dataset, k_fold, 1)

    print('K-fold split complete.')

    for k in range(k_fold):

        train_logger, test_logger = None, None

        if use_local_logger:
            train_logger = Logger(['epoch',
                                   'train_mse'])
            test_logger = Logger(['epoch',
                                  'test_mse'])

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
            DataLoader(tmp1, batch_size=batch_size, shuffle=False, drop_last=True)

        # tmp0, tmp1 = ccle_domain_fold.get_next_train_validation()
        # ccle_tr_loader, ccle_v_loader = \
        #     DataLoader(tmp0, batch_size=batch_size, shuffle=False, drop_last=True), \
        #     DataLoader(tmp1, batch_size=1, shuffle=False, drop_last=True)
        #
        # ccle_ic50_test_loader = DataLoader(ccle_ic50_dataset_test, batch_size=1, shuffle=False)

        model = MMTE(f1=gdsc_ic50_dataset.get_n_x1_feature(), f2=gdsc_ic50_dataset.get_n_x2_feature(), d_model=12, n=1)
        # model = MLP(m1=gdsc_ic50_dataset.get_n_x1_feature(), m2=gdsc_ic50_dataset.get_n_x2_feature(), n=1)

        s_epoch = 1

        ckpt = None
        if argv.from_ckpt != 'None':
            ckpt = torch.load(argv.from_ckpt)
            s_epoch = int(ckpt['epoch']) + 1
            model.load_state_dict(ckpt['model_state_dict'])

            print('Continue training from epoch {}'.format(s_epoch))

        # use_model = model
        if is_parallel > 1:
            model = torch.nn.DataParallel(model, device_ids=[*range(is_parallel)])
        if torch.cuda.is_available():
            print('CUDA MEM ALLOCATED before loading the model: ', torch.cuda.memory_allocated())
            print('CUDA MEM RESERVED before loading the model: ', torch.cuda.memory_reserved())
            model = model.to(0)
            print('CUDA MEM ALLOCATED after loading the model: ', torch.cuda.memory_allocated())
            print('CUDA MEM RESERVED after loading the model: ', torch.cuda.memory_reserved())

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if argv.from_ckpt != 'None':
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print('Start training on fold {}.'.format(k))

        for epoch in range(s_epoch, epochs + 1):
            train(gdsc_tr_loader, model, optimizer, epoch, train_logger, is_parallel)

            if not os.path.exists(dir_weights):
                os.makedirs(dir_weights)

            if epoch % 10 == 0:
                if is_parallel > 1:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}_P.pt'.format(k + 1, batch_size, lr, epoch))
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, dir_weights + 'TRAIN_DANN_FD{}_BS{}_LR{}_EP{}.pt'.format(k + 1, batch_size, lr, epoch))

            test(gdsc_v_loader, model, epoch, test_logger, is_parallel)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default='100', help='total number of epochs')
    parser.add_argument('--from_ckpt', default='None', help='path of the check point weights')

    args = parser.parse_args()

    main(args)
