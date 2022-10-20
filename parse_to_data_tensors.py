import getopt

import torch
import pandas as pd
import sys
from tqdm import tqdm

GDSC_TUPLE_EXCLUDED_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds_excluded.csv'
GDSC_TUPLE_REMAINED_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds_remained.csv'
GDSC_TUPLE_INCLUDED_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds.csv'
GDSC_TUPLE_PATH = GDSC_TUPLE_EXCLUDED_PATH
GDSC_DD_PATH = './data/cleaned/gdsc/gdsc_250_drug_descriptors.csv'
GDSC_CCL_PATH = './data/cleaned/gdsc/sanger_fpkm_inter_gene.csv'

CCLE_TUPLE_PATH = './data/cleaned/ccle/ctrp_tuple_label_folds_cleaned.csv'
CCLE_TUPLE_COMMON_PATH = './data/cleaned/ccle/ctrp_tuple_label_folds_cleaned_common.csv'
CCLE_DD_PATH = './data/cleaned/ccle/ctrp_gdsc_desc_inter.csv'
CCLE_CCL_PATH = './data/cleaned/ccle/ccle_TPM_inter_gene.csv'

TENSOR_PATH = './data/tensors/'


def load_as_df(path):
    return pd.read_csv(path, index_col=0)


def load_all(args):
    loads = []
    for arg in args:
        loads.append(load_as_df(arg))
    return loads


def extract_data(dfs, label):
    tuple_df, dd_df, ccl_df = dfs[0], dfs[1], dfs[2]
    dd_tensor, ccl_tensor, response_tensor = torch.zeros((tuple_df.shape[0], dd_df.shape[1])), torch.zeros((tuple_df.shape[0], ccl_df.shape[0])), torch.zeros((tuple_df.shape[0],))
    for i in tqdm(range(tuple_df.shape[0])):
        drug_name, ccl_name, response = tuple_df['drug'].iloc[[i]], tuple_df['cell_line'].iloc[[i]], tuple_df[label].iloc[[i]]
        dd = dd_df.loc[drug_name]   # n
        dd = torch.tensor(dd.values).reshape(-1, )
        ccl = ccl_df[ccl_name]  # nx1
        ccl = torch.tensor(ccl.values).reshape(-1, )
        dd_tensor[i] = dd
        ccl_tensor[i] = ccl
        response_tensor[i] = torch.tensor(response.values).reshape(-1, )

    return dd_tensor, ccl_tensor, response_tensor


def load_from_name(d_name):
    dfs, dd_tensor, ccl_tensor, response_tensor = None, None, None, None
    if d_name.lower() == 'gdsc':
        dfs = load_all([GDSC_TUPLE_PATH, GDSC_DD_PATH, GDSC_CCL_PATH])
        dd_tensor, ccl_tensor, response_tensor = extract_data(dfs, 'ln_ic50')
    elif d_name.lower() == 'ccle':
        dfs = load_all([CCLE_TUPLE_PATH, CCLE_DD_PATH, CCLE_CCL_PATH])
        dd_tensor, ccl_tensor, response_tensor = extract_data(dfs, 'auc')
    elif d_name.lower() == 'common':
        dfs = load_all([CCLE_TUPLE_COMMON_PATH, CCLE_DD_PATH, CCLE_CCL_PATH])
        dd_tensor, ccl_tensor, response_tensor = extract_data(dfs, 'ln_ic50')
    else:
        print('ERROR IN load_from_name(d_name).')
        exit(1)
    return dd_tensor, ccl_tensor, response_tensor


def save_tensors(ts, names):
    for t, name in zip(ts, names):
        print(t)
        torch.save(t, TENSOR_PATH + name)
        print('Saved ', TENSOR_PATH + name, ' shape ', t.shape)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'ho:n:', ['option=', 'name='])
    except getopt.GetoptError:
        print('parse_to_data_tensor.py --option <exclude/include> --name <gdsc/ccle/all>')
    for opt, arg in opts:
        if opt == '-h':
            print('parse_to_data_tensor.py --option <exclude/include> --name <gdsc/ccle/all/common>')
            exit()
        elif opt in ("-o", "--option"):
            if arg.lower() == 'include':
                global GDSC_TUPLE_PATH
                GDSC_TUPLE_PATH = GDSC_TUPLE_INCLUDED_PATH
                print('Your gdsc tensor will have common dd-ccl with ccle tensor.')
            else:
                print('Your gdsc tensor will not have common dd-ccl with ccle tensor (i.e remove some data from gdsc).')
        elif opt in ("-n", "--name"):
            if arg.lower() == 'gdsc':
                print('Loading only gdsc to tensors.')
                dd_tensor, ccl_tensor, response_tensor = load_from_name('gdsc')
                save_tensors([dd_tensor, ccl_tensor, response_tensor], ['gdsc/DD.pt', 'gdsc/CCL.pt', 'gdsc/IC50.pt'])
                exit()
            elif arg.lower() == 'ccle':
                print('Loading only ccle to tensors.')
                dd_tensor, ccl_tensor, response_tensor = load_from_name('ccle')
                save_tensors([dd_tensor, ccl_tensor, response_tensor], ['ccle/DD.pt', 'ccle/CCL.pt', 'ccle/AUC.pt'])
                dd_tensor, ccl_tensor, response_tensor = load_from_name('common')
                save_tensors([dd_tensor, ccl_tensor, response_tensor],
                             ['ccle/DD_COMMON.pt', 'ccle/CCL_COMMON.pt', 'ccle/IC50_COMMON.pt'])
                exit()
            elif arg.lower() == 'common':
                print('Loading only common part to tensors.')
                dd_tensor, ccl_tensor, response_tensor = load_from_name('common')
                save_tensors([dd_tensor, ccl_tensor, response_tensor],
                             ['ccle/DD_COMMON.pt', 'ccle/CCL_COMMON.pt', 'ccle/IC50_COMMON.pt'])
                exit()
            else:
                print('Loading gdsc and ccle to tensors.')
                dd_tensor, ccl_tensor, response_tensor = load_from_name('gdsc')
                save_tensors([dd_tensor, ccl_tensor, response_tensor], ['gdsc/DD.pt', 'gdsc/CCL.pt', 'gdsc/IC50.pt'])
                dd_tensor, ccl_tensor, response_tensor = load_from_name('ccle')
                save_tensors([dd_tensor, ccl_tensor, response_tensor], ['ccle/DD.pt', 'ccle/CCL.pt', 'ccle/AUC.pt'])
                dd_tensor, ccl_tensor, response_tensor = load_from_name('common')
                save_tensors([dd_tensor, ccl_tensor, response_tensor],
                             ['ccle/DD_COMMON.pt', 'ccle/CCL_COMMON.pt', 'ccle/IC50_COMMON.pt'])
                exit()


if __name__ == "__main__":
    main(sys.argv[1:])
