import pandas as pd
import sys
from tqdm import tqdm

GDSC_TUPLE_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds.csv'
GDSC_TUPLE_EXCLUDED_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds_excluded.csv'
GDSC_TUPLE_REMAINED_PATH = './data/cleaned/gdsc/gdsc_tuple_labels_folds_remained.csv'

CCLE_TUPLE_PATH = './data/cleaned/ccle/ctrp_tuple_label_folds_cleaned.csv'
CCLE_TUPLE_COMMON_PATH = './data/cleaned/ccle/ctrp_tuple_label_folds_cleaned_common.csv'


def remove_intersection(df1, df2):
    dict = {}
    count = 0
    df1_will_be_removed = df1.copy()
    target_indices = []
    for i in tqdm(range(df2.shape[0])):
        drug_name_2 = df2['drug'].iloc[[i]].values
        ccl_name_2 = df2['cell_line'].iloc[[i]].values
        str_2 = str(drug_name_2).lower() + str(ccl_name_2).lower()
        dict[str_2] = 1
    for i in tqdm(range(df1_will_be_removed.shape[0])):
        drug_name_1 = df1_will_be_removed['drug'].iloc[[i]].values
        ccl_name_1 = df1_will_be_removed['cell_line'].iloc[[i]].values
        str_1 = str(drug_name_1).lower() + str(ccl_name_1).lower()
        if dict.get(str_1) is not None:
            count = count + 1
            target_indices.append(i)
            del dict[str_1]     # gdsc tuple may have duplicated pairs
    return df1_will_be_removed.drop(target_indices), df1.loc[target_indices, :]


def label_common_ccle(g, c):
    dict = {}
    common = pd.DataFrame(columns=['drug', 'cell_line', 'auc', 'pair_fold', 'cl_fold', 'ln_ic50'])
    for i in tqdm(range(g.shape[0])):
        drug_name_g = g['drug'].iloc[[i]].values
        ccl_name_g = g['cell_line'].iloc[[i]].values
        string = str(drug_name_g).lower() + str(ccl_name_g).lower()
        dict[string] = g['ln_ic50'].iloc[[i]].values
    for i in tqdm(range(c.shape[0])):
        drug_name_c = c['drug'].iloc[[i]].values
        ccl_name_c = c['cell_line'].iloc[[i]].values
        string = str(drug_name_c).lower() + str(ccl_name_c).lower()
        if dict.get(string) is not None:
            ic50 = dict.get(string)
            row = c.iloc[[i]].copy()
            row['ln_ic50'] = ic50
            common = pd.concat([common, row])
            del dict[string]    # ccle tuple may have duplicated pairs
    return common


def main(argv):
    df1 = pd.read_csv(GDSC_TUPLE_PATH, index_col=0)
    df2 = pd.read_csv(CCLE_TUPLE_PATH, index_col=0)
    df1_ex, df1_re = remove_intersection(df1, df2)
    ccle_with_ic50 = label_common_ccle(df1_re, df2)
    df1_ex.to_csv(GDSC_TUPLE_EXCLUDED_PATH)
    df1_re.to_csv(GDSC_TUPLE_REMAINED_PATH)
    ccle_with_ic50.to_csv(CCLE_TUPLE_COMMON_PATH)
    print(df1_ex.shape, df1_re.shape, ccle_with_ic50.shape)


if __name__ == "__main__":
    main(sys.argv[1:])
