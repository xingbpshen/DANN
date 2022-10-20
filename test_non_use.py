import torch
from customized_dataset import MyDataset, KFold

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'

dataset = MyDataset.from_ccl_dd_ic50(torch.load(GDSC_TENSOR_PATH + 'CCL.pt'),
                                          torch.load(GDSC_TENSOR_PATH + 'DD.pt'),
                                          torch.load(GDSC_TENSOR_PATH + 'IC50.pt'))
# print(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'))
# dataset = MyDataset.from_ccl_dd_ic50(torch.load(CCLE_TENSOR_PATH + 'CCL_COMMON.pt'),
#                                           torch.load(CCLE_TENSOR_PATH + 'DD_COMMON.pt'),
#                                           torch.load(CCLE_TENSOR_PATH + 'IC50_COMMON.pt'))
print(len(dataset))


five_fold = KFold(dataset, 5, 0.9)
train_set, validate_set = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_t, y_t = five_fold.get_test()
# print(x_tr.shape, y_tr.shape, x_v.shape, y_v.shape)
# print(x_t.shape, y_t.shape)

print(len(train_set))
print(validate_set.x)

exit()
