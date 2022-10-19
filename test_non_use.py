import torch
from customized_dataset import MyDataset, KFold

GDSC_TENSOR_PATH = './data/tensors/gdsc/'
CCLE_TENSOR_PATH = './data/tensors/ccle/'

gdsc_dataset = MyDataset.from_ccl_dd_ic50(torch.load(GDSC_TENSOR_PATH + 'CCL.pt'),
                                          torch.load(GDSC_TENSOR_PATH + 'DD.pt'),
                                          torch.load(GDSC_TENSOR_PATH + 'IC50.pt'))
print(len(gdsc_dataset))


five_fold = KFold(gdsc_dataset, 5, 0.9)
train_set, validate_set = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_tr, y_tr, x_v, y_v = five_fold.get_next_train_validation()
# x_t, y_t = five_fold.get_test()
# print(x_tr.shape, y_tr.shape, x_v.shape, y_v.shape)
# print(x_t.shape, y_t.shape)

print(len(train_set))

exit()
