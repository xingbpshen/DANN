import torch
from torch.utils.data import Dataset


class KFold:
    #   whole_dataset should be type of MyDataset
    #   typically we set use_portion_frac as 1
    #   pattern train, validation, test
    def __init__(self, whole_dataset, k, use_portion_frac):
        self.use_size = int(len(whole_dataset) * use_portion_frac)
        self.k = k
        self.dataset = whole_dataset
        #   pointer [0, k-1]
        self.pointer = 0
        self.fold_size = int(self.use_size / k)

    def get_test(self):
        if self.use_size == len(self.dataset):
            return None
        else:
            # return self.dataset.get_items(self.use_size, len(self.dataset) - self.use_size)
            x_t, y_t = self.dataset.get_items(self.use_size, len(self.dataset) - self.use_size)
            return MyDataset.from_x_y(x_t, y_t)

    def get_next_train_validation(self):
        index = self.pointer * self.fold_size
        use_set_x, use_set_y = self.dataset.get_items(0, self.use_size)

        def get_remaining(ux, uy, ex_start_idx, ex_fold_s):
            ex_end_idx = ex_start_idx + ex_fold_s - 1
            if ex_start_idx == 0:
                if ex_end_idx == ux.shape[0] - 1:
                    return None
                else:
                    return ux[ex_end_idx + 1: ux.shape[0]], uy[ex_end_idx + 1: ux.shape[0]]
            else:
                if ex_end_idx == ux.shape[0] - 1:
                    return ux[0: ex_start_idx], uy[0: ex_start_idx]
                else:
                    x0, x1 = ux[0: ex_start_idx], ux[ex_end_idx + 1: ux.shape[0]]
                    y0, y1 = uy[0: ex_start_idx], uy[ex_end_idx + 1: ux.shape[0]]
                    return torch.cat((x0, x1), dim=0), torch.cat((y0, y1), dim=0)

        x_v, y_v = use_set_x[index: index + self.fold_size], use_set_y[index: index + self.fold_size]
        x_tr, y_tr = get_remaining(use_set_x, use_set_y, index, self.fold_size)

        self.pointer = self.pointer + 1
        if self.pointer >= self.k:
            self.pointer = 0

        # return x_tr, y_tr, x_v, y_v
        return MyDataset.from_x_y(x_tr, y_tr), MyDataset.from_x_y(x_v, y_v)


class MyDataset(Dataset):
    # def __init__(self, ccl_tensor, dd_tensor, ic50_tensor):
    #     self.x = torch.zeros((ic50_tensor.shape[0], ccl_tensor.shape[1] + dd_tensor.shape[1]), dtype=torch.float32)
    #     for i in range(self.x.shape[0]):
    #         self.x[i] = torch.cat((ccl_tensor[i], dd_tensor[i]))
    #     self.y = torch.Tensor(ic50_tensor)

    def __init__(self, x, y):
        def standardization(t):
            mean, std = t.mean(), t.std()
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    t[i][j] = (t[i][j] - mean) / std
            return t
        self.x = x
        self.y = y

    @classmethod
    def from_x_y(cls, x, y):
        return cls(x, y)

    @classmethod
    def from_ccl_dd_ic50(cls, ccl_tensor, dd_tensor, ic50_tensor):
        x = torch.zeros((ic50_tensor.shape[0], ccl_tensor.shape[1] + dd_tensor.shape[1]), dtype=torch.float32)
        for i in range(x.shape[0]):
            x[i] = torch.cat((ccl_tensor[i], dd_tensor[i]))
        y = torch.Tensor(ic50_tensor)
        return cls(x, y)

    @classmethod
    def from_ccl_dd_domain(cls, ccl_tensor, dd_tensor, domain):
        x = torch.zeros((ccl_tensor.shape[0], ccl_tensor.shape[1] + dd_tensor.shape[1]), dtype=torch.float32)
        for i in range(x.shape[0]):
            x[i] = torch.cat((ccl_tensor[i], dd_tensor[i]))
        if domain == 1:
            y = torch.ones((x.shape[0], 1))
        else:
            y = torch.zeros((x.shape[0], 1))
        return cls(x, y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_items(self, index, size):
        if size == 0:
            return None
        return self.x[index: (index + size)], self.y[index: (index + size)]

    def get_x(self):
        return self.x

    def get_n_feature(self):
        return self.x.shape[1]
