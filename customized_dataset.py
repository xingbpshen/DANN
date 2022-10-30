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

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_x_y(cls, x, y):
        return cls(x, y)

    @classmethod
    def from_ccl_dd_ic50(cls, ccl_tensor, dd_tensor, ic50_tensor, op='cat', frac=1):
        use_n = int(ic50_tensor.shape[0] * frac)
        x = None
        if op == 'cat':
            x = torch.zeros((use_n, ccl_tensor.shape[1] + dd_tensor.shape[1]), dtype=torch.float32)
            for i in range(use_n):
                x[i] = torch.cat((ccl_tensor[i], dd_tensor[i]))
        elif op == 'mul':
            x = torch.zeros((use_n, ccl_tensor.shape[1], dd_tensor.shape[1]), dtype=torch.float32)
            for i in range(use_n):
                x[i] = torch.matmul(ccl_tensor[i].view(-1, 1), dd_tensor[i].view(1, -1))

        y = torch.Tensor(ic50_tensor[: use_n])

        if x is None:
            print('Error in from_ccl_dd_ic50().')

        return cls(x, y)

    @classmethod
    def from_ccl_dd_domain(cls, ccl_tensor, dd_tensor, domain, op='cat', frac=1):
        use_n = int(ccl_tensor.shape[0] * frac)
        x = None
        if op == 'cat':
            x = torch.zeros((use_n, ccl_tensor.shape[1] + dd_tensor.shape[1]), dtype=torch.float32)
            for i in range(use_n):
                x[i] = torch.cat((ccl_tensor[i], dd_tensor[i]))
        elif op == 'mul':
            x = torch.zeros((use_n, ccl_tensor.shape[1], dd_tensor.shape[1]), dtype=torch.float32)
            for i in range(use_n):
                x[i] = torch.matmul(ccl_tensor[i].view(-1, 1), dd_tensor[i].view(1, -1))

        y = torch.zeros((x.shape[0], 2))
        y[:, domain] = 1

        if x is None:
            print('Error in from_ccl_dd_domain().')

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

    def normalize(self, new_min, new_max, normalize_x=True, normalize_y=True):
        if normalize_x:
            x_min = self.x.min()
            x_max = self.x.max()
            self.x = (self.x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min
        if normalize_y:
            y_min = self.y.min()
            y_max = self.y.max()
            self.y = (self.y - y_min) / (y_max - y_min) * (new_max - new_min) + new_min


class KFoldSep:
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
            x1_t, x2_t, y_t = self.dataset.get_items(self.use_size, len(self.dataset) - self.use_size)
            return MyDatasetSep.from_x_y(x1_t, x2_t, y_t)

    def get_next_train_validation(self):
        index = self.pointer * self.fold_size
        use_set_x1, use_set_x2, use_set_y = self.dataset.get_items(0, self.use_size)

        def get_remaining(ux1, ux2, uy, ex_start_idx, ex_fold_s):
            ex_end_idx = ex_start_idx + ex_fold_s - 1
            length = uy.shape[0]
            if ex_start_idx == 0:
                if ex_end_idx == length - 1:
                    return None
                else:
                    return ux1[ex_end_idx + 1: length], ux2[ex_end_idx + 1: length], uy[ex_end_idx + 1: length]
            else:
                if ex_end_idx == length - 1:
                    return ux1[0: ex_start_idx], ux2[0: ex_start_idx], uy[0: ex_start_idx]
                else:
                    x1_0, x1_1 = ux1[0: ex_start_idx], ux1[ex_end_idx + 1: length]
                    x2_0, x2_1 = ux2[0: ex_start_idx], ux2[ex_end_idx + 1: length]
                    y_0, y_1 = uy[0: ex_start_idx], uy[ex_end_idx + 1: length]
                    return torch.cat((x1_0, x1_1), dim=0), torch.cat((x2_0, x2_1), dim=0), torch.cat((y_0, y_1), dim=0)

        x1_v, x2_v, y_v = use_set_x1[index: index + self.fold_size], use_set_x2[index: index + self.fold_size], use_set_y[index: index + self.fold_size]
        x1_tr, x2_tr, y_tr = get_remaining(use_set_x1, use_set_x2, use_set_y, index, self.fold_size)

        self.pointer = self.pointer + 1
        if self.pointer >= self.k:
            self.pointer = 0

        # return x_tr, y_tr, x_v, y_v
        return MyDatasetSep.from_x_y(x1_tr, x2_tr, y_tr), MyDatasetSep.from_x_y(x1_v, x2_v, y_v)


class MyDatasetSep(Dataset):

    def __init__(self, x1, x2, y):
        # CCL
        self.x1 = x1
        # DD
        self.x2 = x2
        self.y = y

    @classmethod
    def from_x_y(cls, x1, x2, y):
        return cls(x1, x2, y)

    @classmethod
    def from_ccl_dd_ic50(cls, ccl_tensor, dd_tensor, ic50_tensor, frac=1):
        use_n = int(ic50_tensor.shape[0] * frac)
        x1 = torch.zeros((use_n, ccl_tensor.shape[1]), dtype=torch.float32)
        x2 = torch.zeros((use_n, dd_tensor.shape[1]), dtype=torch.float32)
        for i in range(use_n):
            x1[i] = ccl_tensor[i]
            x2[i] = dd_tensor[i]

        y = torch.Tensor(ic50_tensor[: use_n])

        return cls(x1, x2, y)

    @classmethod
    def from_ccl_dd_domain(cls, ccl_tensor, dd_tensor, domain, frac=1):
        use_n = int(ccl_tensor.shape[0] * frac)
        x1 = torch.zeros((use_n, ccl_tensor.shape[1]), dtype=torch.float32)
        x2 = torch.zeros((use_n, ccl_tensor.shape[1]), dtype=torch.float32)
        for i in range(use_n):
            x1[i] = ccl_tensor[i]
            x2[i] = dd_tensor[i]

        y = torch.zeros((use_n, 2))
        y[:, domain] = 1

        return cls(x1, x2, y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]

    def get_items(self, index, size):
        if size == 0:
            return None
        return self.x1[index: (index + size)], self.x2[index: (index + size)], self.y[index: (index + size)]

    def get_x(self):
        return self.x1, self.x2

    def get_n_feature(self):
        return self.x1.shape[1], self.x2.shape[1]

    def get_min_max_tuples(self):
        return (self.x1.min(), self.x1.max()), (self.x2.min(), self.x2.max()), (self.y.min(), self.y.max())

    def get_min1tmin2_max1tmax2(self):
        return self.x1.min() * self.x2.min(), self.x1.max() * self.x2.max()
