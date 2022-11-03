import torch
from logger import Logger
from datetime import date
import os

# train_logger = Logger(['epoch', 'train_accuracy', 'train_loss'])
# train_logger.log({'epoch': 1, 'train_accuracy': 42, 'train_loss': torch.rand(1)})
# train_logger.log({'epoch': 2, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
# train_logger.log({'epoch': 3, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
# train_logger.log({'epoch': 4, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
# train_logger.save_plot('plot.jpg')
# train_logger.save_csv('df.csv')

a = torch.Tensor([[1, 2, 3],
                [2, 1, 3],
                [3, 2, 1],
                [1, 3, 2]])
print((a - torch.mean(a)) / torch.std(a))

exit()
