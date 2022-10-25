import torch
from logger import Logger

train_logger = Logger(['epoch', 'train_accuracy', 'train_loss'])
train_logger.log({'epoch': 1, 'train_accuracy': 42, 'train_loss': torch.rand(1)})
train_logger.log({'epoch': 2, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
train_logger.log({'epoch': 3, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
train_logger.log({'epoch': 4, 'train_accuracy': torch.rand(1), 'train_loss': torch.rand(1)})
print(train_logger.df)
train_logger.save_plot('plot.jpg')

exit()
