# Transfer learning with unsupervised domain adaptation on cancer cell lines

## 1. Preparation

### 1.1 Clone the repository

Please make sure that you have more than 30 GB storage. In any terminal, execute the following. 

```
git clone https://github.com/AntonioShen/DANN.git
```

### 1.2 Download the GDSC and CCLE dataset

Google Drive: 

https://drive.google.com/drive/folders/1yz9UeFCF6MK0AeNCmnb-6rOVQILYaptL?usp=sharing

On the above webpage, download all 3 files in the gdsc folder and move them into the following folder.

```
$yourworkspace$/DANN/data/cleaned/gdsc/
```

Repeat the previous step for the other 3 files in the ccle folder, and move them into the following folder.

```
$yourworkspace$/DANN/data/cleaned/ccle/
```

### 1.3 Install dependencies

Under the project root, execute the following.

```
pip install -r requirements.txt
```

### 1.4 Run dataset helper

This generates essential median csv files for future processing.

```
python dataset_helper.py
```

### 1.3 Run csv parser

This processes csv files and transfer into pt files for loading as tensors.

```
python parse_to_data_tensors.py -o <OPTION> -n <NAME>
```

-h for help, if you are a first-time user, we suggest running the following.

```
python parse_to_data_tensors.py -o exclude -n all
```

This process will take some time to complete.

## 2. Experiments

### 2.1 Run pipeline

The main.py file includes implicit configuration for training and testing.

```
python main.py
```

### 2.2 Access results

If you did not configure your own W&B visualization before running main.py, you can find results and metrics inside the below folder.

```
$yourworkspace$/DANN/results/plots/
```

The embedded visual logger is implemented in logger.py, a typical usage is shown below.

```
from logger import Logger

# An instance of Logger class which requires a list of names
my_logger = Logger(['epoch', 'metric1', 'metric2'])

# Inside the training loop, record the current results
for epoch in range(1, epochs + 1):

    # Do some training
    y_hat = model(x)
    
    # Compute metrics
    m1 = loss_func1(y_hat, y)
    m2 = loss_func2(y_hat, y)
    
    # Call log() to record data or metrics
    # It requires all fields in names be filled in.
    my_logger.log({'epoch': epoch, 
                   'metric1': m1, 
                   'metric2': m2})

# Outside the loop, call save_plot() to save the trend plot
# The x-axis refers to values under the first name, here it is epoch
PATH = './result_plot.jpg'
my_logger.save_plot(PATH)
```