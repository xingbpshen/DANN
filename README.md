# Transfer learning with unsupervised domain adaptation on cancer cell lines

## 1. Preparation

### 1.1 Install dependencies

Please make sure that you have more than 30 GB storage.

```
pip install -r requirements.txt
```

### 1.2 Run dataset helper

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