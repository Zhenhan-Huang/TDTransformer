import os
import sys
import openml
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, MinMaxScaler, Normalizer,
    KBinsDiscretizer
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import trange
from sklearn.tree import DecisionTreeClassifier

OPENML_DATACONFIG = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker']},
    'Click_prediction_small': None,
}

EXAMPLE_DATACONFIG = {
    "example": {
        "bin": ["bin1", "bin2"],
        "cat": ["cat1", "cat2"],
        "num": ["num1", "num2"],
        "cols": ["bin1", "bin2", "cat1", "cat2", "num1", "num2"],
        "binary_indicator": ["1", "yes", "true", "positive", "t", "y"],
        "data_split_idx": {
            "train":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "val":[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "test":[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        }
    }
}

TOKENIZER_CONFIG = {
    'bert': 'bert-base-uncased',
    'gated_transformer': 'bert-base-uncased',
    'roberta': 'FacebookAI/roberta-base',
}


def tokenize_c45_bin(data, y, cfg):
    """
        data (DataFrame) - cell values in numerical columns
    """
    name_tokenizer = TOKENIZER_CONFIG[cfg.MODEL.BACKBONE]
    tokenizer_tmp = AutoTokenizer.from_pretrained(name_tokenizer)
    path_tokenizer = f'cache/{name_tokenizer}'
    tokenizer_tmp.add_tokens(
        [f'nbin{i}' for i in range(cfg.DATASET.DECISION_TREE.MAX_COUNT)])
    start_token_id = tokenizer_tmp.convert_tokens_to_ids('nbin0')

    inds_all = []
    for col_nm in data:
        classifier = DecisionTreeClassifier(
            max_leaf_nodes=cfg.DATASET.DECISION_TREE.MAX_COUNT,
            min_samples_leaf=cfg.DATASET.DECISION_TREE.MIN_SAMPLES_BIN
        )
        X = (data[col_nm].values).reshape(-1, 1)
        tree = classifier.fit(X, y).tree_

        tree_thresholds = []
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] != tree.children_right[node_id]:
                tree_thresholds.append(tree.threshold[node_id])
        tree_thresholds.append(X.max())
        tree_thresholds.append(X.min())
        bin_edges = np.array(sorted(set(tree_thresholds)))

        indices = np.digitize(
            data[col_nm], np.r_[-np.inf, bin_edges[1:-1], np.inf]
        ).astype(np.int64)
        indices += start_token_id - 1
        inds_all.append(indices)

    # (batch_size x num_cols)
    inds_all = np.vstack(inds_all).T
    tokenizer_tmp.save_pretrained(path_tokenizer)

    # map token id back to str
    map_strategy = {i+start_token_id:f'nbin{i}' for i in range(cfg.DATASET.DECISION_TREE.MAX_COUNT)}
    vec_fun = np.vectorize(lambda x: map_strategy[x])
    inds_str = vec_fun(inds_all).tolist()

    return inds_str


def discrete_fix_bin(data, cfg):
    """
        data (DataFrame) - cell values in numerical columns
    """
    name_tokenizer = TOKENIZER_CONFIG[cfg.MODEL.BACKBONE]
    tokenizer_tmp = AutoTokenizer.from_pretrained(name_tokenizer)
    path_tokenizer = f'cache/{name_tokenizer}'
    tokenizer_tmp.add_tokens(
        [f'nbin{i}' for i in range(cfg.DATASET.DECISION_TREE.MAX_COUNT)])
    start_token_id = tokenizer_tmp.convert_tokens_to_ids('nbin0')
    tokenizer_tmp.save_pretrained(path_tokenizer)

    inds_all = []
    for col_nm in data:
        est = KBinsDiscretizer(
            n_bins=cfg.DATASET.DECISION_TREE.MAX_COUNT, encode='ordinal',
            strategy='uniform', subsample=200000
        )
        X = (data[col_nm].values).reshape(-1, 1)
        indices = est.fit_transform(X).astype(int).flatten()
        indices += start_token_id
        inds_all.append(indices)

    # (batch_size x num_cols)
    inds_all = np.vstack(inds_all).T

    # map token id back to str
    map_strategy = {i+start_token_id:f'nbin{i}' for i in range(cfg.DATASET.DECISION_TREE.MAX_COUNT)}
    vec_fun = np.vectorize(lambda x: map_strategy[x])
    inds_str = vec_fun(inds_all).tolist()

    return inds_str


def vectorize_bin(data, cfg):
    nbins = cfg.DATASET.DECISION_TREE.MAX_COUNT
    for col_nm, col_data in data.items():
        qb = pd.qcut(col_data, q=nbins, duplicates='drop', retbins=True)[1]
        # for numerical stability
        qb[0] = qb[0] * 0.999
        qb[-1] = qb[-1] * 1.001
        qb_mean = 0.5 * (qb[None, :-1] + qb[None, 1:])
        # (# rows, # bins)
        x_num_qt = 2.*(col_data.to_numpy()[:, None] - qb_mean) / (qb[None, 1:] - qb[None, :-1])
        
        #import ipdb; ipdb.set_trace()

        x_num_qt = np.clip(x_num_qt, -1, 1)
        pad_width = nbins - qb.size + 1
        x_num_quantile = np.pad(
            x_num_qt, ((0, 0), (pad_width, 0)), mode='constant',
            constant_values=-1
        )


def load_data(
    dataname, dataset_config=None, num_classes=1, data_cut=None, seed=123, cfg=None
):
    if dataset_config is None:
        dataset_config = OPENML_DATACONFIG

    if isinstance(dataname, str) or isinstance(dataname, int):
        # load a single tabular data
        return load_single_data(
            dataname=dataname, dataset_config=dataset_config,
            num_classes=num_classes, data_cut=data_cut, seed=seed, cfg=cfg
        )
    
    elif isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        all_list = []
        train_list, val_list, test_list = [], [], []
        for dataname_ in dataname:
            data_config = dataset_config.get(dataname_, None)
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(
                    dataname_, dataset_config=data_config, num_clases=num_classes,
                    data_cut=data_cut, seed=seed, cfg=cfg
                )
            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)

        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list

    else:
        raise ValueError(f'Dataset name "{dataname}" not supported')

def load_single_data(
    dataname, dataset_config=None, num_classes=1, data_cut=None, seed=123, cfg=None
):
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname}')
        filename = os.path.join(dataname, 'data_processed.csv')
        df = pd.read_csv(filename, index_col=0)
        y = df['target_label']
        X = df.drop(['target_label'],axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]

        X.columns = all_cols
        attribute_names = all_cols
        ftfile = os.path.join(dataname, 'numerical_feature.txt')
        if os.path.exists(ftfile):
            with open(ftfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []
        bnfile = os.path.join(dataname, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        cat_cols = [col for col in all_cols if col not in num_cols and col not in bin_cols]
        
    else:
        dataset = openml.datasets.get_dataset(
            dataname, download_data=True, download_qualities=True, download_features_meta_data=True
        )
        # X (DataFrame), y (Series), categorical_indicator (List[Boolean]),
        # attribute_names (List[Str]): column names
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe', target=dataset.default_target_attribute)
        
        if isinstance(dataname, int):
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            data_id = dataname
            dataname = openml_list.loc[openml_list.did == dataname].name.values[0]
        else:
            data_id = openml_list.loc[openml_list.name == dataname].index[0]
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict

        print('-'*60) 
        print(f'load data from {dataname}, dataset id is {data_id}')

        # drop cols which only have one unique value
        drop_cols = [col for col in attribute_names if X[col].nunique()<=1]

        all_cols = None
        # following the data preprocessing of SAINT: https://github.com/somepago/saint
        if data_id == 42178:
            cat_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod']
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            all_cols = cat_cols + num_cols
            tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
            X['TotalCharges'] = [float(i) for i in tmp]
            y = y[X.TotalCharges != 0]
            X = X[X.TotalCharges != 0]
            X.reset_index(drop=True, inplace=True)
        
        elif data_id == 41700:
            drop_cols.append('instance_id')

        elif data_id in [42728, 42705, 42729, 42571]:
            X, y = X[:50000], y[:50000]
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

        elif data_id == 183:
            # remove rows that has too few instances of a class
            X = X.drop([2108, 2209, 3149, 3280, 236, 294, 480, 719, 2201])
            y = y.drop([2108, 2209, 3149, 3280, 236, 294, 480, 719, 2201])
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

        if all_cols is None:
            all_cols = np.array(attribute_names)
            categorical_indicator = np.array(categorical_indicator)
            cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
            num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
            all_cols = [col for col in all_cols if col not in drop_cols]

        bin_cols = []
        # get categorical cols, binary cols and numerical cols from dataset_config
        if dataset_config is not None:
            if 'bin' in dataset_config:
                bin_cols = [c for c in cat_cols if c in dataset_config['bin']]

        cat_cols = [c for c in cat_cols if c not in bin_cols]


        if num_classes > 1:
            # encode target label for the classification problem
            y = LabelEncoder().fit_transform(y.values)
            y = pd.Series(y, index=X.index)

    if dataset_config is not None:
        if 'column_map' in dataset_config:
            column_map = dataset_config['column_map'] 
            X = X.rename(columns=column_map)

            for idx, bc in enumerate(bin_cols):
                if bc in column_map:
                    bin_cols[idx] = column_map[bc]

            for idx, cc in enumerate(cat_cols):
                if cc in column_map:
                    cat_cols[idx] = column_map[cc]

            for idx, nc in enumerate(num_cols):
                if nc in column_map:
                    num_cols[idx] = column_map[nc]

    # preprocessing numerical values
    if len(num_cols) > 0:
        if cfg.DATASET.NUM_NA_VAL == 'mode':
            for col in num_cols:
                X[col] = X[col].fillna(X[col].mode()[0])
        elif cfg.DATASET.NUM_NA_VAL == 'zero':
            for col in num_cols:
                X[col] = X[col].fillna(0)
            raise NotImplementedError
        else:
            raise ValueError(
                f'NA value assignment "{cfg.DATASET.NUM_NA_VAL}" not supported')

        if cfg.DATASET.NUM_NORM == 'minmax':
            print('Use min max normalization for numerical columns')
            X[num_cols] = MinMaxScaler().fit_transform(X[num_cols].values)

        elif cfg.DATASET.NUM_NORM == 'z-score':
            print('Use z-score normalization for numerical columns')
            X[num_cols] = Normalizer().fit_transform(X[num_cols].values)

        else:
            print('No data processing applied for numerical columns')
            assert cfg.DATASET.NUM_NORM == 'none', (
                f'Normalization for numerical columns "{cfg.DATASET.NUM_NORM}" '
                'not supported'
            )

    if len(cat_cols) > 0:
        #for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        for col in cat_cols: X[col] = X[col].fillna(X[col].mode()[0])
        # process cate
        if cfg.DATASET.CAT_ENCODE:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        #for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        for col in bin_cols: X[col] = X[col].fillna(X[col].mode()[0])
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).map(
                lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).map(
                lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(
                f'binary columns {bin_cols} contains values other than 0/1.')

    X = X[bin_cols + num_cols + cat_cols]
 
    # split train/val/test
    data_split_idx = None
    if dataset_config is not None:
        data_split_idx = dataset_config.get('data_split_idx', None)

    if data_split_idx is not None:
        train_idx = data_split_idx.get('train', None)
        val_idx = data_split_idx.get('val', None)
        test_idx = data_split_idx.get('test', None)

        if train_idx is None or test_idx is None:
            raise ValueError('train/test split indices must be provided together')
    
        else:
            train_dataset = X.iloc[train_idx]
            y_train = y[train_idx]
            test_dataset = X.iloc[test_idx]
            y_test = y[test_idx]
            if val_idx is not None:
                val_dataset = X.iloc[val_idx]
                y_val = y[val_idx]
            else:
                val_dataset = None
                y_val = None
    else:
        stratify = y if num_classes > 1 else None 
        # split train/val/test
        train_dataset, test_dataset, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=stratify, shuffle=True)
        val_size = int(len(y)*0.1)
        val_dataset = train_dataset.iloc[-val_size:]
        y_val = y_train[-val_size:]
        train_dataset = train_dataset.iloc[:-val_size]
        y_train = y_train[:-val_size]

    # if cfg.DATASET.REMOVE_UNDERSCORE:
    #     cat_cols = [c.replace('_', ' ') for c in cat_cols]
    #     num_cols = [c.replace('_', ' ') for c in num_cols]
    #     bin_cols = [c.replace('_', ' ') for c in bin_cols]

    if data_cut is not None:
        np.random.shuffle(all_cols)
        sp_size=int(len(all_cols)/data_cut)
        col_splits = np.split(all_cols, range(0,len(all_cols),sp_size))[1:]
        new_col_splits = []
        for split in col_splits:
            candidate_cols = np.random.choice(np.setdiff1d(all_cols, split), int(sp_size/2), replace=False)
            new_col_splits.append(split.tolist() + candidate_cols.tolist())
        if len(col_splits) > data_cut:
            for i in range(len(col_splits[-1])):
                new_col_splits[i] += [col_splits[-1][i]]
                new_col_splits[i] = np.unique(new_col_splits[i]).tolist()
            new_col_splits = new_col_splits[:-1]

        # cut subset
        trainset_splits = np.array_split(train_dataset, data_cut)
        train_subset_list = []
        for i in range(data_cut):
            train_subset_list.append(
                (trainset_splits[i][new_col_splits[i]], y_train.loc[trainset_splits[i].index])
            )
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (
            (X,y), train_subset_list, (val_dataset,y_val), (test_dataset,y_test),
            cat_cols, num_cols, bin_cols
        )

    else:
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(
            len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (
            (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset,y_test),
            cat_cols, num_cols, bin_cols
        )
