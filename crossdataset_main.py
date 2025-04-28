import os
import time
import sys
import argparse
import logging
import transtab
import berttab

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except Exception:
    has_xgboost = False
from dataset_custom import load_data
from utils import encode_table_data
from config import cfg
from .dataset_config import DATA_CONFIGS, NUM_CLASSES


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tabular Deep Learning")
    parser.add_argument('dataset', nargs='+', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
    #     help='modify config options using the command-line')

    return parser.parse_args()
 

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def main():
    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Working directory: {args.out_dir}')
    setup_logger(args.out_dir)

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    
    #cfg.merge_from_list(args.opts)

    print_args(args, cfg)
    logging.basicConfig(level=logging.INFO)

    num_cls = NUM_CLASSES[(args.dataset)[0]]
    for ds in args.dataset:
        assert ds in DATA_CONFIGS, (
            f'Available datasets are {list(DATA_CONFIGS.keys())}, '
            f'but requested dataset is {args.dataset}.'
        )
        assert num_cls == NUM_CLASSES[ds], (
            f'Expected number of classes is {num_cls}, but get {NUM_CLASSES[ds]} '
            f'for dataset "{ds}"'
        )

    if isinstance(args.dataset, list):
        dataset_config = DATA_CONFIGS
    else:
        dataset_config = DATA_CONFIGS[args.dataset]

    # trainset: Optinal([DataFrame, Series], [(DataFrame, Series), (DataFrame, Series), ...]
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
        args.dataset, dataset_config=dataset_config, cfg=cfg)

    if args.model == 'transtab':
        model = transtab.build_classifier(
            cat_cols, num_cols, bin_cols, cfg=cfg,
            num_class=num_cls
        )
        print(f'Model:\n{model}')
        training_arguments = {
            'num_epoch': cfg.OPTIM.MAX_EPOCH,
            'patience': 20,
            'batch_size': 128,
            'lr': 1e-4,
            'eval_metric': 'val_loss',
            'eval_less_is_better': True,
            'output_dir': args.out_dir,
        }
        transtab.train(model, trainset, valset, cfg=cfg, **training_arguments)

        y_pred_all = []
        y_test_all = []
        for ds_test in testset: 
            x_test, y_test = ds_test
            y_pred = transtab.predict(model, x_test, y_test=y_test)
            y_pred_all.append(y_pred)
            y_test_all.append(y_test.values)

        y_pred_all = np.concatenate(y_pred_all)
        y_test_all = np.concatenate(y_test_all)

    elif args.model == 'berttab':
        model = berttab.build_classifier(
            cat_cols, num_cols, bin_cols, cfg=cfg,
            num_class=NUM_CLASSES[args.dataset])
        print(f'Model:\n{model}')
        training_arguments = {
            'num_epoch': 50,
            'batch_size': 128,
            'lr': 1e-4,
            'eval_metric': 'val_loss',
            'eval_less_is_better': True,
            'output_dir': args.out_dir,
        }
        if isinstance(trainset, list):
            raise NotImplementedError
        transtab.train(model, trainset, valset, cfg=cfg, **training_arguments)
        x_test, y_test = testset
        ypred = transtab.predict(model, x_test, y_test=y_test)

    elif args.model == 'xgboost':
        assert has_xgboost
        x_train, y_train = encode_table_data(trainset[0], trainset[1])
        x_val, y_val = encode_table_data(valset[0], valset[1])
        x_test, y_test = encode_table_data(testset[0], testset[1])
        if NUM_CLASSES[args.dataset] == 2:
            print(f'Total number of classes is {NUM_CLASSES[args.dataset]}, using binary classifier.')
            clf_xgb = XGBClassifier(
                max_depth=8, learning_rate=0.1, n_estimators=1000, verbosity=0,
                silent=None, objective='binary:logistic', booster='gbtree',
                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                subsample=0.7, colsample_bytree=1, colsample_bylevel=1, 
                colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                base_score=0.5, random_state=0, seed=args.seed,
            )
        else:
            print(f'Total number of classes is {NUM_CLASSES[args.dataset]}, using multiclass classifier.')
            clf_xgb = XGBClassifier(
                max_depth=8, learning_rate=0.1, n_estimators=1000, verbosity=0,
                silent=None, objective='multi:softmax', booster='gbtree',
                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                subsample=0.7, colsample_bytree=1, colsample_bylevel=1, 
                colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                base_score=0.5, random_state=0, seed=args.seed,
                num_classes=NUM_CLASSES[args.dataset],
            )

        clf_xgb.fit(
            x_train, y_train, eval_set=[(x_val, y_val)],
            early_stopping_rounds=40, verbose=10
        )
        tmp_pred = np.array(clf_xgb.predict_proba(x_test))
        ypred = tmp_pred[:, 1]

    else:
        raise ValueError(f'Model "{args.model}" not supported')

    if num_cls == 2:
        auc_lst = transtab.evaluate(y_pred_all, y_test_all, seed=args.seed, metric='auc')
    else:
        auc_lst = [0.]

    acc_lst = transtab.evaluate(y_pred_all, y_test_all, seed=args.seed, metric='acc')
    print(f'* Auc is {auc_lst[0]:.2f}, Acc is {acc_lst[0]:.2f}')           


class Logger:
    """Write console output to external text file.
    Imported from `https://github.com/KaiyangZhou/Dassl.pytorch/blob/c61a1b570ac6333bd50fb5ae06aea59002fb20bb/dassl/utils/logger.py#L11`
    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = os.path.join(output, "log.txt")

    if os.path.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath)


if __name__ == '__main__':
    main()