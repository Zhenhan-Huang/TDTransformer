import os
import time
import sys
import argparse
import logging
import random
import torch
import transtab
import tabtransformer
import berttab

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except Exception:
    has_xgboost = False
try:
    from catboost import CatBoostClassifier
    has_catboost = True
except Exception:
    has_catboost = False
from dataset_custom import load_data
from utils import encode_table_data
from config import cfg
from dataset_config import DATA_CONFIGS, NUM_CLASSES
from baselines.mlp import MLPTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tabular Deep Learning")
    parser.add_argument('--dataset', type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cfg_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_dataset_id', action='store_true', default=False)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
        help='modify config options using the command-line')

    return parser.parse_args()
 

def reset_args(args, cfg):
    if args.out_dir:
        cfg.OUT_DIR = args.out_dir


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
    print(' ')


def check_cfg(cfg):
    if not cfg.USE_LOGGER:
        print('Logger is NOT created!')


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Working directory: {args.out_dir}')

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    if cfg.USE_LOGGER:
        setup_logger(args.out_dir)

    assert args.dataset in DATA_CONFIGS, (
        f'Available datasets are {list(DATA_CONFIGS.keys())}, '
        f'but requested dataset is {args.dataset}.'
    )

    set_random_seed(args.seed)

    reset_args(args, cfg)
    print_args(args, cfg)
    check_cfg(cfg)
    logging.basicConfig(level=logging.INFO)
    # trainset: [X: DataFrame, y: Series]
    dataset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
        args.dataset, dataset_config=DATA_CONFIGS[args.dataset],
        num_classes=NUM_CLASSES[args.dataset], cfg=cfg, seed=args.seed
    )

    if args.model == 'transtab':
        model = transtab.build_classifier(
            cat_cols, num_cols, bin_cols, cfg=cfg, hidden_dim=cfg.MODEL.HIDDEN_DIM,
            num_class=NUM_CLASSES[args.dataset], num_layer=cfg.MODEL.DEPTH
        )
        print(f'Model:\n{model}')
        training_arguments = {
            'num_epoch': cfg.OPTIM.MAX_EPOCH,
            'batch_size': cfg.DATASET.BATCH_SIZE,
            'lr': cfg.OPTIM.LR,
            'eval_metric': 'val_loss',
            'eval_less_is_better': True,
            'output_dir': args.out_dir,
            'patience': cfg.OPTIM.PATIENCE,
            'warmup_steps': cfg.OPTIM.WARMUP_STEPS,
        }

        transtab.train(model, trainset, valset, cfg=cfg, **training_arguments)

        x_test, y_test = testset
        y_pred = transtab.predict(model, x_test, y_test=y_test)

    elif args.model == 'tabtransformer':
        checkpoint = None
        if cfg.PRETRAIN:
            print('Start pretraining...')
            pretrain_arguments = {
                'num_epoch': cfg.OPTIM.MAX_EPOCH,
                'batch_size': cfg.DATASET.BATCH_SIZE,
                'lr': cfg.OPTIM.LR,
                'eval_metric': 'val_loss',
                'eval_less_is_better': True,
                'output_dir': args.out_dir,
                'patience': cfg.OPTIM.PATIENCE,
                'warmup_steps': cfg.OPTIM.WARMUP_STEPS,
            } 
            # build pre-train model
            model, collate_fn = tabtransformer.build_contrastive_learner(
                dataset, cat_cols, num_cols, bin_cols,
                cfg=cfg, hidden_dim=cfg.MODEL.HIDDEN_DIM,
                num_layer=cfg.MODEL.DEPTH, supervised=cfg.MODEL.PRETRAIN.SUPERVISE
            )

            print('model:\n', model)

            tabtransformer.train(
                model, trainset, valset, collate_fn=collate_fn, cfg=cfg,
                **pretrain_arguments
            )
            checkpoint = model.state_dict()

        # build fine-tuning model
        model, collate_fn = tabtransformer.build_classifier(
            dataset, cat_cols, num_cols, bin_cols,
            cfg=cfg, hidden_dim=cfg.MODEL.HIDDEN_DIM,
            num_class=NUM_CLASSES[args.dataset], num_layer=cfg.MODEL.DEPTH
        )

        if checkpoint is not None:
            print('Load weights from the pretrain...')
            msg = model.load_state_dict(checkpoint, strict=False)
            print(f'Unexpected keys are {msg[0]}, Missing keys are {msg[1]}')

        print('Start supervised learning...')
        print(f'Model:\n{model}')
        training_arguments = {
            'num_epoch': cfg.OPTIM.MAX_EPOCH,
            'batch_size': cfg.DATASET.BATCH_SIZE,
            'lr': cfg.OPTIM.LR,
            'eval_metric': 'val_loss',
            'eval_less_is_better': True,
            'output_dir': args.out_dir,
            'patience': cfg.OPTIM.PATIENCE,
            'warmup_steps': cfg.OPTIM.WARMUP_STEPS,
        }
        
        tabtransformer.train(
            model, trainset, valset, cfg=cfg, collate_fn=collate_fn,
            **training_arguments
        )

        x_test, y_test = testset
        y_pred = tabtransformer.predict(
            model, x_test, y_test=y_test, collate_fn=collate_fn
        )

    elif args.model == 'xgboost':
        assert has_xgboost
        x_train, y_train = encode_table_data(trainset[0], trainset[1])
        x_val, y_val = encode_table_data(valset[0], valset[1])
        x_test, y_test = encode_table_data(testset[0], testset[1])
        if NUM_CLASSES[args.dataset] == 2:
            print(f'Total number of classes is {NUM_CLASSES[args.dataset]}, using binary classifier.')
            clf_xgb = XGBClassifier(
                objective='binary:logistic', max_depth=16, eta=0.1,
                n_estimators=2000,
                eval_metric='logloss'
            )
        else:
            print(f'Total number of classes is {NUM_CLASSES[args.dataset]}, using multiclass classifier.')
            clf_xgb = XGBClassifier(
                objective='multi:softmax', num_classes=NUM_CLASSES[args.dataset],
                max_depth=16, eta=0.1,
                n_estimators=2000,
                eval_metric='mlogloss'
            )

        clf_xgb.fit(
            x_train, y_train, eval_set=[(x_val, y_val)],
            early_stopping_rounds=160, verbose=10
        )

        tmp_pred = np.array(clf_xgb.predict_proba(x_test))
        if NUM_CLASSES[args.dataset] == 2:
            y_pred = tmp_pred[:, 1]
        else:
            y_pred = tmp_pred

    elif args.model == 'catboost':
        assert has_catboost
        x_train, y_train = encode_table_data(trainset[0], trainset[1])
        x_val, y_val = encode_table_data(valset[0], valset[1])
        x_test, y_test = encode_table_data(testset[0], testset[1])
        if NUM_CLASSES[args.dataset] == 2:
            clf_catb = CatBoostClassifier(
                iterations=2000, learning_rate=0.1, depth=6, loss_function='Logloss',
                task_type='GPU', devices='0'
            )
            
        else:
            clf_catb = CatBoostClassifier(
                iterations=2000, learning_rate=0.1, depth=6, loss_function='MultiClass',
                task_type='GPU', devices='0'
            )

        clf_catb.fit(
            x_train, y_train, eval_set=(x_val, y_val),
            early_stopping_rounds=160, verbose=10
        )

        tmp_pred = np.array(clf_catb.predict_proba(x_test))
        if NUM_CLASSES[args.dataset] == 2:
            y_pred = tmp_pred[:, 1]
        else:
            y_pred = tmp_pred

    elif args.model == 'mlp':
        mlp_trainer = MLPTrainer(
            dataset, trainset, valset, cfg.MODEL.HIDDEN_DIM, NUM_CLASSES[args.dataset],
            cfg.MODEL.DEPTH, cfg.OPTIM.MAX_EPOCH, num_cols, cat_cols, bin_cols,
            args.out_dir,
            batch_size=cfg.DATASET.BATCH_SIZE, lr=cfg.OPTIM.LR,
            cfg=cfg,
        )
        mlp_trainer.train()

        x_test, y_test = testset
        y_pred = mlp_trainer.predict(x_test, y_test)

    else:
        raise ValueError(f'Model "{args.model}" not supported')

    # get performance evaluation
    # ypred: numpy array, y_test: pd Series
    if NUM_CLASSES[args.dataset] == 1:
        raise NotImplementedError
    
    elif NUM_CLASSES[args.dataset] == 2:
        metric1 = transtab.evaluate(y_pred, y_test, metric='auc')
        metric2 = transtab.evaluate(y_pred, y_test, metric='acc')

    else:
        metric1 = transtab.evaluate(y_pred, y_test, metric='f1')
        metric2 = transtab.evaluate(y_pred, y_test, metric='acc')

    print(f'* Auc is {metric1[0]:.4f}, Acc is {metric2[0]:.4f}')


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()