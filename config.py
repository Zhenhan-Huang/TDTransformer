import os

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from config import cfg
cfg = _C

_C.OUT_DIR = None
_C.PRETRAIN = False
_C.USE_LOGGER = True

_C.DATASET = CN()
_C.DATASET.NUM_NA_VAL = 'mode'
_C.DATASET.NUM_NORM = 'minmax'
_C.DATASET.CAT_ENCODE = False
# _C.DATASET.REMOVE_UNDERSCORE = False
_C.DATASET.BATCH_SIZE = 128

_C.DATASET.DECISION_TREE = CN()
# none, tokenize, vectorize
_C.DATASET.DECISION_TREE.DISCRETIZE_METHOD = 'vectorize'
# quantile, decision_tree
_C.DATASET.DECISION_TREE.BIN_METHOD = 'quantile'
# all, train
_C.DATASET.DECISION_TREE.DISCRETIZE_DOMAIN = 'all'
# Number of bins to discretize numerical values
_C.DATASET.DECISION_TREE.MAX_COUNT = 128
_C.DATASET.DECISION_TREE.MIN_SAMPLES_BIN = 8

_C.MODEL = CN()
_C.MODEL.USE_NUM_COL_NMS = True
_C.MODEL.DEPTH = 1
_C.MODEL.HIDDEN_DIM = 128
# gated_transformer, roberta 
_C.MODEL.BACKBONE = 'gated_transformer'
_C.MODEL.USE_POS_EMBED = True
_C.MODEL.USE_COL_TYPE_EMBED = False

_C.MODEL.PRETRAIN = CN()
_C.MODEL.PRETRAIN.SUPERVISE = False
_C.MODEL.PRETRAIN.CORRUPT_RATIO = 0.5

_C.OPTIM = CN()
_C.OPTIM.MAX_EPOCH = 50
_C.OPTIM.LR = 1e-4
_C.OPTIM.PATIENCE = 5
_C.OPTIM.WARMUP_STEPS = None