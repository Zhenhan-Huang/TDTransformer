import os

import pandas as pd

from transtab import constants
from tabtransformer.modeling_tabtransformer import (
    TabTransformerClassifier, TabTransformerFeatureExtractor,
    TabTransformerFeatureProcessor, TabTransformerForCL,
    TabTransformerInputEncoder, TabTransformerModel
)
from tabtransformer.trainer_utils import (
    TabTransformerCollatorForCL, TabTransformerCollatorForSL
)
from tabtransformer.trainer import Trainer
from transtab.dataset import load_data
from transtab.trainer_utils import random_seed


def build_classifier(
    dataset=None,
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    cfg=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs) -> TabTransformerClassifier:

    model = TabTransformerClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        cfg=cfg,
        device=device,
        **kwargs,
        )

    # build collate function for contrastive learning
    collate_fn = TabTransformerCollatorForSL(
        dataset=dataset,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        cfg=cfg,
    )
 
    if checkpoint is not None:
        model.load(checkpoint)

    return model, collate_fn

def build_extractor(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    ignore_duplicate_cols=False,
    disable_tokenizer_parallel=False,
    checkpoint=None,
    **kwargs,) -> TabTransformerFeatureExtractor:
    feature_extractor = TabTransformerFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_path):
            feature_extractor.load(extractor_path)
        else:
            feature_extractor.load(checkpoint)
    return feature_extractor

def build_encoder(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    **kwargs,
    ):
    if num_layer == 0:
        feature_extractor = TabTransformerFeatureExtractor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            )
        
        feature_processor = TabTransformerFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )

        enc = TabTransformerInputEncoder(feature_extractor, feature_processor)
        enc.load(checkpoint)
        
    else:
        enc = TabTransformerModel(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            )
        if checkpoint is not None:
            enc.load(checkpoint)

    return enc

def build_contrastive_learner(
    dataset=None,
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=1,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device='cuda:0',
    checkpoint=None,
    ignore_duplicate_cols=True,
    cfg=None,
    **kwargs,
    ): 
    model = TabTransformerForCL(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        num_partition= num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        cfg=cfg,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)
    
    # build collate function for contrastive learning
    collate_fn = TabTransformerCollatorForCL(
        dataset=dataset,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols,
        supervise=cfg.MODEL.PRETRAIN.SUPERVISE,
        corrupt_ratio=cfg.MODEL.PRETRAIN.CORRUPT_RATIO,
        cfg=cfg
    )
    if checkpoint is not None:
        collate_fn.feature_extractor.load(
            os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        )

    return model, collate_fn


def get_discretize_bin(dataset, cfg):
    dset_x = dataset[0]
    discretize_method = cfg.DATASET.DECISION_TREE.DISCRETIZE_METHOD
    nbins=cfg.DATASET.DECISION_TREE.MAX_COUNT
    bins_ret = {}
    if discretize_method == 'quantile':
        for col_nm, col_data in dset_x.items():
            qb = pd.qcut(col_data, q=nbins, duplicates='drop', retbins=True)[1]
            bins_ret[col_nm] = qb

    elif discretize_method == 'decision_tree':
        raise NotImplementedError
    
    else:
        raise ValueError(f'Discretize method "{discretize_method}" not supported')

    return bins_ret


def train(
    model, 
    trainset, 
    valset=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric='auc',
    output_dir='./ckpt',
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=False,
    eval_less_is_better=False,
    cfg=None,
    **kwargs,
):
    if isinstance(trainset, tuple): trainset = [trainset]

    train_args = {
        'num_epoch':num_epoch,
        'batch_size':batch_size,
        'eval_batch_size':eval_batch_size,
        'lr':lr,
        'weight_decay':weight_decay,
        'patience':patience,
        'warmup_ratio':warmup_ratio,
        'warmup_steps':warmup_steps,
        'eval_metric':eval_metric,
        'output_dir':output_dir,
        'collate_fn':collate_fn,
        'num_workers':num_workers,
        'balance_sample':balance_sample,
        'load_best_at_last':load_best_at_last,
        'ignore_duplicate_cols':ignore_duplicate_cols,
        'eval_less_is_better':eval_less_is_better,
    }
    trainer = Trainer(
        model,
        trainset,
        valset,
        cfg=cfg,
        **train_args,
    )
    trainer.train()
