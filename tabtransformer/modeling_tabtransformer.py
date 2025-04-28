import os, pdb
import math
import collections
import json
import torch
from typing import Dict, Optional, Any, Union, Callable, List

from loguru import logger
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transtab import constants
from transtab.modeling_transtab import TransTabEncoder, TransTabTransformerLayer
from transformers import AutoConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaEncoder
)
from .embed import (
    TabTransformerFeatureExtractor, TabTransformerFeatureProcessor
)
from dataset_custom import TOKENIZER_CONFIG


class TransTabWordEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names, categorical and binary features.
    '''
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)
        return embeddings


class TransTabNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        if num_col_emb.ndim == 2:
            # (ncols, hidden_dim) -> (batch_size, ncols, hidden_dim)
            num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))

        if x_num_ts.ndim == 2:
            # (batch_size, ncols) -> (batch_size, ncols, hidden_dim)
            x_num_ts = x_num_ts.unsqueeze(-1).expand((-1,-1,num_col_emb.size(2)))

        num_feat_emb = num_col_emb * x_num_ts.float() + self.num_bias
        num_feat_emb = self.norm(num_feat_emb)

        return num_feat_emb


class TabTransformerInputEncoder(nn.Module):
    def __init__(self,
        feature_extractor,
        feature_processor,
        device='cuda:0',
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_processor = feature_processor
        self.device = device
        self.to(device)

    def forward(self, x, aux=None):
        tokenized = self.feature_extractor(x, aux=aux)
        embeds = self.feature_processor(**tokenized)
        return embeds


class Classifier(nn.Module):
    def __init__(self, num_class, hidden_dim=128) -> None:
        super().__init__()
        if num_class <= 2:
            # regression task or binary classification task
            self.fc = nn.Linear(hidden_dim, 1)
        else:
            # multi-class classification task
            self.fc = nn.Linear(hidden_dim, num_class)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x) -> Tensor:
        x = x[:,0,:] # take the cls token embedding
        x = self.norm(x)
        logits = self.fc(x)
        return logits


class TransTabProjectionHead(nn.Module):
    def __init__(self,
        hidden_dim=128,
        projection_dim=128):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, x) -> Tensor:
        h = self.dense(x)
        return h
    

class TabTransformerCLSToken(nn.Module):
    '''add a learnable cls token embedding at the end of each sequence.
    '''
    def __init__(self, hidden_dim, use_pos_embed=True, use_extend_mask=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim),b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim
        self.use_pos_embed = use_pos_embed
        self.use_extend_mask = use_extend_mask

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def create_sin_pos_embed(self, sequence_length, embedding_dim):
        position = torch.arange(sequence_length, dtype=torch.float)[:, None]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pos_embeddings = torch.zeros((sequence_length, embedding_dim))
        pos_embeddings[:, 0::2] = torch.sin(position * div_term)
        pos_embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embeddings

    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        # add classification token embedding
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
        dtype = embedding.dtype
        device = embedding.device

        if self.use_pos_embed:
            pos_embed = self.create_sin_pos_embed(embedding.size(1), embedding.size(2))
            embedding += pos_embed[None, :, :].type(dtype).to(device)

        outputs = {'hidden_states': embedding}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0],1).to(attention_mask.device), attention_mask], 1
            )

        att_mask = attention_mask
        if self.use_extend_mask: 
            att_mask = attention_mask[:, None, None, :]
            att_mask = (1.0 - att_mask) * torch.finfo(dtype).min

        outputs['attention_mask'] = att_mask

        return outputs


class TransTabEncoderWrapper(nn.Module):
    def __init__(self,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=2,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu'
    ):
        super().__init__()
        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        output = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        return (output, )


class TabTransformerModel(nn.Module):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        cfg=None,
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__()
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        if feature_extractor is None:
            feature_extractor = TabTransformerFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns,
                binary_columns=self.binary_columns,
                cfg=cfg,
                **kwargs,
            )
            print('Feature extractor is built')
        else:
            print('Feature extractor will not be built')

        feature_processor = TabTransformerFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            use_num_col_nms=cfg.MODEL.USE_NUM_COL_NMS,
            device=device,
            cfg=cfg
        )
        
        self.input_encoder = TabTransformerInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        self.backbone = cfg.MODEL.BACKBONE
        use_extend_mask = False
        if self.backbone == 'gated_transformer':
            self.encoder = TransTabEncoderWrapper(
                hidden_dim=hidden_dim,
                num_layer=num_layer,
                num_attention_head=num_attention_head,
                hidden_dropout_prob=hidden_dropout_prob,
                ffn_dim=ffn_dim,
                activation=activation,
            )

        elif self.backbone == 'roberta':
            config = AutoConfig.from_pretrained('FacebookAI/roberta-base')
            config.hidden_size = hidden_dim
            config.num_hidden_layers = num_layer
            config.num_attention_heads = num_attention_head
            config.hidden_dropout_prob = hidden_dropout_prob
            config.attention_probs_dropout_prob = hidden_dropout_prob
            config.intermediate_size = ffn_dim
            self.encoder = RobertaEncoder(config)
            use_extend_mask = True

        else:
            raise ValueError(f'Backbone "{self.backbone}" not supported.')

        self.cls_token = TabTransformerCLSToken(
            hidden_dim=hidden_dim, use_pos_embed=cfg.MODEL.USE_POS_EMBED,
            use_extend_mask=use_extend_mask
        )
        self.device = device
        self.to(device)

    def forward(self, x, y=None, aux=None):
        embeded = self.input_encoder(x, aux=aux)
        embeded = self.cls_token(**embeded)
        encoder_output = self.encoder(**embeded)[0]
        encoder_output = encoder_output[0]

        # get cls token
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding

    def load(self, ckpt_dir):
        # load model weight state dict
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        # save model weight state dict
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = Classifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')


class TabTransformerClassifier(TabTransformerModel):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        activation='relu',
        cfg=None,
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
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
        self.num_class = num_class
        self.clf = Classifier(num_class=num_class, hidden_dim=hidden_dim)

        if self.num_class == 1:
            self.loss_fn = nn.MSELoss(reduction='none')
        elif self.num_class == 2:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.to(device)

    def forward(self, x, y=None, aux=None):
        # default x is dict and processed by feature_extractor
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # input is dataframe
            inputs = self.input_encoder.feature_extractor(x, aux=aux)
        else:
            raise ValueError(f'TransTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs)[0] # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        if y is not None:
            # compute classification loss
            if self.num_class <= 2:
                y_ts = torch.tensor(y.values).to(self.device).float()
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


class TabTransformerForCL(TabTransformerModel):
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        overlap_ratio=0.1,
        num_partition=2,
        supervised=True,
        temperature=1,
        base_temperature=1,
        activation='relu',
        cfg=None,
        device='cuda:0',
        **kwargs,
    ) -> None:
        super().__init__(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
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
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_partition = num_partition
        self.overlap_ratio = overlap_ratio
        self.supervised = supervised
        self.device = device
        self.to(device)

    def forward(self, x, y=None, aux=None):
        # do positive sampling
        feat_x_list = []
        if isinstance(x, pd.DataFrame):
            raise NotImplementedError

        elif isinstance(x, dict):
            # pretokenized inputs
            for idx, input_x in enumerate(x['input_sub_x']):
                feat_x = self.input_encoder(input_x, aux=aux)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)[0]
                feat_x_proj = feat_x[:, 0, :]
                feat_x_proj = self.projection_head(feat_x_proj)
                feat_x_list.append(feat_x_proj)

        else:
            raise ValueError(f'expect input x to be pd.DataFrame or dict(pretokenized), get {type(x)} instead')

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, n_view, emb_dim

        if y is not None and self.supervised:
            # take supervised loss
            y = torch.tensor(y.values, device=feat_x_multiview.device)
            loss = self.supervised_contrastive_loss(feat_x_multiview, y)
        else:
            # compute cl loss (multi-view InfoNCE loss)
            loss = self.self_supervised_contrastive_loss(feat_x_multiview)
        return None, loss

    def _build_positive_pairs(self, x, n):
        x_cols = x.columns.tolist()
        sub_col_list = np.array_split(np.array(x_cols), n)
        len_cols = len(sub_col_list[0])
        overlap = int(np.ceil(len_cols * (self.overlap_ratio)))
        sub_x_list = []
        for i, sub_col in enumerate(sub_col_list):
            if overlap > 0 and i < n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i+1][:overlap]])
            elif overlap >0 and i == n-1:
                sub_col = np.concatenate([sub_col, sub_col_list[i-1][-overlap:]])
            sub_x = x.copy()[sub_col]
            sub_x_list.append(sub_x)
        return sub_x_list

    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def self_supervised_contrastive_loss(self, features):
        # features: batch size x # pairs x hidden dimension
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device).view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device) 

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # remove the logits on the diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def supervised_contrastive_loss(self, features, labels):
        labels = labels.contiguous().view(-1,1)
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features,dim=1),dim=0)

        # contrast_mode == 'all'
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss