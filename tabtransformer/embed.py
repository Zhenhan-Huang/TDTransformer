import os
import json
import collections
import torch
from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
from typing import Dict, Optional, Any, Union, Callable, List

import numpy as np
import pandas as pd
from torch import nn
from torch import Tensor

from transtab import constants
from sklearn.preprocessing import KBinsDiscretizer
from dataset_custom import TOKENIZER_CONFIG
import tabtransformer.modeling_tabtransformer as tabmodling
from .modeling_tabtransformer import TransTabTransformerLayer


class TabTransformerFeatureExtractor:

    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        cfg=None,
        **kwargs,
    ) -> None:
        self.discrete_method = cfg.DATASET.DECISION_TREE.DISCRETIZE_METHOD
        self.nbins = cfg.DATASET.DECISION_TREE.MAX_COUNT
        tokenizer_path = TOKENIZER_CONFIG[cfg.MODEL.BACKBONE]
        if self.discrete_method == 'bin':
            print(f'TabTransformer: Load tokenizer from "cache/{tokenizer_path}"')
            self.tokenizer = AutoTokenizer.from_pretrained(
                f'cache/{tokenizer_path}'
            )
        else:
            print(f'TabTransformer: Load Huggingface tokenizer from "{tokenizer_path}"')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.tokenizer.__dict__['model_max_length'] = 512

        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        #self.vocab_size = self.tokenizer.vocab_size
        # len(tokenizer) = vocab_size + # added tokens
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # check if column exists overlap
        col_no_overlap, duplicate_cols = self._check_column_overlap(
            self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(
                    f'Find duplicate cols named `{col}`, '
                    'please process the raw data or set `ignore_duplicate_cols` to True!'
                )
            assert col_no_overlap, (
                'The assigned categorical_columns, numerical_columns, '
                'binary_columns should not have overlap! Please check your input. '
                f'categorical columns: {self.categorical_columns}, '
                f'numerical columns: {self.numerical_columns}, '
                f'binary columns: {self.binary_columns}'
            )
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, aux=None, shuffle=False) -> Dict:
        encoded_inputs = {
            'x_num': None,
            'num_col_input_ids': None,
            'x_cat_input_ids': None,
            'x_bin_input_ids': None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill NAN with zero

            if self.discrete_method == 'vectorize':
                x_num_new = []
                qb_bins = aux
                for col_nm, col_data in x_num.items():
                    qb = qb_bins[col_nm]
                    numerator = 2*col_data.to_numpy()[:, None] - (qb[None, :-1] + qb[None, 1:])
                    denominator = qb[None, 1:] - qb[None, :-1]
                    x_num_qt = numerator / denominator
                    x_num_qt = np.clip(x_num_qt, -1, 1)
                    pad_width = self.nbins - qb.size + 1
                    x_num_quantile = np.pad(
                        x_num_qt, ((0, 0), (0, pad_width)),
                        mode='constant', constant_values=-1
                    )
                    x_num_new.append(np.expand_dims(x_num_quantile, axis=1))

                x_num_ts = torch.tensor(np.concatenate(x_num_new, axis=1), dtype=float)

                num_col_ts = self.tokenizer(
                    num_cols, padding=True, truncation=True,
                    add_special_tokens=False, return_tensors='pt'
                )
                num_cell_mask = None

                encoded_inputs['x_num'] = x_num_ts
                encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
                encoded_inputs['num_att_mask'] = num_col_ts['attention_mask']
                encoded_inputs['num_cell_mask'] = num_cell_mask

            elif self.discrete_method == 'none':
                # x_num: batch size x num cols
                x_num_ts = torch.tensor(x_num.values, dtype=float)
                num_col_ts = self.tokenizer(
                    num_cols, padding=True, truncation=True,
                    add_special_tokens=False, return_tensors='pt'
                )
                encoded_inputs['x_num'] = x_num_ts
                encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
                encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

            else:
                raise ValueError(f'Discrete method "{self.discrete_method}" not supported')

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('NAN')
            x_cat = x_cat.apply(lambda x: x.name + ' ' + x) * x_mask # mask out nan features
            x_cat_str = x_cat.agg(','.join, axis=1).values.tolist()
            x_num_ts = self.tokenizer(
                x_cat_str, padding=True, truncation=True,
                add_special_tokens=False, return_tensors='pt'
            )
            encoded_inputs['x_cat_input_ids'] = x_num_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_num_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols] # x_bin should already be integral (binary values in 0 & 1)
            x_bin_ts = torch.tensor(x_bin.values, dtype=float)
            bin_col_ts = self.tokenizer(
                bin_cols, padding=True, truncation=True,
                add_special_tokens=False, return_tensors='pt'
            ) 

            encoded_inputs['x_bin'] = x_bin_ts
            encoded_inputs['x_bin_input_ids'] = bin_col_ts['input_ids']
            encoded_inputs['bin_att_mask'] = bin_col_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        '''save the feature extractor configuration to local dir.
        '''
        save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save tokenizer
        tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        print(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
        '''update cat/num/bin column maps.
        '''
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)
        raise NotImplementedError

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
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


class TabTransformerFeatureProcessor(nn.Module):

    def __init__(self,
        vocab_size=None,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        use_num_col_nms=True,
        device='cuda:0',
        cfg=None
    ) -> None:
        super().__init__()
        self.word_embedding = tabmodling.TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id,
        )
        self.discrete_method = cfg.DATASET.DECISION_TREE.DISCRETIZE_METHOD
        nbins = cfg.DATASET.DECISION_TREE.MAX_COUNT
        self.num_embedding = tabmodling.TransTabNumEmbedding(hidden_dim)
        self.bin_embedding = tabmodling.TransTabNumEmbedding(hidden_dim)
        if self.discrete_method == 'none':
            print('In feature extractor, numerical columns are continuous')
        else:
            print('In feature extractor, numerical columns are discrete')

        num_dim = hidden_dim
        if self.discrete_method == 'vectorize':
            self.num_tf = nn.Linear(nbins, hidden_dim, bias=True)

        self.align_num = nn.Linear(num_dim, hidden_dim, bias=True)
        self.align_cat = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.align_bin = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.use_num_col_nms = use_num_col_nms
        self.device = device

        if cfg.MODEL.USE_COL_TYPE_EMBED:
            self.ctype_embedding = nn.Embedding(3, hidden_dim)
            self.use_col_type_embed = True
        else:
            self.ctype_embedding = None
            self.use_col_type_embed = False

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1,keepdim=True).to(embs.device)
            return embs

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None,
        x_bin=None,
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs,
    ) -> Tensor:
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        if x_num is not None:
            if self.discrete_method == 'bin':
                bs, ncols = x_num.size()
                _, seq_len = num_col_input_ids.size()
                num_feat_embedding = self.word_embedding(num_col_input_ids.to(self.device))
                hidden_size = num_feat_embedding.size(-1)
                num_feat_embedding = num_feat_embedding.reshape(bs, ncols, seq_len, hidden_size)
                num_att_mask = num_att_mask.reshape(bs, ncols, seq_len)
                num_feat_embedding[num_att_mask == 0] = 0
                num_feat_embedding = num_feat_embedding.sum(2) / num_att_mask.sum(
                    2, keepdim=True).to(num_feat_embedding.device)

                x_num = x_num.to(self.device)
                num_feat_embedding = self.num_embedding(num_feat_embedding, x_num)
                num_feat_embedding = self.num_encoder(
                    num_feat_embedding, src_key_padding_mask=torch.ones(
                        num_feat_embedding.shape[0], num_feat_embedding.shape[1]).to(self.device)
                )

            elif self.discrete_method == 'vectorize':
                # (batch_size, # num cols, hidden_dim)
                x_num = x_num.to(self.device)
                x_num = self.num_tf(x_num.float())

                # num_col_input_ids: [# cols, max # tokens]
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device))
                num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
                num_feat_embedding = self.num_embedding(num_col_emb, x_num)

            elif self.discrete_method == 'none':
                num_col_emb = self.word_embedding(num_col_input_ids.to(self.device)) # number of cat col, num of tokens, embdding size
                x_num = x_num.to(self.device)
                if self.use_num_col_nms:
                    num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
                    num_feat_embedding = self.num_embedding(num_col_emb, x_num)

                else:
                    # num_cols_emb has dtype of float
                    # x_num has dtype of double
                    num_feat_embedding = x_num.unsqueeze(-1).expand(
                        -1, -1, num_col_emb.size(-1)).float()
            
            else:
                raise ValueError(f'Discretize method "{self.discrete_method}" not supported')

            # (batch_size, num_cols, hidden_dim)
            num_feat_embedding = self.align_num(num_feat_embedding)
            if self.use_col_type_embed:
                num_type_embedding = self.ctype_embedding(torch.tensor([0], device=self.device))[None, :]
                num_feat_embedding += num_type_embedding

        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            # (batch_size, seq_len, hidden_dim)
            cat_feat_embedding = self.align_cat(cat_feat_embedding)
            if self.use_col_type_embed:
                cat_type_embedding = self.ctype_embedding(torch.tensor([1], device=self.device))[None, :]
                cat_feat_embedding += cat_type_embedding

        if x_bin_input_ids is not None:
            x_bin = x_bin.to(self.device)
            bin_col_embed = self.word_embedding(x_bin_input_ids.to(self.device))
            # (ncols, seq_len, hidden_dim) -> (ncols, hidden_dim)
            bin_col_embed = self._avg_embedding_by_mask(bin_col_embed, bin_att_mask)
            bin_feat_embedding = self.bin_embedding(bin_col_embed, x_bin)
            # (batch_size, num_cols, hidden_dim)
            bin_feat_embedding = self.align_bin(bin_feat_embedding)
            if self.use_col_type_embed:
                bin_type_embedding = self.ctype_embedding(torch.tensor([2], device=self.device))[None, :]
                bin_feat_embedding += bin_type_embedding

        # concat all embeddings
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]

        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [torch.ones(bin_feat_embedding.shape[0], bin_feat_embedding.shape[1])]

        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]

        if len(emb_list) == 0:
            raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
    
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}