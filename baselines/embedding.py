import torch

import torch.nn as nn

from transformers import BertTokenizerFast


class NumEmbedding(nn.Module):
    def __init__(self, nfts, ndims):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nfts, ndims))
        self.bias = nn.Parameter(torch.empty(nfts, ndims))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)

    @property
    def num_features(self):
        return self.weight.shape[0]

    @property
    def num_dims(self):
        return self.weight.shape[1]

    def forward(self, x):
        # (bs, ncols) -> (bs, ncols, dim)
        x = self.weight[None] * x[..., None]
        x = x + self.bias[None]
        return x


class CatEmbedding(nn.Module):
    def __init__(self, num_embeds, nfts, ndims):
        super().__init__()
        self.embedding = nn.Embedding(num_embeds, ndims)
        self.bias = nn.Parameter(torch.empty(nfts, ndims))
        nn.init.normal_(self.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.bias[None]
        return x


class EmbeddingLayer(nn.Module):
    def __init__(
        self, mapping, ndims, num_cols, cat_cols, bin_cols, device='cuda:0'
    ):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.num_cols = num_cols
        self.cat_cols = cat_cols + bin_cols
        nfts_num = len(num_cols)
        nfts_cat = len(self.cat_cols)
        if nfts_num > 0:
            self.num_embed = NumEmbedding(nfts_num, ndims)
        if nfts_cat > 0:
            nmaps = [len(strdict) for strdict in mapping.values()]
            num_embeds = max(nmaps) 
            self.cat_embed = CatEmbedding(num_embeds, nfts_cat, ndims)
        self.mapping = mapping
        self.device = device

    def forward(self, x):
        col_nms = x.columns.tolist()
        num_cols = [c for c in col_nms if c in self.num_cols] if self.num_cols is not None else []
        cat_cols = [c for c in col_nms if c in self.cat_cols] if self.cat_cols is not None else []

        x_num = None
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = torch.tensor(x_num.values, dtype=torch.float).to(self.device)
            x_num = self.num_embed(x_num)

        x_cat = None
        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_inds = []
            for col_nm in x_cat:
                tmp = x_cat[col_nm]
                tmp_inds = [self.mapping[col_nm][element] for element in tmp]
                x_inds.append(tmp_inds)
            x_cat = torch.tensor(x_inds, dtype=torch.long).to(self.device)
            x_cat = x_cat.T
            x_cat = self.cat_embed(x_cat)

        if x_num is not None and x_cat is not None:
            x_embed = torch.cat([x_cat, x_num], dim=1)
        elif x_cat is not None:
            x_embed = x_cat
        elif x_num is not None:
            x_embed = x_num
        else:
            raise RuntimeError('No features is provided')

        bs, seq, dim = x_embed.size()
        x_embed = x_embed.reshape(bs, seq*dim)

        return x_embed
