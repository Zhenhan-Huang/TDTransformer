import torch

import torch.nn as nn

from baselines.trainer import DeepTrainer
from .embedding import EmbeddingLayer


class MLPLayer(nn.Module):
    def __init__(self, dim_in, dim_out, norm='batch_norm', act='gelu'):
        super().__init__()
        self.layer_lst = nn.ModuleList()
        self.layer_lst.append(nn.Linear(dim_in, dim_out))
        
        if norm == 'batch_norm':
            self.layer_lst.append(nn.BatchNorm1d(dim_out))
        
        if act == 'gelu':
            self.layer_lst.append(nn.GELU())
        elif act == 'relu':
            self.layer_lst.append(nn.ReLU())
        
    def forward(self, x):
        for layer in self.layer_lst:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, dim_in, dim, dim_out, depth, mapping,
        num_cols, cat_cols, bin_cols,
        norm='batch_norm', act='gelu', device='cuda:0'
    ):
        super().__init__()
        self.embed = EmbeddingLayer(
            mapping, dim, num_cols, cat_cols, bin_cols, device=device
        )
        self.layer_in = MLPLayer(dim_in, dim)
        self.block_mid = nn.ModuleList()
        for _ in range(depth):
            self.block_mid.append(MLPLayer(dim, dim, norm, act))
        num_class = 1 if dim_out <=2 else dim_out
        self.layer_out = MLPLayer(dim, num_class)

        if dim_out > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.num_class = dim_out
        self.device = device
        self.to(device)

    def forward(self, x, y):
        x_embed = self.embed(x)
        x_embed = self.layer_in(x_embed)
        for block in self.block_mid:
            x_embed = block(x_embed)
        logits = self.layer_out(x_embed)

        if self.num_class <= 2:
            y_ts = torch.tensor(y.values).to(self.device).float()
            loss = self.loss_fn(logits.flatten(), y_ts)
        else:
            y_ts = torch.tensor(y.values).to(self.device).long()
            loss = self.loss_fn(logits, y_ts)
        
        loss = loss.mean()

        return logits, loss


class MLPTrainer(DeepTrainer):
    def __init__(
        self, dataset, trainset, valset, hidden_dim, num_class, depth,
        num_epoch, num_cols, cat_cols, bin_cols, output_dir,
        num_workers=8, shuffle=True, batch_size=4, lr=1e-4,
        scheduler='cosine', device='cuda:0', cfg=None,
    ):
        super().__init__(
            dataset, trainset, valset, output_dir,
            batch_size=batch_size, lr=lr, scheduler=scheduler,
            num_epoch=num_epoch, num_workers=num_workers, shuffle=shuffle,
            num_cols=num_cols, cat_cols=cat_cols, bin_cols=bin_cols, cfg=cfg
        )
        dim_in = (len(num_cols) + len(cat_cols) + len(bin_cols)) * hidden_dim
        dim_out = num_class
        self.model = MLP(
            dim_in, hidden_dim, dim_out, depth, self.mapping,
            num_cols, cat_cols, bin_cols, device=device
        )
