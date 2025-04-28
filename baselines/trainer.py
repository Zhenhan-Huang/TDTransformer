import torch
import time

import numpy as np

from .trainer_utils import DataCollator
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, StepLR
from utils import count_params
from transtab.evaluator import EarlyStopping


class TrainDataset(Dataset):
    def __init__(self, trainset):
        self.x, self.y = trainset

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x.iloc[index-1:index]
        if self.y is not None:
            y = self.y.iloc[index-1:index]
        else:
            y = None
        return x, y


class DeepTrainer(object):
    def __init__(
        self, dataset, trainset, valset, output_dir, batch_size,
        lr=1e-4, scheduler='cosine', num_epoch=200, patience=10,
        num_workers=8, shuffle=True, eval_less_is_better=True,
        num_cols=None, cat_cols=None, bin_cols=None, cfg=None
    ):
        self.model = None
        self.cfg = cfg
        collator = DataCollator(cfg)
        self.trainloader = self._build_dataloader(
            trainset, batch_size, collator, num_workers, shuffle
        )
        self.testloader = self._build_dataloader(
            valset, batch_size, collator, num_workers, shuffle
        )
        self.args = {
            'lr': lr,
            'scheduler': scheduler,
            'num_epoch': num_epoch,
        }
        self.early_stopping = EarlyStopping(
            output_dir=output_dir, patience=patience, verbose=False,
            less_is_better=eval_less_is_better
        )

        if cat_cols is not None or bin_cols is not None:
            cat_cols = cat_cols + bin_cols
            x_cat = dataset[0][cat_cols].astype(str)

            mapping = {}
            for col_nm in x_cat:
                tmp = x_cat[col_nm].to_numpy()
                tmp_unique = np.unique(tmp)
                mapping[col_nm] = {e:i for i, e in enumerate(tmp_unique)}
            self.mapping = mapping
        else:
            self.mapping = None

    def _build_dataloader(
            self, trainset, batch_size, collator, num_workers=8, shuffle=True
    ):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        return trainloader

    def train(self):
        self.model.train()
        self.create_optimizer()
        self.create_scheduler()

        nparams = count_params(self.model)
        print(f'Number of params is {nparams/1e6:.2f} M')

        start_time = time.time()
        for epoch in range(self.args['num_epoch']):
            for data in self.trainloader:
                self.optimizer.zero_grad()
                logits, loss = self.model(*data)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

            eval_res = self.evaluate()
            print(f'epoch: {epoch}, test val_loss: {eval_res:.6f}')
            self.early_stopping(-eval_res, self.model)
            if self.early_stopping.early_stop:
                print('early stopped')
                break

            learning_rate = self.optimizer.param_groups[0]['lr']
            print(f'epoch: {epoch}, train loss: {loss:.4f}, lr: {learning_rate:.6f}, spent: {(time.time()-start_time):.1f} secs')

    def evaluate(self):
        self.model.eval()
        y_test, pred_lst, loss_list = [], [], []
        for data in self.testloader:
            label = data[1]
            y_test.append(label)
            with torch.no_grad():
                logits, loss = self.model(*data)
            loss_list.append(loss.item())
            if logits.shape[-1] == 1:
                pred_lst.append(logits.sigmoid().detach().cpu().numpy())
            else:
                pred_lst.append(torch.softmax(logits,-1).detach().cpu().numpy())
        
        pred_all = np.concatenate(pred_lst, 0)
        if logits.shape[-1] == 1:
            pred_all = pred_all.flatten()

        eval_res = np.mean(loss_list) 
        
        return eval_res

    def predict(self, x_test, y_test, eval_batch_size=256):
        self.model.eval()
        pred_list, loss_list = [], []
        for i in range(0, len(x_test), eval_batch_size):
            bs_x_test = x_test.iloc[i:i+eval_batch_size]
            bs_y_test = y_test.iloc[i:i+eval_batch_size]
            with torch.no_grad():
                logits, loss = self.model(bs_x_test, bs_y_test)
            loss_list.append(loss)
            if logits.shape[-1] == 1:
                pred_list.append(logits.sigmoid().detach().cpu().numpy())
            else:
                pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())

        pred_all = np.concatenate(pred_list, 0)
        return pred_all

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args['lr']
        )

    def create_scheduler(self):
        sch_nm = self.args['scheduler']
        if sch_nm == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, self.args['num_epoch'], eta_min=1e-4
            )
        elif sch_nm == 'constant':
            self.lr_scheduler = ConstantLR(
                self.optimizer, factor=0.1, total_iters=10
            )
        elif sch_nm == 'step':
            self.lr_scheduler = StepLR(
                self.optimizer, step_size=5, gamma=0.5
            )
        else:
            raise ValueError(f'The scheduler "{sch_nm}" not supported')