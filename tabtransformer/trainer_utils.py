import math

import numpy as np
import pandas as pd


class TrainCollator:
    '''A base class for all collate function used for TransTab training.
    '''
    def __init__(self,
        dataset,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        ignore_duplicate_cols=False,
        cfg=None,
        **kwargs,
    ):
        x_num = dataset[0][numerical_columns]
        bin_method = cfg.DATASET.DECISION_TREE.BIN_METHOD
        nbins = cfg.DATASET.DECISION_TREE.MAX_COUNT
        bins_distr = {}
        if bin_method == 'quantile':
            for col_nm, col_data in x_num.items():
                qb = pd.qcut(col_data, q=nbins, duplicates='drop', retbins=True)[1]
                bins_distr[col_nm] = qb

        elif bin_method == 'decision_tree':
            raise NotImplementedError
        
        else:
            raise ValueError(f'The bin method "{bin_method}" not supported')
        
        self.bins_distr = bins_distr
        
    
    def save(self, path):
        print('No feature_extractor in collate function, so it will not be saved')
    
    def __call__(self, data):
        raise NotImplementedError


class TabTransformerCollatorForSL(TrainCollator):
    """
    collator for the supervised learning
    """
    def __init__(self, dataset,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        cfg=None,
        **kwargs
    ):
        super().__init__(
            dataset,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            cfg=cfg
        )
    
    def __call__(self, data):
        x = pd.concat([row[0] for row in data])
        y = pd.concat([row[1] for row in data])
        return x, y, self.bins_distr


class TabTransformerCollatorForCL(TrainCollator):
    """
    collator for the contrastive learning
    """
    def __init__(self, dataset,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        overlap_ratio=0.5, 
        num_partition=1,
        ignore_duplicate_cols=False,
        supervise=False,
        corrupt_ratio=.5,
        cfg=None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            ignore_duplicate_cols=ignore_duplicate_cols,
            cfg=cfg
        )
        assert num_partition > 0, f'number of contrastive subsets must be greater than 0, got {num_partition}'
        assert isinstance(num_partition,int), f'number of constrative subsets must be int, got {type(num_partition)}'
        assert overlap_ratio >= 0 and overlap_ratio < 1, f'overlap_ratio must be in [0, 1), got {overlap_ratio}'
        self.overlap_ratio=overlap_ratio
        self.num_partition=num_partition
        self.supervise = supervise
        self.corrupt_ratio = corrupt_ratio
        self.cat_cols = categorical_columns
        self.num_cols = numerical_columns

    def __call__(self, data):
        '''
        data (List[Tuple(DataFrame, Series)]) - [(data, y)]
        '''
        # 1. build positive pairs
        # 2. encode each pair using feature extractor
        df_x = pd.concat([row[0] for row in data])
        df_y = pd.concat([row[1] for row in data])
        if self.num_partition > 1:
            raise NotImplementedError
        else:
            sub_x_list = self._build_positive_pairs_single_view(df_x)
        res = {'input_sub_x': sub_x_list}
        if self.supervise: 
            return res, df_y, self.bins_distr
        else:
            return res, None, self.bins_distr

    def _build_positive_pairs_single_view(self, x):
        sub_x_list = [x]
        x_corrupt = x.copy()
        # corrupt numerical columns
        n_num_corrupt = int(len(self.num_cols) * self.corrupt_ratio)
        num_select = np.random.choice(self.num_cols, size=n_num_corrupt, replace=False)
        x_num_corrupt = x_corrupt[num_select]
        np.random.shuffle(x_num_corrupt.values)
        x_corrupt[num_select] = x_num_corrupt
        # corrupt categorical columns
        n_cat_corrupt = int(len(self.cat_cols) * self.corrupt_ratio) 
        cat_select = np.random.choice(self.cat_cols, size=n_cat_corrupt, replace=False)
        x_cat_corrupt = x_corrupt[cat_select]
        np.random.shuffle(x_cat_corrupt.values)
        x_corrupt[cat_select] = x_cat_corrupt
        sub_x_list.append(x_corrupt)
        return sub_x_list