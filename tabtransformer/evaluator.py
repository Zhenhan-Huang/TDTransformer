import torch

import numpy as np


def predict(clf, 
    x_test,
    y_test=None,
    collate_fn=None,
    return_loss=False,
    eval_batch_size=256,
):
    '''Make predictions by TransTabClassifier.

    Parameters
    ----------
    clf: TransTabClassifier
        the classifier model to make predictions.

    x_test: pd.DataFrame
            input tabular data.

    y_test: pd.Series
        target labels for input x_test. will be ignored if ``return_loss=False``.
    
    return_loss: bool
        set True will return the loss if y_test is given.
    
    eval_batch_size: int
        the batch size for inference.

    Returns
    -------
    pred_all: np.array
        if ``return_loss=False``, return the predictions made by TransTabClassifier.

    avg_loss: float
        if ``return_loss=True``, return the mean loss of the predictions made by TransTabClassifier.

    '''
    clf.eval()
    pred_list, loss_list = [], []
    for i in range(0, len(x_test), eval_batch_size):
        bs_x_test = x_test.iloc[i:i+eval_batch_size]
        bs_y_test = y_test.iloc[i:i+eval_batch_size]
        with torch.no_grad():
            logits, loss = clf(bs_x_test, bs_y_test, aux=collate_fn.bins_distr)
        
        if loss is not None:
            loss_list.append(loss.item())
        if logits.shape[-1] == 1: # binary classification
            pred_list.append(logits.sigmoid().detach().cpu().numpy())
        else: # multi-class classification
            pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
    pred_all = np.concatenate(pred_list, 0)

    if logits.shape[-1] == 1:
        pred_all = pred_all.flatten()

    avg_loss = np.mean(loss_list)
    print(f'Test loss is {avg_loss:.4f}')

    if return_loss:
        return avg_loss
    else:
        return pred_all