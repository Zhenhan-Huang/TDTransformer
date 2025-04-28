import math

import numpy as np
import pandas as pd


class DataCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, data):
        x = pd.concat([row[0] for row in data])
        y = pd.concat([row[1] for row in data])
        return x, y