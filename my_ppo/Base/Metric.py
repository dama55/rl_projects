import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
import numpy as np
from pathlib import Path
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import matplotlib.pyplot as plt

import time, datetime

from abc import ABC, abstractmethod

from .AbstructBase import MetricAbstract

class MetricReward(MetricAbstract):
    """報酬関数のメトリッククラス

    """
    def __init__(self, metric_name, noneable=False):
        super().__init__(metric_name, noneable=noneable)
        self.curr_ep_reward = 0

    def get_moving_string(self):
        return f"{self.moving_ave_metrics[-1]:15.3f}"
    
    def get_moving_title(self):
        return f"{'MeanReward':>15}"
    
    def _calc_step_metric(self, **kwargs):
        #こんな感じでパラメータを取り出して適切な処理をして返す
        reward = self._extract_param(kwargs, "reward")
        self.curr_ep_reward += reward
    
    def _calc_episode_metric(self):
        return self.curr_ep_reward
    
    def _calc_moving_average(self):
        return super()._calc_moving_average()
    
    def init_episode(self):
        super().init_episode()
        # エピソード報酬の初期化
        self.curr_ep_reward = 0

class MetricLength(MetricAbstract):
    """エピソード長
    """
    def __init__(self, metric_name, noneable=False):
        super().__init__(metric_name, noneable=noneable)
        self.curr_ep_length = 0

    def get_moving_string(self):
        return f"{self.moving_ave_metrics[-1]:15.3f}"
    
    def get_moving_title(self):
        return f"{'MeanLength':>15}"
    
    def _calc_step_metric(self, **kwargs):
        self.curr_ep_length += 1
    
    def _calc_episode_metric(self):
        return self.curr_ep_length
    
    def _calc_moving_average(self):
        return super()._calc_moving_average()
    
    def init_episode(self):
        super().init_episode()
        self.curr_ep_length = 0

class MetricLoss(MetricAbstract):
    """損失
    """
    def __init__(self, metric_name, noneable = True):
        super().__init__(metric_name, noneable=noneable)
        self.curr_ep_loss = 0
        self.curr_ep_loss_size = 0

    def get_moving_string(self):
        return f"{self.moving_ave_metrics[-1]:15.3f}"
    
    def get_moving_title(self):
        return f"{'MeanLoss':>15}"
    
    def _calc_step_metric(self, **kwargs):
        metric = self._extract_param(kwargs, "loss")
        if metric:
            self.curr_ep_loss += metric
            self.curr_ep_loss_size += 1

    def _calc_episode_metric(self):
        if self.curr_ep_loss_size == 0:
            return 0
        else:
            return np.round(self.curr_ep_loss /self.curr_ep_loss_size , 5)
        
    def _calc_moving_average(self):
        return super()._calc_moving_average()
        
    def init_episode(self):
        super().init_episode()
        self.curr_ep_loss = 0
        self.curr_ep_loss_size = 0

class MetricQValue(MetricAbstract):
    """Q値
    """
    def __init__(self, metric_name, noneable=True):
        super().__init__(metric_name, noneable=noneable)
        self.curr_q = 0
        self.curr_q_size = 0

    def get_moving_string(self):
        return f"{self.moving_ave_metrics[-1]:15.3f}"
    
    def get_moving_title(self):
        return f"{'MeanQValue':>15}"
    
    def _calc_step_metric(self, **kwargs):
        metric = self._extract_param(kwargs, "loss")
        if metric:
            self.curr_q += metric
            self.curr_q_size += 1

    def _calc_episode_metric(self):
        if self.curr_q_size == 0:
            return 0
        else:
            return np.round(self.curr_q /self.curr_q_size , 5)
        
    def _calc_moving_average(self):
        return super()._calc_moving_average()
        
    def init_episode(self):
        super().init_episode()
        self.curr_q = 0
        self.curr_q_size = 0
        
class MetricEpsilon(MetricAbstract):
    """Epsilonなど直前の値を表示するもの
    """

    def __init__(self, metric_name, noneable=False):
        super().__init__(metric_name, noneable)
        
    def get_moving_string(self):
        return f"{self.moving_ave_metrics[-1]:10.3f}"
    
    def get_moving_title(self):
        return f"{'Epsilon':>10}"
    
    def _calc_step_metric(self, **kwargs):
        metric = self._extract_param(kwargs, "epsilon")
        if metric:
            return metric

    def _calc_episode_metric(self):
        return self.step_metrics[-1]
    
    def _calc_moving_average(self):
        return self.episode_metrics[-1]
    
    def init_episode(self):
        return super().init_episode()