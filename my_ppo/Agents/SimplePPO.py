import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
import numpy as np
from my_ppo.Agents.PPO import RolloutBuffer, PPO
from my_ppo.Base.AbstructBase import RLNet


################################## PPO Policy ##################################


class ActorCritic(RLNet):
    """シンプルなPPOを構築
    入力状態は数次元の実数値
    """
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 has_continuous_action_space, 
                 action_std_init,
                 lr_actor,
                 lr_critic):
        super().__init__()
        ###############################################################
        # 環境設定


        ###############################################################
        # 静的パラメータ
        self.has_continuous_action_space = has_continuous_action_space
        
        # calcuate distribution 
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        
        # 学習率
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        state_size = int(torch.prod(torch.tensor(state_dim)))

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_size, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_size, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_size, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        ###############################################################
        # 動的パラメータ
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    # 一時点の行動決定
    def act(self, state):

        

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # distは確率分布
        # actionとして特定の行動をサンプリング
        action = dist.sample()
        # 確率分布から対数確率を取り出し
        action_logprob = dist.log_prob(action)
        # 価値関数で状態価値を推定
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    # 複数時点の行動価値評価
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        # 分布のエントロピー
        dist_entropy = dist.entropy()
        # 状態価値
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
    def get_state_dict(self):
        """保存する必要のあるパラメータを辞書型として返す

        Returns:
            ditc: 保存するネットワークパラメータを含んだ辞書
        """
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }
    
    def set_state_dict(self, dict):
        """ネットワークのパラメータデータをセットする

        Args:
            dict (ditc): ネットワークを復元するのに必要なパラメータの保存
        """
        self.actor.load_state_dict(dict["actor"])
        self.critic.load_state_dict(dict["critic"])

    def get_parameter(self):
        return [
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
        ]


class PPO(PPO):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 save_dir,
                 load_dir,
                 save_every,
                 learn_every,
                 lr_actor, 
                 lr_critic, 
                 gamma, 
                 K_epochs, 
                 eps_clip, 
                 has_continuous_action_space, 
                 action_std_init=0.6):
        """
        PPO (Proximal Policy Optimization) エージェントクラス

        このクラスは、強化学習アルゴリズムの一つであるPPOを実装したエージェントを定義します。
        状態次元、行動次元、学習パラメータ、保存・読み込みディレクトリなどを初期化します。

        Args:
            state_dim (tuple): 状態空間の次元を表すタプル。環境の観測次元に対応します。
            action_dim (int): 行動空間の次元。離散または連続行動空間の次元を指定します。
            save_dir (str or Path): モデルやログを保存するディレクトリ。
            load_dir (str or Path): 保存されたモデルを読み込むためのディレクトリ。
            save_every (int): モデルを保存するステップ間隔。
            learn_every (int): 学習を実行するステップ間隔。
            lr_actor (float): Actor（ポリシーネットワーク）の学習率。
            lr_critic (float): Critic（価値関数ネットワーク）の学習率。
            gamma (float): 割引率。将来の報酬の影響度を決定します (0 < gamma ≤ 1)。
            K_epochs (int): ポリシーの更新回数（エポック数）。一回のポリシー更新で
            同じデータについてパラメータを繰り返し更新する回数
            eps_clip (float): PPOクリッピング率。新旧ポリシーの比率の制約を設定します。
            has_continuous_action_space (bool): 行動空間が連続かどうかを指定します。
            action_std_init (float, optional): 連続行動空間での初期アクション標準偏差。デフォルトは0.6。
        """
        super().__init__(state_dim=state_dim, 
                         action_dim=action_dim, 
                         save_dir=save_dir,
                         load_dir=load_dir,
                         save_every=save_every,
                         learn_every=learn_every,
                         lr_actor=lr_actor,
                         lr_critic=lr_critic,
                         gamma=gamma,
                         K_epochs=K_epochs,
                         eps_clip=eps_clip,
                         has_continuous_action_space=has_continuous_action_space,
                         action_std_init=action_std_init
                         )
        ################################################################
        # 環境設定

        ################################################################
        # 静的パラメータ

        ################################################################
        # 動的パラメータ
        
        self.policy = ActorCritic(state_dim=state_dim, 
                                  action_dim=action_dim, 
                                  has_continuous_action_space=has_continuous_action_space, 
                                  action_std_init=action_std_init,
                                  lr_actor=lr_actor,
                                  lr_critic=lr_critic
                                  ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.get_parameter())

        self.policy_old = ActorCritic(state_dim=state_dim, 
                                      action_dim=action_dim, 
                                      has_continuous_action_space=has_continuous_action_space, 
                                      action_std_init=action_std_init,
                                      lr_actor=lr_actor,
                                      lr_critic=lr_critic
                                      ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        
       


