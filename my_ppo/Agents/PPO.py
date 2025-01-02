import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
import numpy as np

from my_ppo.Base.AbstructBase import RLNet, AgentAbstract


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, reward, is_terminal):
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)


class ActorCritic(RLNet):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 has_continuous_action_space, 
                 action_std_init,
                 lr_actor,
                 lr_critic
                 ):
        super().__init__()
        ###############################################################
        # 環境設定


        ###############################################################
        # 静的パラメータ
        self.has_continuous_action_space = has_continuous_action_space
        
        # Pretrained ResNet
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False  # ResNetを凍結

        # state_dimが(フレーム数, 高さ, 幅)の次元を表すことを仮定(4, 84, 84)など
        state_channel = state_dim[0]

        # 学習率
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # 新しいチャネル統合用の学習可能なConv層
        self.channel_adapter = nn.Conv2d(state_channel, 3, kernel_size=1, stride=1, padding=0)
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(1000, 512),
                            nn.Tanh(),
                            nn.Linear(512, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(1000, 512),
                            nn.Tanh(),
                            nn.Linear(512, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(1000, 512),
                        nn.Tanh(),
                        nn.Linear(512, 128),
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

        features = self.channel_adapter(state)  # 4チャネルを3チャネルに変換
        # resnetで特徴量抽出
        features = self.resnet(features)

        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
        
        # distは確率分布
        # actionとして特定の行動をサンプリング
        action = dist.sample()
        # 確率分布から対数確率を取り出し
        action_logprob = dist.log_prob(action)
        # 価値関数で状態価値を推定
        state_val = self.critic(features)

        #detachでテンソルを勾配計算から切り離す
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    # 複数時点の行動価値評価
    def evaluate(self, state, action):

        features = self.channel_adapter(state)  # 4チャネルを3チャネルに変換
        # resnetで特徴量抽出
        features = self.resnet(features)

        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        # 分布のエントロピー
        dist_entropy = dist.entropy()
        # 状態価値
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy
    
    def get_state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "channel_adapter": self.channel_adapter.state_dict()
        }
    
    def set_state_dict(self, dict):
        self.actor.load_state_dict(dict["actor"])
        self.critic.load_state_dict(dict["critic"])
        self.channel_adapter.load_state_dict(dict["channel_adapter"])

    def get_parameter(self):
        return [
            {'params': self.channel_adapter.parameters(), 'lr': self.lr_actor},
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
        ]


class PPO(AgentAbstract):
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
                 action_std_init=0.6
                 ):
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
                         learn_every=learn_every
                         )
        
        ################################################################
        # 静的パラメータ
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 損失関数
        self.MseLoss = nn.MSELoss()

        ################################################################
        # 動的パラメータ
        if has_continuous_action_space:
            self.action_std = action_std_init
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, 
                                  action_dim, 
                                  has_continuous_action_space, 
                                  action_std_init,
                                  lr_actor,
                                  lr_critic,
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
        ################################################################
        # 環境設定


        


        

    def _set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def _decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def _learn(self):
        """パラメータ更新
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # 報酬の割引和をモンテカルロ法で計算
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # 報酬の割引和を正規化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # バッファ内の状態，行動などをテンソルに変換
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        # 位置エピソード分のデータをK_epochs分繰り返し学習
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            # squeezeで要素が1の次元を削除, (1, 3, 1) -> (3)に変更
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            # 重要度サンプリング比の計算
            # あえて対数空間で計算することで数値の安定性を確保する
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            # アドバンテージを比率でスケーリング
            surr1 = ratios * advantages
            # クリッピング
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # lossをnumpyに変換
            # detachで勾配計算をやめ，cpuに戻しnumpyに変換
            loss_np = loss.detach().cpu().numpy()

            # take gradient step
            # 勾配初期化
            self.optimizer.zero_grad()
            # 勾配計算
            loss.mean().backward()
            # 勾配をもとにパラメータを更新
            self.optimizer.step()
            
        # Copy new weights into old policy
        # 古いポリシーネットワークを更新
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        # バッファをクリア
        self.buffer.clear()
        
        # lossの四分位数を出力
        return loss_np.mean()

    
    
    def act(self, state):
        state = np.array(state)

        if self.has_continuous_action_space:
            # 古い状態用のポリシーは勾配計算をしない
            with torch.no_grad():
                # バッチ次元を追加
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            result = action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            result = action.item()
        
        # ステップを進める
        self.curr_step += 1

        return result

    def update(self, state, next_state, action, reward, done):
        # バッファへのデータ保存
        self.buffer.add(reward, done)

        loss = None
        # learn_everyステップごとに学習
        if self.curr_step % self.learn_every == 0:
            loss = self._learn()

        if self.curr_step % self.save_every == 0:
            self.save_model()

        result = {
            "loss": loss
        }
        
        return result
    
    def get_state_dict(self):
        dict = super().get_state_dict()
        # ネットワーク情報を取得
        dict.update({
            "has_continuous_action_space": self.has_continuous_action_space,
            "gamma": self.gamma,
            "eps_clip": self.eps_clip,
            "K_epochs": self.K_epochs,
            "optimizer": self.optimizer.state_dict(),
            "policy": self.policy.get_state_dict()
        })
        if self.has_continuous_action_space:
            dict.update({
                "action_std": self.action_std,
            })

        return dict

    def set_state_dict(self, dict):
        # ネットワークパラメータのセット
        self.policy.set_state_dict(dict.pop("policy"))
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer.load_state_dict(dict.pop("optimizer"))
        # その他のパラメータを一斉にセット
        self.__dict__.update(dict)
        
    
        
       


