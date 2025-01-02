import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        # Pretrained ResNet
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False  # ResNetを凍結

        # state_dimが(フレーム数, 高さ, 幅)の次元を表すことを仮定(4, 84, 84)など
        state_channel = state_dim[0]

        # 新しいチャネル統合用の学習可能なConv層
        self.channel_adapter = nn.Conv2d(state_channel, 3, kernel_size=1, stride=1, padding=0)
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
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
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
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
            cov_mat = torch.diag_embed(action_var).to(device)
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


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.channel_adapter.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
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

    def select_action(self, state):
        state = np.array(state)

        if self.has_continuous_action_space:
            # 古い状態用のポリシーは勾配計算をしない
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # 報酬の割引和を正規化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # バッファ内の状態，行動などをテンソルに変換
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # ロスの値を記録
        quartile_1sts = []
        quartile_3rds = []
        losses = [] 

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
            # detachでtensorをやめ，cpuに戻しnumpyに変換
            loss_np = loss.detach().cpu().numpy()
            quartile_1sts.append(np.percentile(loss_np, 25))
            losses.append(np.percentile(loss_np, 50)) 
            quartile_3rds.append(np.percentile(loss_np, 75))

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
        return np.mean(quartile_1sts), np.mean(losses), np.mean(quartile_3rds)
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


