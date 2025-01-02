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

from my_ppo.Envs.MarioEnv import MarioEnv


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

save_model_dir = Path("./data/DQN/mario/")
# save_model_dir.mkdir(parents=True)


class DQNNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class DQN:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        # デバイスのセット
        self.device = device

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = DQNNet(self.state_dim, self.action_dim).float()
        # デバイスに送る
        self.net = self.net.to(device=self.device)

        # 探索率
        self.exploration_rate = 0.1
        # 探索率の減衰
        self.exploration_rate_decay = 0.99999975
        # 探索率の最小値
        self.exploration_rate_min = 0.1
        # 現在のステップ数
        self.curr_step = 0

        # ネットワークを5*10^5ごとにセーブ
        self.save_every = 5e5 # no. of experiences between saving Mario Net
        
        # TensorDictReplayBuffer は、PyTorch 用のリプレイバッファの実装
        # LazyMemmapStorage は、リプレイバッファのストレージ（データ保存先）を管理するクラス
        # Lazy: データは必要になるまでメモリにロードされない
        # Memmap: 大量のデータを扱う場合、ディスク上にマップ（memory-mapped file）として保存
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9

        # 最適化関数と損失関数
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        # Q値で使われる損失関数で二乗誤差と絶対値誤差を組み合わせたもの
        self.loss_fn = torch.nn.SmoothL1Loss()

        # 学習前の最低実行回数
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            # epsilonの確率でランダム探索
            action_idx = np.random.randint(self.action_dim)
        
        # EXPLOIT
        else:
            # ポリシーネットワークにしたがって探索
            # stateの最初の要素をnumpy 配列に変換
            # stateに余計な要素が入っていてタプルがたになっていることを避ける
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            # unsqueeze[0]でバッチ次元を追加
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            # action_valuesが(batch_size, action_dim)なので行動次元について最大値を取り出す
            # ここでnumpy arrayに戻していることに注意
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            # タプルを崩す
            return x[0] if isinstance(x, tuple) else x
        # LazyFrameをnumpyに変換
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))



    def recall(self):
        """
        Retrieve a batch of experiences from memory
        リプレイバッファからのデータのサンプリング
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        # 配列に入っているスカラーは配列から取り出す
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """状態state, 行動actionにおけるポリシーネットワークの
        Q値の推定値を出力

        Args:
            state (tensor): 状態
            action (tensor): 行動

        Returns:
            tensor: 各バッチごとのQ値
        """
        # まずポリシーネットのQ値を出力
        current_Q = self.net(state, model="online")[
            # 全てのバッチについて行動actionについてのQ値を返す
            np.arange(0, self.batch_size), action
        ] # Q_online(s,a)

        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """TDターゲット，すなわち現時点のターゲットが正しいと仮定したときの真のQ値
        を計算．ただし，targetは勾配計算をしないため，勾配は保存しない

        Args:
            reward (tensor): 報酬
            next_state (tensor): 次時点の状態
            done (tensor): 終了したか

        Returns:
            tensor: TDターゲット
        """
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        # 次の状態で最適行動をとった際の
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float())* self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        """ポリシーネットワークの更新

        Args:
            td_estimate (torch): td法によるQ値の推定値
            td_target (torch): td法によるQ値の真の値

        Returns:
            float: ロス
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        """ターゲットネットワークをポリシーネットワークで更新
        """
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def save_model(self, save_path):
        """モデルの構造とパラメータを保存"""
        # torch.saveはPickleというシリアライズ形式を用いることで
        # データをバイナリ形式で効率的に保存
        torch.save(self.net, save_path)
        print(f"Model saved to {save_path}")


    def load_model(self, save_path):
        """保存したモデルをself.netにロード"""
        self.net = torch.load(save_path)
        print(f"Model loaded from {save_path} and synced with self.net.")

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))


def train_and_test():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    save_dir.mkdir(parents=True)
    env = MarioEnv("SuperMarioBros-1-1-v0", render_mode="None")

    mario = DQN(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    episodes = 40000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    mario.save_model(save_model_dir / f"epoch{episodes}.pth")

def test():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    env = MarioEnv("SuperMarioBros-1-1-v0", render_mode="human")

    load_episodes = 40000

    mario = DQN(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    mario.load_model(save_model_dir / f"epoch{load_episodes}.pth")

    episodes = 40
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        print(f"Episode: {e}/{episodes}")


if __name__ == "__main__":
    # train_and_test()
    test()