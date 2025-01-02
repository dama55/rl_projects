import numpy as np
import torch
from my_ppo.MyPPO_old import PPO
from my_ppo.Agents.SimplePPO import PPO as simpPPO
from my_ppo.Envs.MarioEnv import MarioEnv
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import gym

basedir = "./data/"

def test(env, agent, agent_episode = 10, num_episodes=5, render_mode="human", net_name = "mario"):
    """
    学習済みエージェントをテストする関数

    Parameters:
    - env: 環境オブジェクト
    - agent: 学習済みエージェント
    - num_episodes: テストするエピソード数
    - render_mode: 環境のレンダリングモード（例: "human"）
    """
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    # 学習済みモデルをロード
    agent.load(basedir + f"ppo_{net_name}_ep{agent_episode}.pth")

    for episode in range(num_episodes):
        state, _ = env.reset()  # 環境をリセット
        episode_reward = 0  # 累積報酬

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)

            if render_mode == "human":
                env.render()  # 環境を可視化

            state = next_state
            episode_reward += reward

        print(f"Test Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()

def save_animation(env, agent, agent_episode = 10, num_episodes=5, filename="mario", max_timesteps=500, net_name="mario"):
    """
    学習済みエージェントの動作をGIFとして保存する関数

    Parameters:
    - env: 環境オブジェクト
    - agent: 学習済みエージェント
    - filename: 保存するGIFのファイル名
    - max_timesteps: GIFに含める最大タイムステップ数
    """
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent.load(basedir + f"ppo_{net_name}_ep{agent_episode}.pth")
    episodes = []
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # 環境をリセット
        episode_reward = 0  # エピソードの累積報酬
        frames = []  # アニメーションのフレームを保存するリスト

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)

            # フレームを取得
            frame = env.render()  # 古いバージョンのAPI
            frames.append(frame.copy())


            # 状態を更新
            state = next_state
            episode_reward += reward

            if done:
                break

        episodes.append(frames)
        rewards.append(episode_reward)


    best_frames = episodes[np.argmax(rewards)]

    print("rewards list: ", rewards)
    print("best reward: ", np.max(rewards))

    # フレームを Pillow イメージに変換
    frames_pil = [Image.fromarray(frame) for frame in best_frames]

    # GIF を保存
    frames_pil[0].save(
        basedir + f"{filename}_animation.gif",
        save_all=True,          # すべてのフレームを保存
        append_images=frames_pil[1:],  # 他のフレームを追加
        duration=33,            # フレーム間隔（ミリ秒単位、fps=30なら約33ms）
        loop=0                  # 無限ループ
    )
    print("Animation saved as animation.gif")



def train(env, agent, agent_episode = 0, num_episodes=1000, max_timesteps=500, update_episode=4, file_name = "mario"):
    """
    環境とエージェントを使って強化学習を行う関数

    Parameters:
    - env: 環境オブジェクト（例: OpenAI Gym 環境）
    - agent: 強化学習エージェント（例: PPOクラスのインスタンス）
    - num_episodes: 学習するエピソード数（デフォルト: 1000）
    - max_timesteps: 1エピソード内の最大タイムステップ数（デフォルト: 500）
    - update_timestep: エージェントを更新するタイミングとなる総タイムステップ数（デフォルト: 2000）
    """
    loss_stats = []  # ロスの四分位数を記録するリスト
    if agent_episode > 0:
        try:
            agent.load(basedir + f"ppo_{file_name}_ep{agent_episode}.pth")
        except FileNotFoundError:
            print("Warning: Model file not found. Starting with a fresh model.")
        # ロスの四分位数を保存
        try:
            with open(basedir + f"{file_name}_loss_stats_ep{agent_episode}.json", "r") as f:
                import json
                loss_stats = json.load(f)
        except FileNotFoundError:
            print("Warning: Loss statistics file not found. Starting fresh.")

    # ロードしたエージェントに追加するエピソード
    total_episodes = num_episodes + agent_episode
    
    # 追加エピソード分だけ学習
    for added_apisode in range(num_episodes):
        state, _ = env.reset()  # 環境をリセット
        episode_reward = 0  # エピソードの累積報酬

        episode = added_apisode + agent_episode

        for t in range(max_timesteps):

            # エージェントが行動を選択
            action = agent.select_action(state)
            
            # 環境との相互作用
            next_state, reward, done, trunc, info = env.step(action)

            # バッファに保存
            agent.buffer.add(reward, done)

            # 状態を更新
            state = next_state
            episode_reward += reward


            if done:
                break
        
        # update_episodeごとにエージェントを更新
        if (episode + 1) % update_episode == 0:
            quartile_1st, med, quartile_3rd = agent.update()
            # ロスを記録
            loss_stats.append({
                "episode": episode + 1,
                "quartile_1st": quartile_1st,
                "median": med,
                "quartile_3rd": quartile_3rd,
            })

            # ロスの中央値を出力
            print(f"Update: Episode {episode + 1}, Median Loss = {med:.4f}")

        # エピソード終了時に結果を表示
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    # 学習後のモデルを保存（必要なら）
    agent.save(basedir + f"ppo_{file_name}_ep{total_episodes}.pth")
    print("Training complete!")

    # ロスの四分位数を保存
    with open(basedir + f"{file_name}_loss_stats_ep{total_episodes}.json", "w") as f:
        import json
        json.dump(loss_stats, f, indent=4)
    print("Loss statistics saved to loss_stats.json")

    # グラフを描画して保存
    plot_loss_statistics(loss_stats, basedir + f"{file_name}_loss_statistics_ep{total_episodes}.png")


def plot_loss_statistics(loss_stats, filename):
    """
    ロスの四分位数をプロットして保存する関数

    Parameters:
    - loss_stats: 四分位数のデータリスト
    - filename: 保存するグラフのファイル名
    """
    episodes = [stat["episode"] for stat in loss_stats]
    quartile_1sts = [stat["quartile_1st"] for stat in loss_stats]
    medians = [stat["median"] for stat in loss_stats]
    quartile_3rds = [stat["quartile_3rd"] for stat in loss_stats]

    # グラフを描画
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, quartile_1sts, label="1st Quartile (25%)", linestyle="--", color="blue")
    plt.plot(episodes, medians, label="Median (50%)", linestyle="-", color="green")
    plt.plot(episodes, quartile_3rds, label="3rd Quartile (75%)", linestyle="--", color="red")

    # グラフの装飾
    plt.title("Loss Statistics Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid()

    # グラフを保存
    plt.savefig(filename)
    plt.close()
    print(f"Loss statistics graph saved to {filename}")

def train_and_test_Mario():
    # 環境とエージェントの初期化
    env = MarioEnv(render_mode=None)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                K_epochs=4, eps_clip=0.2, has_continuous_action_space=False)

    # 学習開始
    # train(env, agent, agent_episode=2000, num_episodes=20000, max_timesteps=1000, update_episode=20, file_name="mario")
    env_test = MarioEnv(render_mode="human")
    # テスト
    # test(env_test, agent, agent_episode=22000, num_episodes=10, net_name="mario")
    # アニメーション保存
    env_ani = MarioEnv(render_mode="rgb_array")
    save_animation(env_ani, agent, agent_episode=22000, num_episodes=20, filename="mario", net_name="mario")

def train_and_test_CartPole():
    # 環境とエージェントの初期化
    env = gym.make("CartPole-v1", render_mode=None)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = simpPPO(np.prod(state_dim), action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                K_epochs=4, eps_clip=0.2, has_continuous_action_space=False)

    # 学習開始
    train(env, agent, agent_episode=2000, num_episodes=10, update_episode=10, file_name="CartPole")
    env_test = gym.make("CartPole-v1", render_mode="human")
    # テスト
    # test(env_test, agent, agent_episode=2000, num_episodes=10, net_name="CartPole")
    # アニメーション保存
    env_ani = gym.make("CartPole-v1", render_mode="rgb_array")
    # save_animation(env_ani, agent, agent_episode=2000, filename="CartPole", net_name="CartPole")

def train_and_test_MountainCar():
    # 環境とエージェントの初期化
    env = gym.make("MountainCar-v0", render_mode=None)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = simpPPO(np.prod(state_dim), action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                K_epochs=4, eps_clip=0.2, has_continuous_action_space=False)

    # 学習開始
    train(env, agent, num_episodes=5000, update_episode=20, file_name="MountainCar")
    env_test = gym.make("MountainCar-v0", render_mode="human")
    # テスト
    # test(env_test, agent, agent_episode=2000, num_episodes=10, net_name="MountainCar")
    # アニメーション保存
    env_ani = gym.make("MountainCar-v0", render_mode="rgb_array")
    # save_animation(env_ani, agent, agent_episode=2000, num_episodes=20, filename="MountainCar", net_name="MountainCar")

if __name__ == "__main__":
    train_and_test_Mario()
    # train_and_test_CartPole()
    # train_and_test_MountainCar()
