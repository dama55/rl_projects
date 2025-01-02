import numpy as np
import torch
from my_ppo.Agents.PPO import PPO
from my_ppo.Envs.MarioEnv import MarioEnv
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import gym
from pathlib import Path

from my_ppo.Util.TimeUtils import get_current_time_string
from my_ppo.Util.FileUtils import exist_or_initialize_dir
from my_ppo.Base.AbstructBase import LoggerBase
from my_ppo.Base.Metric import MetricReward, MetricLength, MetricLoss

basedir = "./data/"

class AgentTrainer():
    """エージェント，環境を受け取って学習を実行する
    """

    def __init__(self, env,
                 agent, 
                 logger, 
                 log_save_dir, 
                 model_save_dir, 
                 model_load_dir):
        self.env = env
        self.agent = agent
        self.logger:LoggerBase = logger
        self.log_save_dir = log_save_dir
        self.model_save_dir = model_save_dir
        self.model_load_dir = model_load_dir
        

    def test(self, num_episodes):
        """
        学習済みエージェントをテストする関数

        Parameters:
        - env: 環境オブジェクト
        - agent: 学習済みエージェント
        - num_episodes: テストするエピソード数
        - render_mode: 環境のレンダリングモード（例: "human"）
        """
        # ログのセーブはしないようにする
        self.logger.set_save_log(False)
        self.agent.set_save_log(False)

        for e in range(num_episodes):

            state, _ = self.env.reset()

            episode_reward = 0

            # Play the game!
            while True:

                # Run agent on the state
                action = self.agent.act(state)

                # Agent performs action
                next_state, reward, done, trunc, info = self.env.step(action)

                self.env.render()

                # Learn. outputs = {loss, q, ....}
                # 中身がない場合は空の辞書型{}として計算
                # outputs = self.agent.update(state, next_state, action, reward, done)

                log_input = dict(
                    step = self.agent.curr_step,
                    reward = reward,
                    done = done,
                    # **outputs
                )
                # Logging
                self.logger.log_step(**log_input)

                # Update state
                state = next_state
                episode_reward += reward

                # Check if end of game
                if done: #or info["flag_get"]:
                    break

            self.logger.log_episode()

            if (e % 20 == 0) or (e == num_episodes - 1):
                self.logger.record(episode=e,  step=self.agent.curr_step)

            print(f"episode: {e+1:3.0f} reward: {episode_reward:5.0f}")

    def save_animation(self, num_episodes, load_model_dir, num_chkpt, file_name = "mario"):
        """
        学習済みエージェントの動作をGIFとして保存する関数

        Parameters:
        - env: 環境オブジェクト
        - agent: 学習済みエージェント
        - filename: 保存するGIFのファイル名
        - max_timesteps: GIFに含める最大タイムステップ数
        """
        episodes = []
        rewards = []

        # ログのセーブはしないようにする
        self.logger.set_save_log(False)
        self.agent.set_save_log(False)

        for e in range(num_episodes):

            state, _ = self.env.reset()
            episode_reward = 0  # エピソードの累積報酬
            frames = []  # アニメーションのフレームを保存するリスト

            # Play the game!
            while True:

                # Run agent on the state
                action = self.agent.act(state)

                # Agent performs action
                next_state, reward, done, trunc, info = self.env.step(action)

                # フレームを取得
                frame = self.env.render()  # 古いバージョンのAPI
                frames.append(frame.copy())


                # Learn. outputs = {loss, q, ....}
                # 中身がない場合は空の辞書型{}として計算
                # outputs = self.agent.update(state, next_state, action, reward, done)

                log_input = dict(
                    step = self.agent.curr_step,
                    reward = reward,
                    done = done,
                    # **outputs
                )
                # Logging
                self.logger.log_step(**log_input)

                # Update state
                state = next_state
                episode_reward += reward

                # Check if end of game
                if done: #or info["flag_get"]:
                    break

            self.logger.log_episode()
            episodes.append(frames)
            rewards.append(episode_reward)

            if (e % 20 == 0) or (e == num_episodes - 1):
                self.logger.record(episode=e,  step=self.agent.curr_step)

            print(f"episode: {e+1:3.0f} reward: {episode_reward:5.0f}")

        best_frames = episodes[np.argmax(rewards)]

        print("rewards list: ", rewards)
        print("best reward: ", np.max(rewards))

        # フレームを Pillow イメージに変換
        frames_pil = [Image.fromarray(frame) for frame in best_frames]

        exist_or_initialize_dir(load_model_dir)

        # GIF を保存
        frames_pil[0].save(
            load_model_dir / f"{file_name}_chkpt_{num_chkpt}.gif",
            save_all=True,          # すべてのフレームを保存
            append_images=frames_pil[1:],  # 他のフレームを追加
            duration=33,            # フレーム間隔（ミリ秒単位、fps=30なら約33ms）
            loop=0                  # 無限ループ
        )
        print("Animation saved as animation.gif")



    def train(self, num_episodes=1000):
        """
        環境とエージェントを使って強化学習を行う関数

        Parameters:
        - env: 環境オブジェクト（例: OpenAI Gym 環境）
        - agent: 強化学習エージェント（例: PPOクラスのインスタンス）
        - num_episodes: 学習するエピソード数（デフォルト: 1000）
        - max_timesteps: 1エピソード内の最大タイムステップ数（デフォルト: 500）
        - update_timestep: エージェントを更新するタイミングとなる総タイムステップ数（デフォルト: 2000）
        """
        
        for e in range(num_episodes):

            state, _ = self.env.reset()

            # Play the game!
            while True:

                # Run agent on the state
                action = self.agent.act(state)

                # Agent performs action
                next_state, reward, done, trunc, info = self.env.step(action)


                # Learn. outputs = {loss, q, ....}
                # 中身がない場合は空の辞書型{}として計算
                outputs = self.agent.update(state, next_state, action, reward, done)

                log_input = dict(
                    step = self.agent.curr_step,
                    reward = reward,
                    done = done,
                    **outputs
                )
                # Logging
                self.logger.log_step(**log_input)

                # Update state
                state = next_state

                # Check if end of game
                if done: # or info["flag_get"]:
                    break

            self.logger.log_episode()

            if (e % 20 == 0) or (e == num_episodes - 1):
                self.logger.record(episode=e,  step=self.agent.curr_step)

            
        # 最後にモデルの保存
        self.agent.save_model()
            