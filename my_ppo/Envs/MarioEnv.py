import gym
print(f"Gym version: {gym.__version__}")
from gym.wrappers import FrameStack, TimeLimit
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
import numpy as np
import torch
from torchvision import transforms as T


import gym_super_mario_bros

class MarioEnv(gym.Env):
    """
    OpenAI Gym の抽象クラス `gym.Env` を継承したマリオ環境。
    """
    metadata = {"render_modes": ["human", "rgb", "rgb_array"]}


    def __init__(self, 
                 version="SuperMarioBros-1-1-v0", 
                 render_mode="rgb", 
                 skip_frames=4, 
                 shape=84, 
                 num_stack=4,
                 max_timesteps = None):
        """
        マリオ環境の初期化

        Parameters:
        - version: 環境のバージョン (例: 'SuperMarioBros-1-1-v0')
        - render_mode: レンダリングモード (例: 'rgb' or 'human')
        - skip_frames: スキップするフレーム数
        - shape: リサイズ後の観測サイズ (例: 84)
        - num_stack: フレームスタック数
        """
        super().__init__()

        # 環境の初期化
        self.version = version
        self.render_mode = render_mode
        
        if gym.__version__ < '0.26':
            self.env = gym_super_mario_bros.make(version, new_step_api=True)
        else:
            self.env = gym_super_mario_bros.make(version, render_mode=render_mode, apply_api_compatibility=True)

        # 行動空間を制限
        self.env = JoypadSpace(self.env, [["right"], ["right", "A"]])

        # フレームスキップと前処理
        self.env = SkipFrame(self.env, skip_frames)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape)
        self.env = FrameStack(self.env, num_stack=num_stack)

        if max_timesteps:
            self.env = TimeLimit(self.env, max_episode_steps=max_timesteps)

        # Action Space と Observation Space を設定
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """環境をリセット"""
        return self.env.reset()

    def step(self, action):
        """環境に対する行動を実行"""
        return self.env.step(action)

    def render(self, mode=None):
        """環境をレンダリング"""
        return self.env.render()

    def close(self):
        """環境を終了"""
        self.env.close()

# フレームスキップのラッパー
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info

# グレースケール変換のラッパー
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # [H, W, C] -> [H, W]初めの二つを取り出し
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        # 形式変換
        observation = self.permute_orientation(observation)
        # グレースケール変換関数
        transform = T.Grayscale()
        # グレースケール変換
        observation = transform(observation)
        return observation
    

# リサイズのラッパー
class ResizeObservation(gym.ObservationWrapper):
    #downsamples each observation into a square image. New size: [1, 84, 84]
    def __init__(self, env, shape):
        super().__init__(env)
        # 画像幅shapeがintかどうかによって画像のサイズを再定義
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        # 三次元目以降があるならそれをつなげる
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Composeでトランスフォームをまとめることが可能
        transforms = T.Compose(
            # T.Resize(self.shape, antialias=True)
            # リサイズ．antialiasでエイリアスを防ぐ
            # T.Normalize(0, 255)
            # -1,1で正規化
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        # 0次元のサイズが1なら消す
        return transforms(observation).squeeze(0)

# 0-1正規化のラッパー
class DynamicNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space

        # 観測空間の低値と高値を利用して正規化範囲を設定
        self.obs_min = obs_space.low
        self.obs_max = obs_space.high
        # print("obs_min: ", self.obs_min )
        # print("obs_max: ", self.obs_max )

        # 観測値の空間を正規化
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # 動的な最大値と最小値に基づいて正規化
        print("動的な obs_min: ", np.min(observation))
        print("動的な obs_max: ", np.max(observation))
        norm_obs = (observation - self.obs_min) / (self.obs_max - self.obs_min + 1e-7)
        return norm_obs.astype(np.float32)
    
    

if __name__ == "__main__":
    mario_env = MarioEnv(render_mode="human")

    # 環境のリセット
    state, _ = mario_env.reset()
    print(f"Initial state shape: {state.shape}")

    done = False
    while not done:
        # ランダムな行動を選択
        action = mario_env.env.action_space.sample()
        next_state, reward, done, trunc, info = mario_env.step(action)
        mario_env.render() # 画面を表示

        state = np.array(next_state)
        # print(f"Reward: {reward}, Done: {done}, Info: {info}")
        # 空間の範囲
        # print("Unique low values:", np.min(state))
        # print("Unique high values:", np.max(state))

        # print(f"State type: {type(state)}")
        # print(f"State shape: {state.shape if isinstance(state, np.ndarray) else 'Not an ndarray'}")
        # print(f"State data type: {state.dtype if isinstance(state, np.ndarray) else 'Unknown'}")

    # 環境を閉じる
    mario_env.close()