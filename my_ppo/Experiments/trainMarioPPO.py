import numpy as np
import torch
from my_ppo.Agents.PPO import PPO
from my_ppo.Agents.SimplePPO import PPO as simpPPO
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
from my_ppo.Trainers.Trainer import AgentTrainer

ID = "001"
LoadID = "001"

def train_and_test_Mario():
    
    ########################################################
    # パラメータ
    num_chkpt = 141 # ロードするチェックポイント数
    num_episodes = 10 # 繰り返すエピソード数
    render_mode = "human"#None, human, rgb_mode
    ########################################################
    # 環境の初期化
    env = MarioEnv(render_mode=render_mode) 
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    ########################################################
    # ディレクトリの初期化
    base_dir = Path(__file__).resolve().parent.parent  # my_ppoディレクトリ
    save_log_dir = base_dir / Path("logs") /  get_current_time_string()
    save_model_dir = None
    load_model_dir = None
    if ID and ID != "":
        save_model_dir = base_dir / Path("checkpoints") / "MarioPPO" / ID
    if LoadID and LoadID !="":
        load_model_dir = base_dir / Path("checkpoints") / "MarioPPO" / LoadID

    ########################################################
    # エージェントとロガーの初期化
    agent = PPO(state_dim, 
                action_dim,
                save_dir=save_model_dir,
                load_dir=load_model_dir,
                save_every=100,
                learn_every=3,
                lr_actor=0.0003, 
                lr_critic=0.001, 
                gamma=0.99, 
                K_epochs=4, 
                eps_clip=0.2, 
                has_continuous_action_space=False)
    

    ########################################################

    agent.load_model(num_chkpt=num_chkpt)
    
    # ロガーの初期化
    logger = LoggerBase.RoggerBuilder(save_log_dir)
    logger = logger.add(MetricLength("Length")). \
        add(MetricReward("Reward")). \
        add(MetricLoss("Loss")).build()
    

    # トレイナーの初期化
    trainer = AgentTrainer(env,
                           agent,
                           logger,
                           log_save_dir=save_log_dir,
                           model_save_dir=save_model_dir,
                           model_load_dir=load_model_dir)
    
    # trainer.train(num_episodes=num_episodes)]
    trainer.test(num_episodes=num_episodes)
    # trainer.save_animation(num_episodes=num_episodes, 
    #                        load_model_dir=load_model_dir,
    #                        num_chkpt=num_chkpt,
    #                        file_name="mario")


if __name__ == "__main__":
    train_and_test_Mario()