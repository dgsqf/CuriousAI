from obstacle_tower_env import ObstacleTowerEnv
import CuriosityWrapper
import gym
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt

env = ObstacleTowerEnv(retro=True, realtime_mode=False)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
env.seed(20)
#env = CuriosityWrapper.CuriosityWrapper(env)
env = DummyVecEnv([lambda: env])
try:
    model = PPO2.load("CuriousNet")
except:
    model = PPO2(MlpPolicy, env, verbose=1)
model.set_env(env)
model.learn(total_timesteps=1000000)
print("Finished Training")
model.save('CuriousNet')
obs = env.reset()
