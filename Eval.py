from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
from CuriosityWrapper import CuriosityWrapper
import gym

env = ObstacleTowerEnv(retro=True,realtime_mode=True)
env.seed(20)
env = CuriosityWrapper(env)
model = PPO2.load('CuriousNet')
model.set_env(env)
print(evaluate_policy(model,env))