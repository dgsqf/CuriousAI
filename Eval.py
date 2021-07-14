from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ACER
from CuriosityWrapper import CuriosityWrapper

from stable_baselines.common.vec_env import DummyVecEnv

env = ObstacleTowerEnv(retro=True,realtime_mode=False)
env.seed(20)
env = CuriosityWrapper(env)
env = DummyVecEnv([lambda: env])
model = ACER.load('CuriousNet')
model.set_env(env)
print(evaluate_policy(model,env))