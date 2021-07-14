from obstacle_tower_env import ObstacleTowerEnv
import CuriosityWrapper


env = ObstacleTowerEnv(retro=True, realtime_mode=False)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER
env.seed(20)
#env = CuriosityWrapper.CuriosityWrapper(env)
env = DummyVecEnv([lambda: env])
try:
    model = ACER.load("CuriousNet")
except:
    model = ACER(MlpPolicy, env, verbose=1,n_steps=10)
model.set_env(env)
model.learn(total_timesteps=1000000)
print("Finished Training")
model.save('CuriousNet')
obs = env.reset()
