from obstacle_tower_env import ObstacleTowerEnv
import CuriosityWrapper
env = ObstacleTowerEnv(retro=True, realtime_mode=True)
env.seed(20)


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER
env=CuriosityWrapper.CuriosityWrapper(env)
env = DummyVecEnv([lambda: env])
try:
    model = ACER.load("CuriousNet")
except:
    print("Train the model before testing it")
model.set_env(env)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(reward)
