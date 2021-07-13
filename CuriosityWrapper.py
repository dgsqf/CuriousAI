import gym
class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.stateslist = []
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if not next_state in self.stateslist:
            reward+=0.1
        return next_state, reward, done, info