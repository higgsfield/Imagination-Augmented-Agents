import gym
from gym import spaces

from deepmind import PillEater, observation_as_rgb

class MiniPacman:
    def __init__(self, mode, frame_cap):
        self.mode      = mode
        self.frame_cap = frame_cap
        
        self.env = PillEater(mode=mode, frame_cap=frame_cap)
        
        self.action_space      = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(3, 15, 19))

    def step(self, action):
        self.env.step(action)
        env_reward, env_pcontinue, env_frame = self.env.observation()
        self.done = env_pcontinue != 1
        env_frame = env_frame.transpose(2, 0, 1)
        return env_frame, env_reward, self.done, {}

    def reset(self):
        image, _, _ = self.env.start()
        image = observation_as_rgb(image)
        self.done = False
        image = image.transpose(2, 0, 1)
        return image