from gym import Wrapper
import torch

class GroupFramesWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action, reward=0):
        print("STEP")
        print(action)
        if reward == 1:
            action = 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_list= [torch.tensor(obs)]
        total_reward = reward
        for _ in range(3):
            if not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
            obs_list.append(torch.tensor(obs))


        return obs, total_reward, terminated, truncated, info
    
    def process_obs(self, obs_list):
        obs = torch.stack(obs_list)
        obs = torch.nn.functional.interpolate(torch.unsqueeze(obs, dim=0), scale_factor=.5, mode="nearest-exact")
        obs = torch.squeeze(obs, dim=0)
        obs = obs[:, 17:97, :] /255.0
        obs = obs.unsqueeze(0)
        print(obs.size())
        return obs



