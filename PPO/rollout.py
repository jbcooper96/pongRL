import gymnasium as gym
import ale_py
import argparse
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from settings import Settings
import torch

parser = argparse.ArgumentParser(description='Script to train or test pong agent.')
parser.add_argument('-t', '--train', help='Is training', action='store_true')
parser.add_argument('-l', '--load', help='Load saved model', action='store_true')
parser.add_argument('-r', '--render', help='Render rollout for testing', action='store_true')
parser.add_argument('-v', '--record', help='Record video for testing', action='store_true')
parser.add_argument('-d', '--device', help="Device to run torch models on")

train = False
args = parser.parse_args()
train = args.train
load = args.load
render = args.render
print(args.device)
if args.device in ["cuda", "cpu", "mps"]:
    Settings.device = torch.device(args.device)


from models import PModel, ValueModel
from ppo import PPO

gym.register_envs(ale_py)

ENV_NUMBER = 6 if not render else 1

env = make_atari_env("PongNoFrameskip-v4", n_envs=ENV_NUMBER, seed=0)
env = VecFrameStack(env, n_stack=4)

action_model = PModel(6)
value_model = ValueModel()
print(load)
ppo_agent = PPO(env, action_model, value_model, ENV_NUMBER, load=load)
if render:
    ppo_agent.render()
else:
    ppo_agent.run(100000000, 1500)