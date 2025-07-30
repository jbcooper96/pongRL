import gymnasium as gym
import ale_py
import argparse
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from settings import Settings
import torch
import wandb

parser = argparse.ArgumentParser(description='Script to train or test pong agent.')
parser.add_argument('-l', '--load', help='Load saved model', action='store_true')
parser.add_argument('-r', '--render', help='Render rollout for testing', action='store_true')
parser.add_argument('-v', '--record', help='Record video for testing', action='store_true')
parser.add_argument('-p', '--debugPrint', help='Pause while running and print hidden states', action='store_true')
parser.add_argument('-d', '--device', help="Device to run torch models on")
parser.add_argument('-e', '--env', help="Number of environments")

args = parser.parse_args()
load = args.load
render = args.render
debug = args.debugPrint
if args.device in ["cuda", "cpu", "mps"]:
    Settings.device = torch.device(args.device)

learning_rate = 1e-5
batch_size = 40
epochs = 10
entropy_coef = .01

if not load:
    wandb.init(
        project="Pong ppo",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "entropy": entropy_coef,
            "optimizer": "adamW"
        }
    )


print(Settings.device)
print("CUDA version:", torch.version.cuda)

from models import PModel, ValueModel
from ppo import PPO

gym.register_envs(ale_py)

ENV_NUMBER = 10 if not render else 1

if args.env:
    ENV_NUMBER = int(args.env)

env = make_atari_env("PongNoFrameskip-v4", n_envs=ENV_NUMBER, seed=1)
env = VecFrameStack(env, n_stack=4)

action_model = PModel(6)
value_model = ValueModel()
ppo_agent = PPO(env, action_model, value_model, ENV_NUMBER, load=load, learning_rate=learning_rate, entropy_coef=entropy_coef, epochs=epochs, batch_size=batch_size)
if render:
    ppo_agent.render(debug, debug)
else:
    ppo_agent.run(10000000000, 1500)