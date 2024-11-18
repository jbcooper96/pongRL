import gymnasium as gym
import ale_py
import argparse
from agent import Agent
from groupFramesWrapper import GroupFramesWrapper
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import os
import sys
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Script to train or test pong agent.')
parser.add_argument('-l', '--load', help='Load saved model', action='store_true')

args = parser.parse_args()
load = args.load

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def run_thread(rank, world_size):
    setup(rank, world_size)

    try:
        gym.register_envs(ale_py)


        agent = Agent(load=load, distibuted=True, device=rank)

        env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")

        wrapped_env = GroupFramesWrapper(env)

        total_num_episodes = int(5e3)

        for episode in range(total_num_episodes):
            wrapped_env.reset()
            state, reward, terminated, truncated, info = wrapped_env.step(1)
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = wrapped_env.step(2 + action, reward=reward)
                done = terminated or truncated
                if train:
                    agent.save_to_buffer(state, action, reward, next_state, done)
                    agent.replay(40)
    finally:
        cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(run_thread,
             args=(world_size,),
             nprocs=world_size,
             join=True)