import gymnasium as gym
import ale_py
import argparse
from agent import Agent
from groupFramesWrapper import GroupFramesWrapper
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch

parser = argparse.ArgumentParser(description='Script to train or test pong agent.')
parser.add_argument('-t', '--train', help='Is training', action='store_true')
parser.add_argument('-l', '--load', help='Load saved model', action='store_true')
parser.add_argument('-r', '--render', help='Render rollout for testing', action='store_true')
parser.add_argument('-v', '--record', help='Record video for testing', action='store_true')

train = False
args = parser.parse_args()
train = args.train
load = args.load
render = args.render
gym.register_envs(ale_py)

learning_rate = .0001

record_video = False
if train:
    record_video = args.record

agent = Agent(load=load)

env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")
if render:
    env = gym.make('PongNoFrameskip-v4', render_mode="human", obs_type="grayscale")

if record_video:
    wrapped_env = gym.wrappers.RecordVideo(env, video_folder="recordings", episode_trigger= lambda x: True, disable_logger=True)
    wrapped_env = GroupFramesWrapper(wrapped_env)
else:
    wrapped_env = GroupFramesWrapper(env)

total_num_episodes = int(5e5)



for episode in range(total_num_episodes):
    wrapped_env.reset()
    if render:
        wrapped_env.render()
    state, reward, terminated, truncated, info = wrapped_env.step(1)
    done = False
    score = 0 
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = wrapped_env.step(action, reward=reward)
        score += reward
        done = terminated or truncated
        if train:
            agent.save_to_buffer(state, action, reward, next_state, done)
            agent.replay(40)

    print(f"score:{score}")

