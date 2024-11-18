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

train = False
args = parser.parse_args()
train = args.train
load = args.load
gym.register_envs(ale_py)

learning_rate = .0001

record_video = True
if train:
    record_video = False

agent = Agent(load=load)

env = gym.make('PongNoFrameskip-v4', render_mode="rgb_array", obs_type="grayscale")

if record_video:
    wrapped_env = gym.wrappers.RecordVideo(env, video_folder="recordings", episode_trigger= lambda x: True, disable_logger=True)
    wrapped_env = GroupFramesWrapper(wrapped_env)
else:
    wrapped_env = GroupFramesWrapper(env)

total_num_episodes = int(5e5)

    
"""
wrapped_env.reset()
obs, reward, terminated, truncated, info = wrapped_env.step(1)

for _ in range(50):
    obs, reward, terminated, truncated, info = wrapped_env.step(1)
print(obs.size())
test_image = obs[3] / 255.0

# Save the image
save_image(obs[0] / 255.0, 'test1.png')
save_image(obs[1] / 255.0, 'test2.png')
save_image(obs[2] / 255.0, 'test3.png')
save_image(obs[3] / 255.0, 'test4.png')
"""

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

