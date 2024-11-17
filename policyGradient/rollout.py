import gymnasium as gym
import ale_py
import argparse
from agent import Agent

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

agent = Agent(learning_rate=learning_rate, training=train)

env = gym.make('ALE/Pong-v5', render_mode="rgb_array", obs_type="grayscale")

if record_video:
    wrapped_env = gym.wrappers.RecordVideo(env, video_folder="recordings", episode_trigger= lambda x: True, disable_logger=True)
else:
    wrapped_env = env

total_num_episodes = int(5e3)

if load:
    agent.load("policy.pt")
    


for episode in range(total_num_episodes):
    obs, info = wrapped_env.reset()
    done = False
    reward = 0
    while not done:
        action = agent.get_action(obs, reward)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated

    print("episode end")
    if train:
        agent.backward(reward)
