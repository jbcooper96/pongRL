from helpers import get_next_done_vectorized
import numpy as np
import torch
import torch.nn as nn
from ppo_agent_lstm import PPOAgent
import math
from models_lstm import HIDDEN_SIZE
from settings import Settings
import wandb

OPT_PATH = "opt.pt"
class PPO:
    def __init__(self, env, model, value_model, env_number, batch_size=40, learning_rate=2.5e-4, gamma=.99, lam=.9, clip_coef=.2, entropy_coef=0, val_coef=1, epochs=3, load=False):
        self.env = env
        self.model = model
        self.value_model = value_model
        self.agent = PPOAgent(model, value_model)
        if load:
            self.agent.load()
        self.env_number = env_number
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.opt = torch.optim.AdamW(self.agent.parameters(), learning_rate)
        if load:
            self.opt.load_state_dict(torch.load(OPT_PATH, weights_only=True))
        self.max_grad_norm = 0.5
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_coef = val_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate

    def get_torch_obs(self, obs):
        obs = torch.tensor(obs) /255.0
        return torch.transpose(obs, 1, 3).to(Settings.device)
    
    def save_opt(self):
        torch.save(self.opt.state_dict(), OPT_PATH)

    def log_rewards(self, rewards, dones):
        self.episode_rewards_cur += rewards
        for i in range(self.env_number):
            if dones[i]:
                self.episode_rewards.append(self.episode_rewards_cur[i])
                self.episode_rewards_cur[i] = 0

        if len(self.episode_rewards) >= self.env_number:
            avg_rewards = torch.tensor(self.episode_rewards).mean().item()
            self.episode_rewards = []
            wandb.log({"avg_reward_per_episode": avg_rewards})
            print("avg reward per ep", avg_rewards)

    def get_hidden_states(self, next_done, h_val, c_val, h_act, c_act):
        for i in range(next_done.size()[0]):
            if next_done[i] == 1:
                h_val[0, i] = torch.zeros(HIDDEN_SIZE)
                c_val[0, i] = torch.zeros(HIDDEN_SIZE)
                h_act[0, i] = torch.zeros(HIDDEN_SIZE)
                c_act[0, i] = torch.zeros(HIDDEN_SIZE)

        return h_val, c_val, h_act, c_act

    def render(self):
        obs = self.env.reset()
        obs = self.get_torch_obs(obs)
        while True:
            with torch.no_grad():
                actions, _, _ = self.agent.get_action_and_value(obs)
            obs, reward, dones, info = self.env.step(actions.numpy())
            obs = self.get_torch_obs(obs)
            self.env.render("human")

    def run(self, total_timestamps, time_stamp_per_it):
        next_obs = self.env.reset()
        next_obs = self.get_torch_obs(next_obs)
        next_done = torch.zeros(self.env_number).to(Settings.device)
        h_value = None
        c_value = None
        h_action = None
        c_action = None

        for update_number in range(total_timestamps//(time_stamp_per_it*self.env_number)):
            all_obs = torch.zeros(time_stamp_per_it, self.env_number, 4, 84, 84).to(Settings.device)
            all_actions = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            rewards = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            all_values = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            done = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            log_probs = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            init_h_value = h_value
            init_c_value = c_value
            init_h_action = h_action
            init_c_action = c_action

            
            for step in range(time_stamp_per_it):
                obs = next_obs
                done[step] = next_done
                with torch.no_grad():
                    actions, values, log_probs[step], (h_value, c_value), (h_action, c_action) = self.agent.get_action_and_value(obs, h_value, c_value, h_action, c_action)
                next_obs, reward, dones, info = self.env.step(actions.numpy())
                next_obs = self.get_torch_obs(next_obs)
                next_done = torch.tensor(get_next_done_vectorized(dones)).to(Settings.device)
                all_obs[step] = obs
                all_actions[step] = actions
                rewards[step] = torch.tensor(reward)
                all_values[step] = values

                h_value, c_value, h_action, c_action = self.get_hidden_states(next_done, h_value, c_value, h_action, c_action)
                self.log_rewards(rewards[step], dones)

                
            states = [h_value, c_value, h_action, c_action, init_h_value, init_c_value, init_h_action, init_c_action]
            self.learn(all_obs, all_actions, rewards, all_values, log_probs, done, next_obs, next_done, time_stamp_per_it, states, save=(update_number%10) == 0)

                

    def learn(self, all_obs, all_actions, rewards, all_values, all_log_probs, done, next_obs, next_done, time_stamp_per_it, states, save=False):
        [h_value, c_value, h_action, c_action, init_h_value, init_c_value, init_h_action, init_c_action] = states

        advantages = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
        with torch.no_grad():
            next_values = self.agent.get_next_value(next_obs, h_value, c_value)
        last_advantage_lam = 0.0
        for t in reversed(range(time_stamp_per_it)):
            if t == time_stamp_per_it - 1:
                next_not_done = 1.0 - next_done
            else:
                next_not_done = 1.0 - done[t+1]
                next_values = all_values[t+1]

            delta = rewards[t] + self.gamma * next_values * next_not_done - all_values[t]
            advantages[t] = last_advantage_lam = delta + self.gamma * self.lam * next_not_done * last_advantage_lam

        returns = advantages + all_values

        batch_count = (time_stamp_per_it*self.env_number) // self.batch_size
        last_batch_size = (time_stamp_per_it*self.env_number) % self.batch_size
        if last_batch_size > 0:
            batch_count += 1

        
        print("Training")
        print(torch.max(rewards))
        nonzero = torch.count_nonzero(rewards)
        print(nonzero)
        print(torch.sum(rewards)/ nonzero)
        print(returns.mean())
        for epoch in range(self.epochs):
            last_i = 0
            for i in range(time_stamp_per_it):
                cur_next_dones = next_done if i == (time_stamp_per_it - 1) else done[i + 1]
                log_probs, value, entropy, (init_h_value, init_c_value), (init_h_action, init_c_action) = self.agent.get_logprobs_and_value(all_obs[i], all_actions[i], init_h_value, init_c_value, init_h_action, init_c_action)
                log_probs_ratio = log_probs - all_log_probs[i]
                
                ratio = torch.exp(log_probs_ratio)
                
                batch_advantages = advantages[i]
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - returns[i]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + self.val_coef * v_loss + self.entropy_coef * entropy_loss
                loss /= self.batch_size
                loss.backward()

                init_h_value, init_c_value, init_h_action, init_c_action = self.get_hidden_states(cur_next_dones, init_h_value.detach(), init_c_value.detach(), init_h_action.detach(), init_c_action.detach())
                if math.isnan(loss):
                    print(log_probs)
                    print(ratio)
                    print(v_loss)
                    print(pg_loss)
                    
                    print("norm adv")
                    print(batch_advantages)
                    print("batch adv")
                    print(advantages.index_select[i])
                    import sys
                    sys.exit()


                if ((i + 1) % self.batch_size) == 0:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                    last_i = i + 1

            if last_i < time_stamp_per_it:
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()


        if save:
            print("SAVING")
            self.agent.save()
            self.save_opt()

            

        
            

