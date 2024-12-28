from helpers import get_next_done_vectorized
import numpy as np
import torch
import torch.nn as nn
from ppo_agent import PPOAgent
import math
from settings import Settings
import wandb

OPT_PATH = "opt.pt"
class PPO:
    def __init__(self, env, model, value_model, env_number, batch_size=40, learning_rate=2.5e-4, gamma=.99, lam=.9, clip_coef=.2, entropy_coef=.01, val_coef=1, epochs=3, load=False):
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
            for param_group in self.opt.param_groups:
                param_group['lr'] = learning_rate

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

    def render(self, with_pause=False, debug=False):
        if debug:
            print(self.agent.model.parameters())
            for name, param in self.agent.model.named_parameters():
                if param.requires_grad:
                    print(name, param.data.size())
                    if param.data.ndim > 2:
                        x = torch.flatten(param.data, start_dim=1)
                        print(x.max(dim=1))
                    elif param.data.ndim == 1:
                        print(param.data)
                    else:
                        print(param.data.max(dim=1))
            return
        obs = self.env.reset()
        obs = self.get_torch_obs(obs)
        last_obs = obs
        i = 0
        while True:
            i +=1
            with torch.no_grad():
                actions, _, _ = self.agent.get_action_and_value(obs, do_sample=False)
            obs, reward, dones, info = self.env.step(actions.numpy())
            obs = self.get_torch_obs(obs)
            self.env.render("human")
            if with_pause and (i % 4) == 0:
                c = input("Continue?")

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

    def run(self, total_timestamps, time_stamp_per_it):
        next_obs = self.env.reset()
        next_obs = self.get_torch_obs(next_obs)
        next_done = torch.zeros(self.env_number).to(Settings.device)
        self.episode_rewards_cur = torch.zeros(self.env_number).to(Settings.device)
        self.episode_rewards = []

        for update_number in range(total_timestamps//(time_stamp_per_it*self.env_number)):
            all_obs = torch.zeros(time_stamp_per_it, self.env_number, 4, 84, 84).to(Settings.device)
            all_actions = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            rewards = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            all_values = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            done = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
            log_probs = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)

            
            for step in range(time_stamp_per_it):
                obs = next_obs
                done[step] = next_done
                with torch.no_grad():
                    actions, values, log_probs[step] = self.agent.get_action_and_value(obs)
                next_obs, reward, dones, info = self.env.step(actions.numpy())
                next_obs = self.get_torch_obs(next_obs)
                next_done = torch.tensor(get_next_done_vectorized(dones)).to(Settings.device)
                all_obs[step] = obs
                all_actions[step] = actions
                rewards[step] = torch.tensor(reward)
                all_values[step] = values
                self.log_rewards(rewards[step], dones)
                

            self.learn(all_obs, all_actions, rewards, all_values, log_probs, done, next_obs, next_done, time_stamp_per_it, (update_number%10) == 0, update_number)

                

    def learn(self, all_obs, all_actions, rewards, all_values, all_log_probs, done, next_obs, next_done, time_stamp_per_it, save=False, update_number=0):
        advantages = torch.zeros(time_stamp_per_it, self.env_number).to(Settings.device)
        with torch.no_grad():
            next_values = self.agent.get_next_value(next_obs)
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

        all_obs = torch.flatten(all_obs, start_dim=0, end_dim=1)
        all_actions = torch.flatten(all_actions).int()
        rewards = torch.flatten(rewards)
        all_values = torch.flatten(all_values)
        advantages = torch.flatten(advantages)
        all_log_probs = torch.flatten(all_log_probs)
        returns = torch.flatten(returns)

        batch_count = (time_stamp_per_it*self.env_number) // self.batch_size
        last_batch_size = (time_stamp_per_it*self.env_number) % self.batch_size
        if last_batch_size > 0:
            batch_count += 1

        
        print("Training")
        print(torch.max(rewards))
        nonzero = torch.count_nonzero(rewards)
        print(nonzero)
        avg_reward = torch.sum(rewards)/ nonzero
        print(avg_reward)
        print(returns.mean())
        all_entropy = []
        all_approx_kl = []
        for epoch in range(self.epochs):
            e_inds = torch.randperm(time_stamp_per_it*self.env_number).to(Settings.device)
            total_loss = 0.0
            approx_kl = None
            for batch in range(batch_count):
                cur_batch_size = self.batch_size
                if batch == batch_count - 1 and last_batch_size > 0:
                    cur_batch_size = last_batch_size
                
                start_idx = batch * self.batch_size
                end_idx = batch * self.batch_size + cur_batch_size
                b_inds = e_inds[start_idx:end_idx]

                log_probs, value, entropy = self.agent.get_logprobs_and_value(all_obs.index_select(0, b_inds), all_actions.index_select(0, b_inds))
                log_probs_ratio = log_probs - all_log_probs.index_select(0, b_inds)
                
                ratio = torch.exp(log_probs_ratio)

                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1) - log_probs_ratio)
                
                batch_advantages = advantages.index_select(0, b_inds)
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((value - returns.index_select(0, b_inds)) ** 2).mean()
                mean_entropy = entropy.mean()
                entropy_loss = -mean_entropy
                all_entropy.append(mean_entropy)
                loss = pg_loss + self.val_coef * v_loss + self.entropy_coef * entropy_loss

                if math.isnan(loss):
                    print(log_probs)
                    print(ratio)
                    print(v_loss)
                    print(pg_loss)
                    
                    print("norm adv")
                    print(batch_advantages)
                    print("batch adv")
                    print(advantages.index_select(0, b_inds))
                    import sys
                    sys.exit()


                self.opt.zero_grad()
                loss.backward()
                total_loss += loss.item()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.opt.step()

            print(total_loss/batch_count)
            if approx_kl != None:
                if approx_kl > .02:
                    print("KL too high")
                    break

        all_entropy = torch.tensor(all_entropy)
        avg_entropy = all_entropy.mean().item()

        wandb.log({
            "avg_reward": avg_reward,
            "avg_entropy": avg_entropy,
            "returns": returns.mean().item(),
            "approx_kl": torch.tensor(all_approx_kl).mean().item()
        })
        print(avg_entropy)
        if save:
            print("SAVING")
            self.agent.save(update_number)
            self.save_opt()

            

        
            

