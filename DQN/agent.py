from collections import deque
from qModel import QModel
import torch
import random
from torch.nn.parallel import DistributedDataParallel as DDP

class Agent:
    def __init__(self, discount_factor=.95, buffer_size=2500, k=1000, load=False, load_file="qModel.pt", save_file ="qModel.pt", epsilon=1, epsilon_decay=.995, epsilon_min=.05, distributed=False, device=torch.device("cpu"), train=True):
        self.discount_factor = discount_factor
        self.buffer = deque(maxlen=buffer_size)
        self.k = k
        self.i = 0
        self.epsilon = epsilon if not load else epsilon_min

        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.save_file = save_file
        self.distributed = distributed
        self.device = device

        self.q_model = QModel().to(device)
        if load:
            self.q_model.load_state_dict(torch.load(load_file, weights_only=True))
        self.q_model.train()

        self.frozen_q_model = QModel().to(device)
        self.frozen_q_model.load_state_dict(self.q_model.state_dict())
        self.frozen_q_model.eval()
        if distributed:
            self.q_model = DDP(self.q_model, device_ids=[device])

        self.optim = torch.optim.AdamW(self.q_model.parameters(), lr=.00001)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 5)
        scores = self.q_model(state.to(self.device))
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
            self.epsilon = self.epsilon if self.epsilon >= self.epsilon_min else self.epsilon_minxw
        return torch.argmax(scores).item()
    
    def save_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        batch_size = batch_size if batch_size <= len(self.buffer) else len(self.buffer)
        """
        sample_weights = []
        cumulative_reward_weight = 0.0
        discount = .4
        for sample in reversed(self.buffer):
            if sample[2] == 0:
                cumulative_reward_weight = cumulative_reward_weight * discount
                sample_weights.insert(0, cumulative_reward_weight + 1)
            else:
                cumulative_reward_weight = 10 * self.epsilon
                sample_weights.insert(0, cumulative_reward_weight + 1)
        sample_weights = torch.tensor(sample_weights)
        samples = torch.multinomial(sample_weights, batch_size)
        """

        samples = random.sample(self.buffer, batch_size)
        avg_loss = 0.0
        avg_reward = 0.0
        avg_loss_non_zero_reward = 0.0
        total_reward = 0.0
        reward_count = 0.0
        loss = torch.tensor(0).to(self.device).to(torch.float)
        for sample in samples:
            state, action, reward, next_state, done = sample
            
            next_state_scores = self.frozen_q_model(next_state.to(self.device))
            y = torch.tensor(reward)
            if not done:
                y = reward + self.discount_factor * torch.max(next_state_scores)

            scores = self.q_model(state.to(self.device))
            loss = torch.nn.functional.mse_loss(scores[action], y.to(self.device))
            if reward != 0:
                total_reward += reward
                reward_count += 1
                avg_loss_non_zero_reward += loss.item()
                loss = loss * 4
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            avg_loss += loss.item()
            

        if reward_count > 0:
            avg_reward = total_reward / reward_count
            avg_loss_non_zero_reward = avg_loss_non_zero_reward / reward_count
        
        self.i += 1
        if self.i % 200 == 0:
            if not self.distributed or self.device == 0:
                print(f"loss:{avg_loss/len(samples)}")
                print(f"reward_loss:{avg_loss_non_zero_reward}")
                print(f"nonzeroreward:{reward_count}")
                print(f"totalreward:{total_reward}")
        if self.i % self.k == 0:
            if self.distributed:
                self.frozen_q_model.load_state_dict(self.q_model.module.state_dict())
                if self.device == 0:
                    torch.save(self.q_model.module.state_dict(), self.save_file)
            else:
                self.frozen_q_model.load_state_dict(self.q_model.state_dict())
                torch.save(self.q_model.state_dict(), self.save_file)
            





