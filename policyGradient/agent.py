import torch
from policy import Policy

DISCOUNT_FACTOR = .9
class Agent:
    def __init__(self, learning_rate=.001, training=True, save_file="policy.pt"):
        self.training = training
        self.save_file = save_file

        self.model = Policy()
        self.obs_list = []
        self.reward_list = []
        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, maximize=True, weight_decay=.001)
        if training:
            self.model.train()

    def get_action(self, obs, reward):
        self.obs_list.append(torch.tensor(obs).to(torch.float)/255.0)
        should_return_special_action, action = self.should_return_special_action(reward)
        if should_return_special_action:
            return action
        
        
        probs = self.model(self.obs_list[-1] - self.obs_list[-2])
        if len(self.reward_list) > 0:
            self.reward_list[-1] = (self.reward_list[-1][0], reward)

        max_index = torch.argmax(probs)
        if self.training:
            max_index = torch.multinomial(probs, 1)[0]
        self.reward_list.append((torch.log(probs[max_index]), 0))
        #print(up_prob)
        self.obs_list = self.obs_list[-2:]
        return 2 if max_index == 0 else 3
        
        
    def backward(self, reward_final):
        if len(self.reward_list) == 0:
            print("reward 0")
            return
        
        self.reward_list[-1] = (self.reward_list[-1][0], reward_final)

        gradient = 0.
        positive_reward_count = 0.0
        total_reward_count = 0.0
        for i in range(len(self.reward_list) - 1, -1, -1):
            if self.reward_list[i][1] == 1:
                reward_final = 1
                positive_reward_count += 1
                total_reward_count += 1
            elif self.reward_list[i][1] == -1:
                reward_final = -1
                total_reward_count += 1
            gradient += reward_final * self.reward_list[i][0]

        gradient = gradient / len(self.reward_list)
        print("backwards")
        print(gradient)
        if total_reward_count > 0:
            print(positive_reward_count/total_reward_count)
        self.opt.zero_grad()
        gradient.backward()
        self.opt.step()

        self.save()
        self.obs_list = []
        self.reward_list = []

    def should_return_special_action(self, reward):
        if reward == 1:
            return True, 1 #return fire if won last round
        if len(self.obs_list) < 2:
            return True, 0 # return NOOP if not enough obs
        
        return False, 0
    
    def load(self, file):
        self.model.load_state_dict(torch.load(file, weights_only=True))

    def save(self):
        torch.save(self.model.state_dict(), self.save_file)