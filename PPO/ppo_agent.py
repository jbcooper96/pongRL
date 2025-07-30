import torch
from settings import Settings


VALUE_FILE = "value.pt"
ACTION_FILE = "action.pt"
SAVE_VALUE_FILE = "value.pt"
SAVE_ACTION_FILE = "action.pt"

class PPOAgent:
    def __init__(self, model, value_model):
        self.model = model.to(Settings.device)
        self.value_model = value_model.to(Settings.device)

    def parameters(self):
        return list(self.model.parameters()) + list(self.value_model.parameters())
    
    def get_action_and_value(self, obs, do_sample=True, print_probs=False):
        values = self.value_model(obs)
        probs = self.model(obs)

        if print_probs:
            print(probs)
            print(values)
        #probs = probs.softmax(dim=1)
        batch_size = probs.size(dim=0)
        actions = torch.zeros(batch_size)
        log_probs = torch.zeros(batch_size)

        for action_idx in range(batch_size):
            if do_sample:
                actions[action_idx] = indices = torch.multinomial(probs[action_idx], 1).item()
            else:
                actions[action_idx] = indices = torch.argmax(probs[action_idx]).item()

            log_probs[action_idx] = torch.log(probs[action_idx, indices] + 1e-8)

        return actions.int(), torch.squeeze(values, dim=-1), log_probs
        
    def get_next_value(self, obs):
        values = self.value_model(obs)
        return torch.squeeze(values, dim=-1)
    
    def get_logprobs_and_value(self, obs, actions):
        values = self.value_model(obs)
        probs = self.model(obs)
        log_probs = torch.log(probs + 1e-8)
        entropy = -probs * log_probs
        entropy = torch.sum(entropy, dim=1)

        batch_size = actions.size(dim=0)
        log_probs_for_actions = torch.zeros(batch_size).to(Settings.device)
        for batch in range(batch_size):
            log_probs_for_actions[batch] = log_probs[batch, actions[batch]]

        return log_probs_for_actions, values, entropy
    
    def save(self, update_number=None):
        action_file, value_file = (SAVE_ACTION_FILE, SAVE_VALUE_FILE)

        #action_file += str(update_number)
        #value_file += str(update_number)
        torch.save(self.model.state_dict(), action_file)
        torch.save(self.value_model.state_dict(), value_file)

    def load(self): 
        self.model.load_state_dict(torch.load(ACTION_FILE, weights_only=True, map_location=Settings.device))
        self.value_model.load_state_dict(torch.load(VALUE_FILE, weights_only=True, map_location=Settings.device))



