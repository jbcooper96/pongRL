import torch
from settings import Settings


VALUE_FILE = "value_lstm.pt"
ACTION_FILE = "action_lstm.pt"

class PPOAgent:
    def __init__(self, model, value_model):
        self.model = model.to(Settings.device)
        self.value_model = value_model.to(Settings.device)

    def parameters(self):
        return list(self.model.parameters()) + list(self.value_model.parameters())
    
    def get_action_and_value(self, obs, h_value=None, c_value=None, h_action=None, c_action=None, do_sample=True):
        values, states_val = self.value_model(obs, h_value, c_value)
        probs, states = self.model(obs, h_action, c_action)

        batch_size = probs.size(dim=0)
        actions = torch.zeros(batch_size)
        log_probs = torch.zeros(batch_size)

        for action_idx in range(batch_size):
            if do_sample:
                actions[action_idx] = indices = torch.multinomial(probs[action_idx], 1).item()
            else:
                actions[action_idx] = indices = torch.argmax(probs[action_idx]).item()

            log_probs[action_idx] = torch.log(probs[action_idx, indices])

        return actions.int(), torch.squeeze(values, dim=-1), log_probs, states_val, states
        
    def get_next_value(self, obs, h_value=None, c_value=None):
        values, _ = self.value_model(obs, h_value, c_value)
        return torch.squeeze(values, dim=-1)
    
    def get_logprobs_and_value(self, obs, actions, h_value=None, c_value=None, h_action=None, c_action=None):
        values, states_val = self.value_model(obs, h_value, c_value)
        probs, states = self.model(obs, h_action, c_action)
        log_probs = torch.log(probs)
        entropy = -probs * log_probs
        entropy = torch.sum(entropy, dim=1)

        batch_size = actions.size(dim=0)
        log_probs_for_actions = torch.zeros(batch_size).to(Settings.device)
        for batch in range(batch_size):
            log_probs_for_actions[batch] = log_probs[batch, actions[batch].int()]

        return log_probs_for_actions, values, entropy, states_val, states
    
    def save(self):
        torch.save(self.model.state_dict(), ACTION_FILE)
        torch.save(self.value_model.state_dict(), VALUE_FILE)

    def load(self): 
        self.model.load_state_dict(torch.load(ACTION_FILE, weights_only=True))
        self.value_model.load_state_dict(torch.load(VALUE_FILE, weights_only=True))



