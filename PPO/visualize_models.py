from models import PModel
import torch
import torchvision

MODEL_FILE = "action.pt"
ACTION = 2

model = PModel(6)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))

random_input = torch.rand(1, 4, 84, 84)
random_input = torch.autograd.Variable(random_input, requires_grad=True)
opt = torch.optim.Adam([random_input], lr=.1, weight_decay=1e-6)

for i in range(31):
    opt.zero_grad()
    out = model(random_input)
    loss = -out[0, ACTION]
    loss.backward()
    opt.step()
    print(loss)


output = random_input.squeeze(0)
for i in range(4):
    torchvision.utils.save_image(output[i], f'img_{i}_action{ACTION}.png') 