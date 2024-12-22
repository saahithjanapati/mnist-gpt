import torch
import torch.backends
from mnist_gpt import *
import matplotlib.pyplot as plt


PATH_TO_STATE_DICT = "./trained_model_state_dict.pt"


def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device



model = GPT(GPTConfig())

state_dict = torch.load(PATH_TO_STATE_DICT, map_location='cpu')
model.load_state_dict(state_dict)

device = get_best_device()
model.to(device)

num_parameters = model.get_num_params()
print(f"Model has {num_parameters} parameters")


print("Generating!!!!...")
indices = model.generate(5)

indices_reshaped = indices[1:-1].cpu().numpy().reshape(28, 28)
plt.imshow(indices_reshaped, cmap="viridis")
plt.colorbar()
plt.show()