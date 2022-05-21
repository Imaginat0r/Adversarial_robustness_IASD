import matplotlib.pyplot as plt
import random
import torch
from skimage.util import random_noise


def plot_examples(examples, title="Exemples"):
  """Plot a set of images"""
  plt.figure(figsize=(15,8))
  num_examples = len(examples)
  cnt = 0
  for j in range(num_examples):
    cnt += 1
    plt.subplot(1,num_examples,cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    ex = examples[j]
    plt.imshow(ex.transpose(2,0))
    plt.title(title)
  plt.show()


def generate_noisy_data(X, device,noise_type="gaussian"):  
    X = X.cpu()

    if noise_type == "gaussian":
        perturbed_data = X + 0.05*torch.randn(*X.shape)
        perturbed_data = torch.clamp(perturbed_data,0.,1.)
    elif noise_type == 'salt':
        perturbed_data = random_noise(X, mode=noise_type, amount=0.05)
    elif noise_type == 'speckle':
        perturbed_data = random_noise(X, mode=noise_type, mean=0, var=0.05, clip=True)

    return torch.tensor(perturbed_data).to(device)
