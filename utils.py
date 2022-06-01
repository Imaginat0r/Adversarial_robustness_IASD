import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from skimage.util import random_noise


def plot_examples(examples, title="Exemples"):
  """Plot a set of images

    Parameters:
        examples : images to plot   
        title : title of the figure
  """

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
    '''
    Generates noisy images (data augmentation)

    Parameters:
        X: images to add noise on    
        device : cpu ou cuda
        noise_type : type of noise
    Returns: 
        perturbed_data : noisy images
    '''

    X = X.cpu()

    if noise_type == "gaussian":
        perturbed_data = X + 0.05*torch.randn(*X.shape)
        perturbed_data = torch.clamp(perturbed_data,0.,1.)
    elif noise_type == 'salt':
        perturbed_data = random_noise(X, mode=noise_type, amount=0.05)
    elif noise_type == 'speckle':
        perturbed_data = random_noise(X, mode=noise_type, mean=0, var=0.05, clip=True)

    perturbed_data = torch.tensor(perturbed_data).to(device)
    return perturbed_data

def plot_augmentation(original_set, aug_set, n):
    assert len(original_set) == len(aug_set)
    idx = np.random.randint(0, len(original_set), n)
    original_img = [original_set[i][0].numpy().T for i in idx]
    aug_img = [aug_set[i][0].numpy().T for i in idx]

    num_cols = len(original_img)
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_cols):
        ax = plt.subplot(2, num_cols, i + 1)
        ax.imshow(original_img[i])
        if i == 0:
            ax.set_ylabel("Original images")
        ax = plt.subplot(2, num_cols, i + 1 + n)
        ax.imshow(aug_img[i])
        if i == 0:
            ax.set_ylabel("Augmented images")
    plt.tight_layout()
