from utils import plot_examples
import torchattacks
from tqdm import tqdm
import torch.nn as nn
import torch

from attacks import PGD_linf

def test_cw(model, device, dataloader,steps,epsilon,alpha,params):
  '''
    Applies C-W attack on dataloader  and evaluates the robustness of a 
    given model

    Parameters:
        model: the network through which we pass the inputs     
        device : Pytorch device
        dataloader: dataloader containing the original images and the true labels which we aim to perturb to make an adversarial example
        steps : number of steps for CW attack
        epsilon: perturbation budget 
        alpha: step size
        params : other parameters

    Returns: 
        images : adversarial examples
  '''
  attack = torchattacks.CW(model,steps=steps,c=epsilon)
  loss_fn = nn.CrossEntropyLoss()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()


  test_loss = 0
  accuracy = 0

  for X, y in tqdm(dataloader):

      X = X.to(device)
      y = y.to(device)

      test_exemples = X[:params["num_exemples"]].detach().cpu()

      # Attaque
      perturbed_data = attack(X, y)

      if params["noise"]:
        perturbed_data = params["noise"](perturbed_data)            


      if params["denoiser"]:
        perturbed_data = params["denoiser"](perturbed_data)

      pred = model(perturbed_data)
      test_loss += loss_fn(pred, y).item()
      accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
      pert_test_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()

    
  plot_examples(pert_test_exemples)

  test_loss /= num_batches
  accuracy /= size

  print(
      f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
  )

  return accuracy, test_loss

def test_pgd(model, device, dataloader,steps,epsilon,alpha,params):
  '''
    Applies PGD attack on dataloader  and evaluates the robustness of a 
    given model

    Parameters:
        model: the network through which we pass the inputs     
        device : Pytorch device
        dataloader: dataloader containing the original images and the true labels which we aim to perturb to make an adversarial example
        steps : number of steps for CW attack
        epsilon: perturbation budget 
        alpha: step size
        params : other parameters

    Returns: 
        images : adversarial examples
  '''
  loss_fn = nn.CrossEntropyLoss()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()


  test_loss = 0
  correct = 0

  for X, y in tqdm(dataloader):

      X = X.to(device)
      y = y.to(device)

      test_exemples = X[:params["num_exemples"]].detach().cpu()

      # Attaque
      perturbed_data = PGD_linf(model, X, y,device,
                                  epsilon=epsilon,
                              alpha=alpha,
                              n_iter=steps)



      if params["noise"]:
        perturbed_data = params["noise"](perturbed_data)            


      if params["denoiser"]:
        perturbed_data = params["denoiser"](perturbed_data)

      pred = model(perturbed_data)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      pert_test_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()

    
  plot_examples(pert_test_exemples)

  test_loss /= num_batches
  correct /= size

  print(
      f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
  )



def pgd_attack_effect_on_accuracy(model, device, dataloader,params):
    '''
    Applies PGD attack on dataloader and evaluates the robustness of a 
    given model for a set of values of epsilon

    Parameters:
        model: the network through which we pass the inputs     
        device : Pytorch device
        dataloader: dataloader containing the original images and the true labels which we aim to perturb to make an adversarial example
        steps : number of steps for CW attack
        epsilon: perturbation budget 
        alpha: step size
        params : other parameters

    Returns: epsilons, accuracies
    '''
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    epsilons = []
    accuracies = []


    for epsilon in params["epsilons"]:
        test_loss = 0
        correct = 0


        print(f"Epsilon : {epsilon}")
        for X, y in tqdm(dataloader):

            X = X.to(device)
            y = y.to(device)

            test_exemples = X[:params["num_exemples"]].detach().cpu()

            # Attaque
            perturbed_data = PGD_linf(model, X, y,device,
                                        epsilon=epsilon,
                                    alpha=params["alpha"],
                                    n_iter=params["n_iter"])



            if params["noise"]:
                perturbed_data = params["noise"](perturbed_data)            


            if params["denoiser"]:
                perturbed_data = params["denoiser"](perturbed_data)

            pred = model(perturbed_data)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            pert_test_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()

        
        plot_examples(pert_test_exemples)

        test_loss /= num_batches
        correct /= size

        epsilons.append(epsilon)
        accuracies.append(100*correct)
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

        
    return epsilons, accuracies





def deepfool_attack_effect_on_accuracy(model, device, dataloader,params):
  '''
    Applies DeepFool attack on dataloader and evaluates the robustness of a 
    given model for a set of values of epsilon

    Parameters:
        model: the network through which we pass the inputs     
        device : Pytorch device
        dataloader: dataloader containing the original images and the true labels which we aim to perturb to make an adversarial example
        params : other parameters

    Returns: epsilons, accuracies
  '''
  loss_fn = nn.CrossEntropyLoss()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()

  steps = []
  accuracies = []


  for step in params["n_steps"]:
    test_loss = 0
    correct = 0
    attack_deepfool = torchattacks.DeepFool(model, 
                                            steps=step,
                                            overshoot=params["overshoot"])


    print(f"Steps : {step}")
    for X, y in tqdm(dataloader):

        X = X.to(device)
        y = y.to(device)

        test_exemples = X[:params["num_exemples"]].detach().cpu()

        # Attaque
        perturbed_data = attack_deepfool(X, y)


        if params["noise"]:
          perturbed_data = params["noise"](perturbed_data)            


        if params["denoiser"]:
          perturbed_data = params["denoiser"](perturbed_data)

        pred = model(perturbed_data)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        pert_test_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()

    
    plot_examples(pert_test_exemples)

    test_loss /= num_batches
    correct /= size

    steps.append(step)
    accuracies.append(100*correct)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    
  return steps, accuracies