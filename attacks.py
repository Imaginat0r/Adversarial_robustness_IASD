import torch
import torch.nn as nn
from torchattacks import BIM


def PGD_linf(model, images, labels, device, epsilon=0.3, alpha=2/255, n_iter=40) :
    '''
    PGD attack (l_infinity)

    Parameters:
        model: the network through which we pass the inputs      
        images: the original images which we aim to perturb to make an adversarial example
        labels: the true label of the original images
        alpha: step size
        epsilon: perturbation budget 
        n_iter: number of iterations in the PGD algorithm

    Returns: 
        images : adversarial examples
    '''
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(n_iter) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()            

    return images



def fgsm(model, images, labels, device,epsilon=0.3) :
    """FGSM attack

    Parameters:
        model: the network through which we pass the inputs      
        images: the original images which we aim to perturb to make an adversarial example
        labels: the true label of the original images
        device : cpu or cuda
        epsilon: perturbation budget 
    Returns: 
        perturbed_image : adversarial examples
    """
    loss = nn.CrossEntropyLoss()
    
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()
    
    perturbed_image = images + epsilon*images.grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image


def BIM(net,X,Y,epsilon=8/255, alpha=3/255,n_iter = 4):
  """BIM attack"""
  atk = BIM(net, eps=epsilon, alpha=alpha, steps=n_iter)
  X = atk(X, Y)
  return X

def deepfool(X, model, device,num_classes=10, max_iter = 10):
    """DeepFool attack
    Parameters:
        X: the original images which we aim to perturb to make an adversarial example
        model: the network through which we pass the inputs      
        device : cpu or cuda
        num_classes : number of classes
        max_iter: maximum number of iterations
    Returns: 
        perturbed_images : adversarial examples
        
    """
    model.eval()
    list_pertubed_imgs = []
    
    for num,x in enumerate(X):  
      x = x.unsqueeze(0)   
      with torch.no_grad(): 
        f_classifier = model(x)
        r = torch.zeros_like(x)

        # classe initiale prédite par le modèle
        k_x0 = torch.argmax(f_classifier[0]).item()

        classes = list(range(num_classes))
        classes.remove(k_x0)


      x_i = torch.tensor(x, requires_grad=False)


      cpt = 0
      while True:
        l_values = []
        wk_values = []
        wk_values2 = []
        x_i.requires_grad = True

        # Calcul de fk_x0
        f_classifier = model(x_i)
        # Calcul de grad_fk_x0
        selector = torch.zeros_like(f_classifier)
        selector[0][k_x0] = 1
        f_classifier.backward(selector)
        grad_fk_x0 = x_i.grad.data[0]
        fk_x0 = f_classifier[0][k_x0]

        x_i = x_i.detach()


        for i,k in enumerate(classes):
          x_i.requires_grad = True
          f_classifier = model(x_i)

          # Calcul de grad_fk_xi
          selector = torch.zeros_like(f_classifier)
          selector[0][k] = 1
          f_classifier.backward(selector)

          grad_fk_xi = x_i.grad.data[0]
          x_i = x_i.detach()

          # Calcul de fk_xi
          fk_xi = f_classifier[0][k]

          with torch.no_grad():

            #Calcul de wk'
            wk = grad_fk_xi - grad_fk_x0
            wk_values.append(wk)

            #Calcul de fk'          
            fk = fk_xi-fk_x0

            #norme p = 2
            l_values.append(torch.abs(fk)/torch.norm(wk,p=2))
            wk_values2.append(torch.norm(wk,p=2))

        with torch.no_grad():
            l = torch.argmin(torch.tensor(l_values)).item()
            r_i = l_values[l]*wk_values[l]/wk_values2[l]
            x_i = x_i + r_i

            r += r_i

            f_classifier = model(x_i)
            k_xi = torch.argmax(f_classifier[0]).item()

            if k_xi != k_x0:
              break

            cpt += 1

            if cpt > max_iter:
              break
        

      x_r = x + r
      list_pertubed_imgs.append(x_r)

    perturbed_images = torch.cat(list_pertubed_imgs,dim=0)   

    return perturbed_images