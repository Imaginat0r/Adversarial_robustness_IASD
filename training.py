import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from attacks import PGD_linf
import random
import torchattacks
from torchvision import datasets, transforms
from utils import plot_examples, generate_noisy_data

def test_model(dataloader,device, model, loss_fn):
    '''
    Evaluate the model on the dataloader with 
    respect to a given loss function

    Parameters:
        dataloader: Pytorch dataloader
        model : nn.Module Pytorch object
        loss_fn : loss function 

    Returns: accuracy , loss
    '''

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    accuracy /= size

    print(
        f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}% \n"
    )

    return accuracy, loss


def train_model(model,optimizer,loss_fn,batch_data,targets):
      '''
      Train the model on the dataloader with 
      respect to a given loss function

      Parameters:
          model : nn.Module Pytorch object
          optimizer : model optimizer
          loss_fn : loss function 
          batch_data : a batch of input data
          targets : targets

      Returns: 
          model_output : the output of the model
          loss : value of loss function
      '''
      optimizer.zero_grad()
      model_output = model(batch_data)
      loss = loss_fn(model_output,targets)            
      loss.backward()
      optimizer.step()
      return model_output, loss.item()

def train_autoencoder(model,optimizer,loss_fn,batch_data,targets,train=True):
      '''
      Train the DDSA* model on the dataloader with 
      respect to a given loss function

      *DDSA: A Defense Against Adversarial Attacks Using Deep 
      Denoising Sparse Autoencoder
                  
      Parameters:
          model : nn.Module Pytorch object
          optimizer : model optimizer
          loss_fn : loss function 
          batch_data : a batch of input data
          targets : targets
          train(bool) : if True, training mode; if False, inference mode

      Returns: 
          model_output : the output of the model
          loss : value of loss function
      '''
      if train :
        model.train()
        optimizer.zero_grad()
        out = model(batch_data)
        loss = loss_fn(out,targets) 

        l1_norm = 0
        l1_lambda = 0.000001
        l1_norm += sum(p.abs().sum()
                    for p in model.linear1.parameters())        
        l1_norm += sum(p.abs().sum()
                    for p in model.linear2.parameters())        
        l1_norm += sum(p.abs().sum()
                    for p in model.linear3.parameters())

        loss = loss + l1_lambda*l1_norm

        loss.backward()
        optimizer.step()
      else:
        model.eval()
        out = model(batch_data)
        loss = loss_fn(out,targets) 


      return out, loss.item()



def training(model,device,trainloader,testloader, params):
    '''
    Trains on trainloader and evaluates the model on testloader

    Returns: 
        images : adversarial examples
    '''
    test_accuracies = []
    test_losses = []

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=params["params_model"]["lr"])
    
    for epoch in range(params["epochs"]):
        model.train()

        for batch, (X, y) in enumerate(tqdm(trainloader)):               
            X = X.to(device)
            y = y.to(device)           

            # Ajout d'un bruit gaussien
            X_noise = X + 0.01*torch.randn(*X.shape).to(device)
            X_noise = torch.clamp(X_noise,0.,1.)  

            # Train
            train_model(model,optimizer,loss_fn,
                        batch_data=X_noise,
                        targets=y)
            
            train_model(model,optimizer,loss_fn,
                        batch_data=X,
                        targets=y)

        #### Test 
        model.eval()
        size = len(testloader.dataset)
        num_batches = len(testloader)
        test_loss, accuracy = 0, 0


        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)


            # Prédiction
            preds = model(X)             

            test_loss += loss_fn(preds, y).item()
            accuracy += (preds.argmax(1) == y).type(torch.float).sum().item()     
        
        test_loss /= num_batches
        accuracy /= size
        print(
            f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        print("-----------------------------------------------------")

        test_accuracies.append(accuracy)
        test_losses.append(test_loss)

    return test_accuracies, test_losses




def adversarial_training(model,device,trainloader,testloader, params):
  loss_fn = nn.CrossEntropyLoss()
  # optimizer = optim.Adam(model.parameters(),
  #                          lr=params["params_model"]["lr"])
  
  optimizer = optim.SGD(model.parameters(), lr=params["params_model"]["lr"],
                      momentum=0.9, weight_decay=5e-4)
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

  test_losses = []
  accuracies = []

  attack_cw = torchattacks.CW(model,c=1,steps=20)

  max_score = 32

  for epoch in range(params["epochs"]):
      model.train()

      for batch, (X, y) in enumerate(tqdm(trainloader)):               
        X = X.to(device)
        y = y.to(device)   


        # Image attaquée par PGD
        attacks = random.sample([1,2], 2)

        # perturbed_data = X

        if 1 in attacks or 5 in attacks or 7 in attacks or 8 in attacks or 9 in attacks:
          perturbed_data = PGD_linf(model, X, y,device,
                                    alpha=params["alpha"],
                                    epsilon=params["epsilon"],
                                    n_iter=params["n_iter"])
        if 2 in attacks:
          # perturbed_data = attack_deepfool(X,y)
          perturbed_data = transforms.RandomHorizontalFlip()(perturbed_data)

        if 3 in attacks :  
          # Image bruitée
          perturbed_data = perturbed_data + 0.03*torch.randn(*X.shape).to(device)
          perturbed_data = torch.clamp(perturbed_data,0.,1.)

        if 4 in attacks:
          perturbed_data = transforms.RandomCrop(32,padding=2,padding_mode='reflect')(perturbed_data)
        

 
        perturbed_data = torch.tensor(perturbed_data,requires_grad=False)
        # perturbed_data.requires_grad = True
        # Adversarial training
        # train_model(model,optimizer,loss_fn,
        #             images=X,
        #             targets=y)


        train_model(model,optimizer,loss_fn,
                    batch_data=perturbed_data,
                    targets=y) 


        # ex = perturbed_data.detach().cpu()
        # plot_examples(ex[:5])

      scheduler.step()



      #### Test 
      model.eval()
      size = len(testloader.dataset)
      num_batches = len(testloader)
      test_loss = 0
      correct = 0


      for X, y in testloader:
          X = X.to(device)
          y = y.to(device)

          # Attack
          perturbed_data = PGD_linf(model, X, y,device,
                                alpha=params["alpha"],
                                epsilon=params["epsilon"],
                                n_iter=params["n_iter"])

          # Prédiction
          preds = model(perturbed_data)   


          test_loss += loss_fn(preds, y).item()
          correct += (preds.argmax(1) == y).type(torch.float).sum().item()     
      
      test_loss /= num_batches
      correct /= size
      print(
          f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
      )
      print("-----------------------------------------------------")
      accuracies.append(correct)
      test_losses.append(test_loss)


      torch.save(model.state_dict(), f"/content/drive/MyDrive/model_CIFAR/resnet_adv_({100*correct}).pt")

  
  return accuracies






def adversarial_training_defense(model,autoencoder,device, trainloader,testloader,params):
    "Entrainement advsersarial"
    correct = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=params["params_model"]["lr"])
    
    
    criterion_autoencoder = nn.MSELoss()
    optimizer_autoencoder = optim.RMSprop(autoencoder.parameters(),
                                       lr=params["params_autoencoder"]["lr"])


    for epoch in range(params["epochs"]):
      autoencoder_loss = 0.0

      for batch, (X, y) in enumerate(tqdm(trainloader)):               
        X = X.to(device)
        y = y.to(device)


        ### Clean examples to train autoencoder
        if params["use_autoencoder"]:
          outputs, loss_autoencoder = train_autoencoder(autoencoder,
                                                  optimizer_autoencoder,
                                                  criterion_autoencoder,
                                                  batch_data=X,
                                                  targets=X,
                                                  train = False)

          reconstructed_exemples = outputs[:params["num_exemples"]].detach().cpu()
          autoencoder_loss += loss_autoencoder
        else:
          outputs = perturbed_data


        outputs_ = torch.tensor(outputs,requires_grad=False)
        outputs_.requires_grad = True
        
        train_model(model,
                    optimizer,
                    loss_fn,
                    batch_data=outputs_,
                    targets=y)
                      



        #### Data Augmentation
        noise_types = ["gaussian","salt","speckle"]
        noise_type = random.choice(noise_types)
        with torch.no_grad():
          perturbed_data = generate_noisy_data(X, device,
                                              noise_type="gaussian")


        noisy_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()

        ## Entrainement sur l'image bruitée
        train_model(model,
                    optimizer,
                    loss_fn,
                    batch_data=perturbed_data,
                    targets=y)


        ## Recontructor
        if params["use_autoencoder"]:
          outputs, loss_autoencoder = train_autoencoder(autoencoder,
                                                  optimizer_autoencoder,
                                                  criterion_autoencoder,
                                                  batch_data=perturbed_data,
                                                  targets=X,
                                                  train = False)            
          reconstructed_exemples = outputs[:params["num_exemples"]].detach().cpu()
          autoencoder_loss += loss_autoencoder
        else:
          outputs = perturbed_data



        ## Entrainement sur la sortie de l'encodeur
        outputs_ = torch.tensor(outputs,requires_grad=False)
        outputs_.requires_grad = True

        train_model(model,
                    optimizer,
                    loss_fn,
                    batch_data=outputs_,
                    targets=y)
        
      
        ## Sauvegarde d'exemple d'images d'autoencodeur
        origin_exemples = X[:params["num_exemples"]].detach().cpu()
        

        #### Perturbation PGD 
        perturbed_data = PGD_linf(model, X, y,  device,                                  
                          epsilon=params["epsilon"],
                          alpha=params["alpha"],
                          n_iter=params["n_iter"])  

        pertub_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()
        
                      
        
        # Recontructor
        if params["use_autoencoder"]:
          outputs, loss_autoencoder = train_autoencoder(autoencoder,
                                  optimizer_autoencoder,
                                  criterion_autoencoder,
                                  batch_data=perturbed_data,
                                  targets=X,
                                  train = False)            
          reconstructed_exemples = outputs[:params["num_exemples"]].detach().cpu()
          autoencoder_loss += loss_autoencoder
        else:
          outputs = perturbed_data
                      
        
        ##### Entrainement sur la sortie de l'autoencodeur
        outputs_ = torch.tensor(outputs,requires_grad=False)
        outputs_.requires_grad = True

        outputs, loss_model = train_model(model,
                                          optimizer,
                                          loss_fn,
                                          batch_data=outputs_,
                                          targets=y)
        

        ##### Entrainement sur l'image attaquée directement
        outputs, loss_model = train_model(model,
                                          optimizer,
                                          loss_fn,
                                          batch_data=perturbed_data,
                                          targets=y)



      # #### PLOTS
      plot_examples(origin_exemples, title="Original")
      plot_examples(noisy_exemples, title="Reconstructed")
      plot_examples(reconstructed_exemples, title="Reconstructed")

      if params["use_autoencoder"]:
        plot_examples(reconstructed_exemples, title="Reconstructed")
        print("Autoencoder loss", autoencoder_loss)

      #### Test 
      model.eval()
      autoencoder.eval()
      size = len(testloader.dataset)
      num_batches = len(testloader)
      test_loss, correct = 0, 0


      for X, y in testloader:
          X = X.to(device)
          y = y.to(device)

          # Image de test attaquée
          X.requires_grad = True
          init_output = model(X)

          loss = loss_fn(init_output, y)
          model.zero_grad()
          loss.backward()

          perturbed_data = PGD_linf(model, X, y,  device,                                  
                                epsilon=params["epsilon"],
                                alpha=params["alpha"],
                                n_iter=params["n_iter"])     
                   
          
          pertub_test_exemples = perturbed_data[:params["num_exemples"]].detach().cpu()
          

          ##### Defense contre l'attaque
                        
          # Gausian noise
          perturbed_data = perturbed_data + params["noise_factor"]*torch.randn(*X.shape).cuda()
          perturbed_data = torch.clamp(perturbed_data,0.,1.)

          # Reconstruction de l'image
          if params["use_autoencoder"]:          
            outputs = autoencoder(perturbed_data)
            recon_test_exemples = outputs[:params["num_exemples"]].detach().cpu()
          else:
            outputs = perturbed_data



          # Prédiction
          preds = model(outputs)     
          test_loss += loss_fn(preds, y).item()
          correct += (preds.argmax(1) == y).type(torch.float).sum().item()
 


      plot_examples(pertub_test_exemples, title="Images attaquées")
          
      if params["use_autoencoder"]:    
        plot_examples(recon_test_exemples, title="Images attaquées reconstruites")

      
      # torch.save(autoencoder.state_dict(), "/content/drive/MyDrive/model_CIFAR/autoencoder_ddsaV2.pt")


      test_loss /= num_batches
      correct /= size
      print(
          f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
      )
      print("-----------------------------------------------------")
