# Source: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

# Transforms the image into numbers between 0 and 1 for
# red, green, and blue
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Downloading the tranining and testing datasets and load them to DataLoader
# which combines the data-set and a sampler and provides single- or 
# multi-process iterators over the data-set. 
# batch_size = # of images to be processed every time
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# ============ Building the neural network model ==============
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# ReLU activation ( a simple function which allows positive values to pass through, 
# whereas negative values are modified to zero ). The output layer is a linear 
# layer with LogSoftmax activation because this is a classification problem.
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print("\n")

print(model) # printing the model

criterion = nn.NLLLoss() # negative likelihood loss

# ======================= Training process =========================
# optim: perform gradient descent and update the weights by back-propagation
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 10

print("\n")

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        # This is where the model learns by backpropagating
        loss.backward()
        
        # Weight optimization occurs here
        optimizer.step()
        
        running_loss += loss.item()
    
    print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

# ======================= Testing process =========================
correct_count, all_count = 0, 0

for images,labels in valloader: # loading from the testing set
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    with torch.no_grad():
        logps = model(img) # log probabilities

    ps = torch.exp(logps) # actual probabilities

    print(ps)
    
    probab = list(ps.numpy()[0])

    pred_label = probab.index(max(probab)) # prediction based on model
    true_label = labels.numpy()[i]         # true label

    if(true_label == pred_label):
      correct_count += 1

    all_count += 1

print("\nNumber Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count), "\n")

# =================== Saving the model =================
torch.save(model, './my_mnist_model.pt') 
