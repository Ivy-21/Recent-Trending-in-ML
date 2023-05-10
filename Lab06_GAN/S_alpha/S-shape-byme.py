#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from logger import *

#%%
class s_dataset(Dataset):
    
    def __init__(self, num_sample = 1000, transform=None):
        
        pi = np.pi
        self.data = torch.zeros([num_sample,2])

        for i in range(num_sample):
            theta = torch.FloatTensor(1).uniform_(0, 2*pi)
            r = torch.randn(1)
            x = (10+r) * torch.cos(theta)
            

            if 0.5*pi <= theta and theta <= 1.5*pi:
                y = ((10+r) * torch.sin(theta)) + 10
            else:
                y = ((10+r) * torch.sin(theta)) - 10

            self.data[i,0] = x
            self.data[i,1] = y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        return data

#%%
train_dataset = s_dataset(num_sample = 100000, transform = None)

plt.figure(figsize = (8,8))
for i in range(1000):
    data = train_dataset.__getitem__(i)
    plt.scatter(data[0], data[1])
plt.show()
# %%

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
num_batches = len(data_loader)

for i, data in enumerate(data_loader):
    plt.scatter(data[i,0], data[i,1])
    if i == 50:
        break
plt.figure(figsize=(8,8))
plt.show()
# %%

import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        #n_features = 100
        n_features = 2
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x
  
# def images_to_vectors(images):
#     print(images.size(0))
#     return images.view(images.size(0), 784)

# def vectors_to_images(vectors):
#     return vectors.view(vectors.size(0), 1, 28, 28)

#%%

import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
#from matplotlib import pyplot as plt
import torch
import torch.nn as nn


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 2
        n_out = 2
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2)
        # )
        
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x

# Function to create noise samples for the generator's input

def noise(size):
    n = torch.randn(size, 2)
    if torch.cuda.is_available(): return n.cuda() 
    return n


#%%
discriminator = DiscriminatorNet()
generator = GeneratorNet()

# # # from chosen_gpu import get_freer_gpu
# device = torch.device(cuda if torch.cuda.is_available() else torch.device("cpu")
# # print("Configured device: ", device)
# # #device = 1
# if torch.cuda.is_available():
#     discriminator.cuda(device)
#     generator.cuda(device)

if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

#%%
# Optimizers

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function

loss = nn.BCELoss()

# How many epochs to train for

num_epochs = 30

# Number of steps to apply to the discriminator for each step of the generator (1 in Goodfellow et al.)

d_steps = 1

#%%

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1)
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1)
    if torch.cuda.is_available(): return data.cuda()
    return data


#%%
def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # Propagate real data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # Propagate fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # Take a step
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

#%%
def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # Propagate the fake data through the discriminator and backpropagate.
    # Note that since we want the generator to output something that gets
    # the discriminator to output a 1, we use the real data target here.
    prediction = discriminator(fake_data)
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    # Return error
    return error

#%%
# # Function to create noise samples for the generator's input

# def noise(size):
#     n = torch.randn(size, 2)
#     if torch.cuda.is_available(): return n.cuda() 
#     return n

num_test_samples = 100
test_noise = noise(num_test_samples)

#%%

def plt_output(fake_data):
    plt.figure(figsize=(8,8))
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.scatter(fake_data[:,0], fake_data[:,1])
    plt.show()



#%%


logger = Logger(model_name='VGAN', data_name='S-Shape')

for epoch in range(num_epochs):
    for n_batch, real_batch in enumerate(data_loader):

        # Train discriminator on a real batch and a fake batch
        
        real_data =real_batch
        if torch.cuda.is_available(): real_data = real_data.cuda()
       
        fake_data = generator(noise(real_data.size(0))).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)
        
        # Train generator

        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)
        
        # Log errors and display progress

        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        if (n_batch) % 100 == 0:
            display.clear_output(True)
            # Display Images
            #test_images = vectors_to_images(generator(test_noise)).data.cpu()
            #test_plot = plt_output(generator(test_noise).cpu().detach().numpy())
            #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
            
        # Save model checkpoints
        logger.save_models(generator, discriminator, epoch)


