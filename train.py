import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from Processing_file import image_dataset
from model_info import Arch_select, model_cfy, validation
from workspace_utils import active_session

import PIL
from PIL import Image

Dataset = input("Indicate location of Dataset: ")
dataloader, dataset, image_classes = image_dataset(Dataset)

Arch_input = input("Indicate architecture for the model: ")
model, input_size = Arch_select(Arch_input)
    
model, criterion, optimizer, output_cat, hidden_layers, input_size = model_cfy(model, image_classes, input_size)

# Making sure the model is computed with GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device);

# This code trains the model and displays it's accuracy as it runs.

epochs = int(input("Indicate number of epochs to run: "))
print_every = 40
steps = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
with active_session():
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in iter(dataloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloader)))

                running_loss = 0
                
model.class_to_idx = dataset.class_to_idx
model.cpu()
checkpoint = {'input_size': input_size,
              'arch': 'densenet169',
              'output_size': output_cat,
              'hidden_layers': hidden_layers,
              'Epochs': epochs,
              'Optimizer': optimizer.state_dict,
              'Criterion': criterion.state_dict,
              'state_dict': model.state_dict(),
             'class_to_idx': model.class_to_idx}

save_file = input("Enter Save File name with file extension: ")
torch.save(checkpoint, save_file)