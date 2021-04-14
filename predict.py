import matplotlib.pyplot as plt
import json

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from Processing_file import process_image, predict
from model_info import load_checkpoint
from workspace_utils import active_session

import PIL
from PIL import Image

file_load = input("Input checkpoint file to be loaded: ")
model, optimizer, criterion, epoch, input_size, output_size, hidden_layers = load_checkpoint(file_load)

file_image = input("Input the image file to be categorised: ")

topk = int(input("Input the number of top predictions you want: "))
probs, classes = predict(file_image, model, topk)


json_file = input("JSON file to be loaded for label referencing: ")
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
    print()

print("Classes : Probability\n\n")

for i in range(topk):
    print(cat_to_name[str(classes[i]+1)] + " : " + str(probs[i]))