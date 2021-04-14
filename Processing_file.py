import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

import PIL
from PIL import Image

def image_dataset(train_dir):
    train_transforms = transforms.Compose([transforms.Resize(300),
                                           transforms.CenterCrop(224),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    
    class_names = train_datasets.classes
    
    
    return train_dataloaders, train_datasets, len(class_names)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    Img = PIL.Image.open(image)
    
    image_trans = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    return np.array(image_trans(Img))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device);
    # TODO: Implement the code to predict the class from an image file
    
    #Processing image
    test_image = process_image(image_path)
    test_image = torch.from_numpy(test_image)
    test_image = torch.unsqueeze(test_image, 0)
    test_image = test_image.to(device)
    
    #Passing Image through model
    output = model.forward(test_image)
    
    probs, classes  = output.topk(topk)
    
    probs = torch.exp(probs)[0].cpu().detach().numpy()
    classes = classes[0].cpu().detach().numpy()
    
    return probs, classes