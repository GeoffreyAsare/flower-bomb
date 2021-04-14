import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

def Arch_select(arch):
    if arch == "densenet169":
        model = models.densenet169(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    else:
        Print("\nPlease select densenet169 or vgg16.")
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model, input_size

def model_cfy(model, image_classes, input_size):
    
    #Obtaining size information for hidden layers and Output layers
    hid_layNo = int(input("Enter the number of hidden layers the model classifier will have: "))
    hidden_layers = []
    dropP = []
    for i in range(hid_layNo):
        hidden_layers.append(int(input("Enter size of Hidden Layer " + str(i+1) + ": ")))
        dropP.append(input("Enter dropout value for this layer: "))
    
    output_size = image_classes
    
    module = [nn.Linear(input_size, hidden_layers[0])]
    module1 = [nn.Linear(hidden_layers[-1], output_size)]
    
    crit = input("Enter Criterion to be used (Cross Entropy or Negative Log Liklihood: ")
    
    if crit == "Cross Entropy":
        criterion = nn.CrossEntropyLoss()
        module1.append(nn.Softmax(dim=1))
    
    elif crit == "Negative Log Liklihood":
        criterion = nn.NLLLoss()
        module1.append(nn.LogSoftmax(dim=1))
    
    else:
        print("Please enter Criterion as 'Cross Entropy' or 'Negative Log Liklihood'")
    
    for i in range(hid_layNo-1):
        module.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        module.append(nn.ReLU())
        if dropP[i] == "":
            module.append(nn.Dropout(p=1))
        else:
            module.append(nn.Dropout(p=float(dropP[i])))
        
    
        
    classifier = nn.Sequential(*module, *module1)
    
    

    model.classifier = classifier
        
    
    optimzr = input("Input optimizer to use (SGD or Adam): ")
    learnrate = float(input("Input learning rate for the training: "))
    if optimzr == "SGD":
        optimizer = optim.SGD(model.classifier.parameters(), lr=learnrate)
    elif optimzr == "Adam":
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    

    
    return model, criterion, optimizer, output_size, hidden_layers, input_size

# This is the validation function for checking accuracuy of the model
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for images, labels in testloader:
        
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# A function that loads a checkpoint and rebuilds the model.
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == 'densenet169':
        model = models.densenet169(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    hidden_LS = [1000, 800, 600]
    output_cat = 102
    input_size = model.classifier.in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(model.classifier.in_features, hidden_LS[0])),
                              ('drop1', nn.Dropout(p=0.8)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_LS[0], hidden_LS[1])),
                              ('drop2', nn.Dropout(p=0.6)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_LS[1], hidden_LS[2])),
                              ('drop3', nn.Dropout(p=0.5)),
                              ('relu3', nn.ReLU()),
                              ('fc4', nn.Linear(hidden_LS[2], output_cat)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['Epochs']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    
    return model, optimizer, criterion, epoch, input_size, output_size, hidden_layers