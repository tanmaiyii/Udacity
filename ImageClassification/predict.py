#Import Libraries used in prediction
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms, datasets
from collections import OrderedDict
from torch.optim import lr_scheduler
import time
from PIL import Image
import numpy as np
from network import Network
import sys
import json
import pickle

#Load the checkpoint

def load_checkpoint(filepath):
    
    """ 
    Ensure the Working Directory is set to the one containing the .py file to run the default settings. If else please
    provide full path of the .py and full path of data directory in the arguments. 
    """

    checkpoint = torch.load(filepath)
    
    
    if checkpoint['model_type'] == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.name = 'vgg11'
    
    elif checkpoint['model_type'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = 'vgg13'
    
    elif checkpoint['model_type'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
    
    else:
        model = models.vgg19(pretrained=True)
        model.name = 'vgg19'
    
    #Repeating the same process as for training the network to load the model.
   # model = models.checkpoint['model_type'](pretrained=True)
    print(f"Model {model.name} Loaded successfully \n")
    
    #Freeze the parameters for this.
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device);
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    #np_image = np.array(im)
    
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    
    return process(im)
                    
def predict(image_path, model, device, top_k, category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Calculate the class probabilities (softmax) for images in test set    
    image = process_image(image_path)
    image.unsqueeze_(0)
    #print(image.shape)
    model.to(device).float()
    model.eval()

    classes = []
    with torch.no_grad():
            pred = model.forward(image.to(device))
            ps = torch.exp(pred)
            top_p, top_class = ps.topk(top_k, dim=1)
    #print(top_class[0])
#Get the class value from the index
    classes = top_class[0]
    
    #classes = top_class.tolist()[0]
    
 #Take the class from the index
    #print(model.class_to_idx)
    preds = []
    for v in classes:
        preds.append(list(model.class_to_idx)[v])

#Load the json file
    with open(category_names, 'r') as f:
            category_names = json.load(f)
        
#Get the appropriate class label using cat_to_name json file created before
    top_labels = []
    for p in preds:
        top_labels.append(category_names[p])
    
    return top_p, top_labels

#Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='load the image path - mandatory argument. Can only load per image')
parser.add_argument('filepath', help='the path to load the checkpoint - mandatory argument')
parser.add_argument ('--top_k', help = 'Top K most likely classes', type = int, default=1)
parser.add_argument ('--category_names', help = 'Map category labels to actual names. JSON file name to be provided. Optional', type = str)
parser.add_argument('--gpu', action='store_true', help = 'Option to use GPU')
args = parser.parse_args()

#check for gpu
if args.gpu:
    cuda_ = torch.cuda.is_available() 
    if cuda_:
        print("Training model on GPU.")
        device = 'cuda'
    else:
        print("Cuda not available. Training model on CPU instead")
        device = 'cpu'
else:
    print("Training model on CPU.")
    device = 'cpu'

filepath = args.filepath
#Load the checkpoint model
model = load_checkpoint(filepath)


#Setup image path
image_path = args.image_path
    
#Make Prediction
top_k = args.top_k 

#Mapping to Category Names
if not args.category_names:
        category_names = 'cat_to_name.json'
else:
    category_names = args.category_names
        
top_p, top_labels = predict(image_path, model, device, top_k, category_names)
    

print(f"Prediction: Flower of Class {top_labels} with {top_p} probability")
    
    
    

    



