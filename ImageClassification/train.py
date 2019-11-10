#Import Libraries used for training
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms, datasets
from collections import OrderedDict
#from workspace_utils import active_session
from torch.optim import lr_scheduler
import time
from PIL import Image
import numpy as np
from network import Network


def load_data_transform(data_dir):
    """ 
    Ensure the Working Directory is set to the one containing the .py file to run the default settings. If else please
    provide full path of the .py and full path of data directory in the arguments. 
    """
    
    if not data_dir:
        data_dir = 'flowers'
        
    #Make training, validation and testing images directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Create transformations for training, validation and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data
    

def load_model(arch):
    
    """
    This function loads a pre-trained model with the given architecture.
    In this case, I've only set it up for vggnet 
    
    Arguments:
        arch: the architecture for vgg16 or vgg19.
        
    Returns: pre-trained model
    """
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.name = 'vgg11'
        
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = 'vgg13'
    
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
    
    else:
        model = models.vgg19(pretrained=True)
        model.name = 'vgg19'
        
    return model, arch


def classifier(model,hidden_layers):
    #The default hidden layers used to build the architecture (incase no hidden_layers are inputted in terminal
    if not hidden_layers:
        hidden_layers = args.hidden_units
    
       
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #Build the classifier to be used with the pre-trained model
    classifier = Network(input_size = 25088,
                         output_size = 102,
                         hidden_layers = hidden_layers,
                         drop_p = 0.2)
    
    #Assign the model's classifier with the above classifier
    model.classifier = classifier
    model.to(device)
    
def train(model, trainloader, testloader, criterion, optimizer, epochs):
    
    
    print("Model Training is starting \n")
    #Train the network and use the validation set to improve accuracy
    since = time.time()
    steps = 0
    running_loss = 0
    #print_every = 5
    
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            #scheduler.step()
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        else:
        #if steps % print_every == 0:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    val_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            #train_losses.append(running_loss/len(trainloader))
            #val_losses.append(val_loss/len(validloader))
            
            time_since = time.time() - since
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(validloader)),
                  "Validation accuracy: {:.3f}..".format(accuracy/len(validloader)),
                  "Training complete in {:.0f}m {:.0f}s".format(time_since // 60, time_since % 60))
            
            running_loss = 0
            model.train()
            
            print("Training Completed \n")

def validation(model, testloader, criterion):
    
    print("Validation on test set starting \n")
    model.to(device)
    model.eval()
    test_loss = 0
    steps = 0
    accuracy = 0

    # Calculate the class probabilities (softmax) for images in test set
    with torch.no_grad():
        for images, labels in testloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)

            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    
def save_model(model, data, filepath):

    #Extract the class_to_idx transformation 
    model.class_to_idx = train_data.class_to_idx
    model.to('cpu')

    #Create the checkpoint
    checkpoint = {'input_size':25088,
                  'output_size':102,
                  'hidden_layers':[each.out_features for each in model.classifier.hidden_layers],
                  'drop_p':0.2,
                  'state_dict':model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'classifier':model.classifier,
                  'model_type' : arch}
    
    #Save the model
    torch.save(checkpoint, filepath)
    
#Load the arguments 

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help='train on the data directory - mandatory argument. If not set please use ')
parser.add_argument('--save_dir', help='the path where to save the checkpoint - optional')
parser.add_argument('--arch', default='vgg19', help='choose different VGGNet models. Default is vgg19', type=str)
parser.add_argument('--learning_rate', type=float, default=0.001, help= 'The deault learning rate is 0.001')
parser.add_argument('--hidden_units',  nargs='+', type=int, default=[2048,1024], help= 'The default is that we use 2048 and 1024 as the hidden units. Please enter the number of hidden units with space, no comma and do not enclose in list')
parser.add_argument('--epochs', type=int, default=6, help = 'The default epochs is 6' )
parser.add_argument('--gpu', action='store_true')
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

if not args.data_dir:
    data_dir = 'flowers'
else:
    data_dir = args.data_dir

#Load data and make necessary transformations
trainloader, validloader, testloader, train_data = load_data_transform(data_dir)

arch = args.arch

#Load model if desired architecture
model_load = load_model(args.arch)
model = model_load[0]
                    
hidden_layers = args.hidden_units
                   
                        
                    

#Build classifier and incorporate it into the pre-trained model
classifier(model, hidden_layers)

#Define the loss function (criterion), the optimizer and the device
criterion = nn.NLLLoss()

#Define Learning Rate
lr = args.learning_rate

#Build optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Send model to device
model.to(device);



#Train the model
epochs = args.epochs


train(model,trainloader, validloader, criterion, optimizer, epochs)



#Test the Model
validation(model,testloader,criterion)

#Save the model
if args.save_dir:
    filepath = args.save_dir + '/checkpoint.pth'
else:
    filepath = 'checkpoint.pth'                   


save_model(model, train_data, filepath)
