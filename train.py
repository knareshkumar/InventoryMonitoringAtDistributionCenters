#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import sys
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd

def test(model, test_loader, criterion, device, hook):
    '''
    Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
   
 
    logger.info(f"Test set: Average loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs, hook):
    '''
    Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Start Model Training")

    hook.set_mode(smd.modes.TRAIN)
    best_loss = 1e6
    image_dataset = {'train':train_loader, 'valid':validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)

            running_loss = 0.0
            running_corrects = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))

        if loss_counter == 1:
            break
    return model


def net():

    #initializes a pretrained model
    
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features

    model.fc = nn.Sequential(
                   nn.Linear(num_features, 64),
                   nn.ReLU(inplace=True),
                   nn.Linear(64, 5))
    return model 


def create_data_loaders(train_data, test_data, validation_data, batch_size):
    #data loaders for train, test and validation data sets

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
    train_set = torchvision.datasets.ImageFolder(root = train_data, transform = training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    
    test_set = torchvision.datasets.ImageFolder(root = test_data, transform = testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    
    validation_set = torchvision.datasets.ImageFolder(root = validation_data, transform = testing_transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size) 
    
    return train_loader, test_loader, validation_loader


def main(args):
    #Initialize a model by calling the net function
    model = net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on Device {device}")
    
    #Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)

    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)

    train_loader, test_loader, validation_loader = create_data_loaders(
                        args.train_data, args.test_data, args.val_data, args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(model, train_loader, validation_loader,
                  loss_criterion, optimizer, device, args.epochs, hook)
    
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Start Model Testing")
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Model Saving")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # Specify all the hyperparameters you need to use to train your model.
    # Training settings
    parser.add_argument(
        "--batch-size", type=int, default=64
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=32
    )
    parser.add_argument(
        "--epochs", type=int, default=10
    )
    parser.add_argument(
        "--lr", type=float, default=1.0
    )
    parser.add_argument(
        "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"]
    )
    parser.add_argument(
        "--val_data", type=str, default=os.environ["SM_CHANNEL_VAL"]
    )
    
    args = parser.parse_args()
    print("args to train_model.py : ", args)
    main(args)
