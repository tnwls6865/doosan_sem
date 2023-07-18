import csv
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.nn as nn

from utils.accuracy import accuracy_check, accuracy_check_for_batch
from utils.save_history import save_prediction_image

def train_model(epoch, model, data_train, criterion, optimizer):

    """Train the model and report training error with accuracy

    Args:
        epoch: epoch
        model: the model to be trained
        data_train(DataLoader): training dataset
        criterion: loss function
        optimizer: optimizer

    Retrun:
        total loss: loss value for training dataset
        total acc: accuracy value for training dataset
    """

    total_acc = 0
    total_loss = 0
    batches = 0

    model.train()

    for batch, (name, images, masks) in enumerate(data_train):

        images = Variable(images.cuda(), requires_grad=True) 
        masks = Variable(masks.cuda())
       
        b, h, w = masks.shape
        tmp_masks = (masks.clone().detach().view(b,1,h,w)).type(torch.float)
        
        outputs, encoder_features, decoder_output = model(images)
        
        loss = criterion(outputs, tmp_masks)

        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        acc = accuracy_check_for_batch(tmp_masks.cpu(), outputs.cpu(), images.size()[0])
        total_acc = total_acc + acc
        total_loss = total_loss + loss.cpu().item()

        if (batch+1) % 5 == 0:
            print('Epoch', str(epoch+1), 'Train loss:', loss.item(), "Train acc", acc)

    return total_loss/len(data_train), total_acc/len(data_train)


def validate_model(epoch, model, data_val, criterion, make_prediction=True, save_dir='prediction'):
    """Validate the model and report validation error with accuracy

    Args:
        epoch: epoch
        model: the trained model
        data_val(DataLoader): validation dataset
        criterion: loss function
        make_prediction: flag of the saving prediction images
        save_dir: saving the prediction images folder

    Retrun:
        total loss: loss value for validation dataset
        total acc: accuracy value for validation dataset
    """
    # calculating validation loss
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    minmax_scaler = MinMaxScaler()
    for batch, (image, mask, original_msk, name) in enumerate(data_val):
        
        with torch.no_grad():
            
            image = Variable(image.unsqueeze(0).cuda())
            mask = Variable(mask.cuda())
        
            b, h, w = mask.shape
            tmp_mask = (mask.clone().detach().view(b,1,h,w)).type(torch.float)
            output, _, _ = model(image)
    
            loss = criterion(output, tmp_mask)
            total_val_loss = total_val_loss + loss.cpu().item()

            result = minmax_scaler.fit_transform(output[0][0].cpu())
            result[result < 0.5] = 0
            result[result >= 0.5] = 255

            gt_msk = minmax_scaler.fit_transform(tmp_mask[0][0].cpu())
            gt_msk[gt_msk < 0.5] = 0
            gt_msk[gt_msk >= 0.5] = 255

            acc_val = accuracy_check(gt_msk, result)
            total_val_acc = total_val_acc + acc_val

        if make_prediction:
            im_name = name 
            pred_msk = save_prediction_image(result, im_name, epoch, save_dir)

    return total_val_loss/len(data_val), total_val_acc/len(data_val)

