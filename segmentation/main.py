import argparse
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import CustomDatasetTrain, CustomDatasetVal
from modules import train_model, validate_model
from utils.save_history import export_history, save_models
from utils.util import train_val_split_

import segmentation_models as smp


def main(args):
    
    train_val_split_(args.data_root_path)

    trainset = CustomDatasetTrain(args.image_path, args.mask_path)
    valset = CustomDatasetVal(args.image_path, args.mask_path) 

    save_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    SEM_train_load = torch.utils.data.DataLoader(dataset=trainset,
                                                 num_workers=args.num_worker, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=True)
    SEM_val_load = torch.utils.data.DataLoader(dataset=valset,
                                               num_workers=args.num_worker,
                                               batch_size=1, 
                                               shuffle=False)

    if args.data.lower() == 'in792sx' :   
        model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif args.data.lower() == 'in792sx_inter' :
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif args.data.lower() == 'cm939w':
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']

    fn_loss = nn.BCEWithLogitsLoss()

    # Train
    best_val_acc = 0
    print("Initializing Training!") 
    for i in range(0, args.epochs):
        # # train the model
        train_loss, train_acc = train_model(i, model, SEM_train_load, fn_loss, optimizer)
        
        # # # # # Validation every 5 epoch
        if (i+1) % args.ckpt_interval == 0:
            val_loss, val_acc = validate_model(
                i+1, model, SEM_val_load, fn_loss, True, save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_path)

            if best_val_acc <= val_acc :  
                best_val_acc = val_acc
                save_models(model, save_path, i+1, True)

        if (i+1) % args.ckpt_save == 0: 
            save_models(model, save_path, i+1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sem img segmenation')

    parser.add_argument('--data', type=str, default='CM939W', help=" IN792sx | IN792sx_inter | CM939W")
    parser.add_argument('--data_root_path', type=str, default='./data/CM939W/')
    parser.add_argument('--image_path', type=str, default='./data/CM939W/img_random')
    parser.add_argument('--mask_path', type=str, default='./data/CM939W/img_seg_random')
    parser.add_argument('--save_dir', type=str, default='./data//history/230302/CM939W/')
    parser.add_argument('--exp_name', type=str, default='resnet18_4_32_3_check2')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--ckpt_interval', type=int, default=2)
    parser.add_argument('--ckpt_save', type=int, default=10)

    args = parser.parse_args()

    main(args)
