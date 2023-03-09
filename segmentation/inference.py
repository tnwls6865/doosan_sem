import argparse
import cv2
import os
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import segmentation_models_pytorch as smp
from utils.accuracy import accuracy_check
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def main(args):

    with open(os.path.join(args.data_root_path, 'images_anno.txt'), 'rb') as f:
        data_list = pickle.load(f)

    save_dir = os.path.join('/HDD/tnwls/doosan/history/230302/', 'CM939W/seg_result_anno/')
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model_best.pt')))

    model.cuda()

    # Train
    total_acc = 0
    toTensor = ToTensor() 
    minmax_scaler = MinMaxScaler()
    model.eval()

    transform = transforms.Compose([transforms.Resize((512,512)),
        transforms.ToTensor(),
     ])

    for data in data_list:
        data = data.split(' ')[0]

        img = Image.open(os.path.join(args.image_path, data))
        msk = Image.open(os.path.join(args.mask_path, data))
        
        img = toTensor(img).cuda().unsqueeze(0)
        msk = toTensor(msk).cuda().unsqueeze(0)

        with torch.no_grad():
            output = model(img)

        result = output[0][0].cpu().numpy()
        result[result < 0.5] = 0
        result[result >= 0.5] = 255

        mask_ori = msk[0][0].cpu().numpy()
        mask_ori[mask_ori < 0.5] = 0
        mask_ori[mask_ori >= 0.5] = 255

        palette = [0,0,0, 255,255,255]
        out = Image.fromarray(result.astype(np.uint8), mode='P')
        out.putpalette(palette)

        mask = Image.fromarray(mask_ori.astype(np.uint8), mode='P')
        mask.putpalette(palette)

        export_name = str(data)
        out.save(save_dir + export_name)
        acc = accuracy_check(mask_ori, out)
        total_acc += acc

        fig = plt.figure()
        rows = 1
        cols = 3

        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(img[0][0].cpu().numpy(), 'gray')
        ax1.set_title('original image')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(result.astype(np.uint8), 'gray')
        ax2.set_title('prediction')
        ax2.axis("off")

        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(mask_ori, 'gray')
        ax3.set_title('ground truth')
        ax3.axis("off")

        export_name_fig = export_name[:-4] +'_fig.png'
        plt.savefig(save_dir + export_name_fig)
    
    print('total_acc : ', total_acc/len(data_list))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sem img inference')

    parser.add_argument('--data_root_path', type=str, default='/HDD/dataset/doosan/CM939W/')
    parser.add_argument('--image_path', type=str, default='/HDD/dataset/doosan/CM939W/images')
    parser.add_argument('--mask_path', type=str, default='/HDD/dataset/doosan/CM939W/segmentation_images')
    parser.add_argument('--save_dir', type=str, default='/HDD/tnwls/doosan/history/230302/CM939W/')
    parser.add_argument('--model_path', type=str, default='/HDD/tnwls/doosan/history/230302/CM939W/resnet18_4_32_3/saved_models')

    args = parser.parse_args()

    main(args)

