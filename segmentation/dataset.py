import cv2
import os
import numpy as np
import pickle
from PIL import Image
import glob
import random
from random import randint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

from utils.pre_processing import *
from utils.mean_std import *

Training_MEAN = 0.4911
Training_STDEV = 0.1658

class CustomDatasetTrain(Dataset):

    def __init__(self, image_path, mask_path):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
        self.image_path = image_path
        self.mask_path = mask_path
        self.data_list = self.__read_data__()

        # Calculate len
        self.data_len = len(self.data_list)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        """
        # GET IMAGE
        """
        # eta_list = ['210415-AC1-1_m001_r1', '210415-AC1-1_m002_r1', '210415-AC1-1_m003_r1']
        single_image_name = self.data_list[index]
        img_as_img = Image.open(os.path.join(self.image_path, single_image_name)).convert("L")

        # if randint(0,1):
        #     rotater = T.RandomRotation(degrees=(0, 180))
        #     img_as_img = rotater(img_as_img)
        if randint(0,1):
            sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=10)
            img_as_img = sharpness_adjuster(img_as_img)

        img_as_np = np.asarray(img_as_img)
      
        flip_num = randint(0, 3)
        img_as_np = flip(img_as_np, flip_num)

        # # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        if randint(0, 1):
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

        
        # Brightness
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)

        """
        # GET MASK
        """
        single_mask_name = self.data_list[index]
        msk_as_img = Image.open(os.path.join(self.mask_path, single_mask_name)).convert("L")
        msk_as_np = np.asarray(msk_as_img)

        # flip the mask with respect to image
        msk_as_np = flip(msk_as_np, flip_num)
        

        # Normalize the image
        img_as_np = normalization2(img_as_np, max=1, min=0)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        return self.data_list[index], img_as_tensor, msk_as_tensor
    

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len

    def __read_data__(self):
        
        name = 'train_random.txt'

        root_path = '/'.join(self.image_path.split('/')[:-1])
        with open(os.path.join(root_path, name), 'rb') as f:
            data_list = pickle.load(f)
       
        return data_list

class CustomDatasetVal(Dataset):
    def __init__(self, image_path, mask_path):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # paths to all images and masks
        self.image_path = image_path
        self.mask_path = mask_path
        self.data_list = self.__read_data__()
        self.data_len = len(self.data_list)

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """
        eta_list = ['210415-AC1-1_m001_r1', '210415-AC1-1_m002_r1', '210415-AC1-1_m003_r1']
        
        single_image = self.data_list[index]
        
        # if single_image.split('.')[0] in eta_list:
        #     label = torch.FloatTensor([1])
        # else:
        #     label = torch.FloatTensor([0])

        
        img_as_img = Image.open(os.path.join(self.image_path, single_image)).convert("L")
        
        # if randint(0,1):
        #     rotater = T.RandomRotation(degrees=(0, 180))
        #     img_as_img = rotater(img_as_img)
        # if randint(0,1):
        #     sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
        #     img_as_img = sharpness_adjuster(img_as_img)

        # Convert the image into numpy array
        img_as_np = np.asarray(img_as_img)
        
        # Median blur
        # img_as_np = cv2.medianBlur(img_as_np, 11)

        # Otsu binary
        # otsu_threshold, image_result = cv2.threshold(img_as_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # th, dst = cv2.threshold(img_as_np, otsu_threshold, 255, cv2.THRESH_BINAR)

        """
        # GET MASK
        """
        single_mask_name = self.data_list[index]
        msk_as_img = Image.open(os.path.join(self.mask_path, single_mask_name)).convert("L")
        # msk_as_img = msk_as_img.resize((512,512))
            # msk_as_img.show()
        msk_as_np = np.asarray(msk_as_img)
        
        
        # for array in img_as_np:
            # Normalize the cropped arrays
        img_as_np = normalization2(img_as_np, max=1, min=0)
            # Convert normalized array into tensor
            # processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(img_as_np)
        
        msk_as_np = msk_as_np/255

        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        original_msk = torch.from_numpy(np.asarray(msk_as_img))
        return img_as_tensor, msk_as_tensor, original_msk, single_image
    

    def __read_data__(self):
        
        name = 'val_random.txt'
            
        root_path = '/'.join(self.image_path.split('/')[:-1])
        
        with open(os.path.join(root_path, name), 'rb') as f:
            data_list = pickle.load(f)
       
        return data_list

    def __len__(self):

        return self.data_len

