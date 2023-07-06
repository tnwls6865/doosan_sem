import os
import numpy as np
import pickle
import dill
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
import argparse

parser = argparse.ArgumentParser()

# "option" _원하는 데이터 설정
parser.add_argument('-data_option', help="in792sx | in792sx_interrupt | cm939w")
# "feature_num" _이미지 피쳐값 설정
parser.add_argument('-feature_num', help="1 ~ 6")

opt = parser.parse_args()


def main():
    option = opt.data_option
    feature_num = opt.feature_num
    
    ## 데이터셋 위치 설정
    if option == 'in792sx' :   
        data_root_path = '/HDD/dataset/doosan/tmp/'    
        image_path = '/HDD/dataset/doosan/tmp/images/'
        model_path = '/home/tnwls/code/gasturbin/tnwls/gamma_model_best.pt'
    elif option == 'in792sx_interrupt' :
        data_root_path = '/HDD/jungmin/doosan/interrupt_add'
        image_path = '/HDD/jungmin/doosan/interrupt_add/Interrupt_Image'
        model_path = '/HDD/tnwls/doosan/history/230227/IN792sx_interrupt/resnet18_4_32/saved_models/model_best.pt' 
    elif option == 'cm939w':
        data_root_path = '/HDD/jungmin/doosan/cm939_add'
        image_path = '/HDD/jungmin/doosan/cm939_add/0324'
        model_path = '/HDD/tnwls/doosan/history/230324/CM939W/model_best.pt'

    if option == 'in792sx' :   
        with open(os.path.join('/HDD/jungmin/doosan', 'images.txt')) as f:
            lines = f.readlines()
        data_list = [line.rstrip('\n') for line in lines]
    else:    
        with open(os.path.join(data_root_path, 'images.txt')) as f:
            lines = f.readlines()
        data_list = [line.rstrip('\n') for line in lines]
            
    if option == 'in792sx' :   
        model = smp.DeepLabV3('resnet34', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif option == 'in792sx_interrupt' :
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    elif option == 'cm939w':
        model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights=None, in_channels=1,decoder_channels=32)
    
    model.load_state_dict(torch.load(os.path.join(model_path)))
    model.cuda()

    ## Train
    total_acc = 0
    toTensor = ToTensor() 
    minmax_scaler = MinMaxScaler()
    model.eval()

    image_name = []
    feature_output=[]

    transform = transforms.Compose([transforms.Resize((512,512)),
        transforms.ToTensor(),
     ])
    
    for data in data_list:
        data = data.split(' ')[0]
        img = Image.open(os.path.join(image_path, data))
        img = transform(img).cuda().unsqueeze(0)

        image_name.append(data)

        with torch.no_grad():
            output, features, tmp = model(img)
            decoder_output = tmp
            
            ## feature 및 decoder_output 추출 후 저장
            if feature_num == "1":
                feature_output.append(features[0])
                feature_name = "f1"
            elif feature_num == "2":
                feature_output.append(features[1])
                feature_name = "f2"
            elif feature_num == "3":
                feature_output.append(features[2])
                feature_name = "f3"
            elif feature_num == "4":
                feature_output.append(features[3])
                feature_name = "f4"
            elif feature_num == "5":
                feature_output.append(features[4])
                feature_name = "f5"
            elif feature_num == "6":
                feature_output.append(decoder_output)
                feature_name = "decoder_output"
    
    data={
         'image_name' : image_name,
         'feature_output':feature_output,
           }
    
    ## 결과 저장 위치 설정
    output_dir = f'/home/jungmin/workspace/doosan/image_features_{str(option)}_{str(feature_name)}.pkl'
    print('[Info] Dumping the image features to pickle file')
    dill.dump(data, open(output_dir, 'wb'))
    print('[Info] Done..')
    
if __name__ == "__main__":
    main()
