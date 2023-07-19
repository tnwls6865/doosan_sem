import argparse
import os
import pickle
import glob
import numpy as np



def train_test_split_(path, save_path):
    data_list = glob.glob(path+'*.*')
      
    indices = np.random.choice(['train', 'test'], p=[.9, .1], size=len(data_list))
    train_list = []
    val_list = []
    
    
    for i in range(len(data_list)):
        if indices[i] == 'train':
            train_list.append(data_list[i])

        elif indices[i] == 'test':
            val_list.append(data_list[i])

    with open(os.path.join(save_path, 'train.txt'), 'wb') as t:
        pickle.dump(train_list, t)

    with open(os.path.join(save_path, 'test.txt'), 'wb') as v:
        pickle.dump(val_list, v)


if __name__ == "__main__":

   data_root_path = './data/IN792sx/gamma/anno_images/'
   save_path = './data/IN792sx/gamma/'
   
   train_test_split_(data_root_path, save_path)