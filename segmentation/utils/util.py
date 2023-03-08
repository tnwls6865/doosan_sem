import os
import pickle
import numpy as np


def train_val_split_(path):

    with open(os.path.join(path,'images_random.txt'), 'rb') as f:
        data = pickle.load(f)

      
    indices = np.random.choice(['train', 'valid'], p=[.9, .1], size=len(data))
    train_list = []
    val_list = []
    
    
    for i in range(len(data)):
        if indices[i] == 'train':
            train_list.append(data[i])

        elif indices[i] == 'valid':
            val_list.append(data[i])

    with open(os.path.join(path, 'train_random.txt'), 'wb') as t:
        pickle.dump(train_list, t)

    with open(os.path.join(path, 'val_random.txt'), 'wb') as v:
        pickle.dump(val_list, v)