
import numpy as np
from PIL import Image
import glob as gl
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import MinMaxScaler


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.detach().numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0

    minmax_scaler = MinMaxScaler()
    for index in range(batch_size):
        
        result = predictions[index][0].cpu().detach().numpy()
        result[result < 0.5] = 0
        result[result >= 0.5] = 255

        mask_ = masks[index][0].cpu().detach().numpy()
        mask_[mask_ < 0.5] = 0
        mask_[mask_ >= 0.5] = 255

        total_acc += accuracy_check(mask_, result)
    return total_acc/batch_size



    
