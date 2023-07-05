import os
import csv
import numpy as np
from tkinter.filedialog import SaveFileDialog
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

import torch


def export_history(header, value, folder):
    """ export data to csv format
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder (list): folder path
        file_name: file name with path
    """

    file_name = folder + '/history_doosan.csv'
    # if folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)

    # if there is no file make file
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # if there is file overwrite
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # close file when it is done with writing
    file.close()


def save_models(model, path, epoch, flag=False):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
        epoch: the epoch the model finished training
    """
    path = os.path.join(path, 'saved_models/')
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+"/model_epoch_{0}.pt".format(epoch))
    if flag:
        torch.save(model.state_dict(), path+"/model_best.pt")


def save_prediction_image(result, im_name, epoch, save_dir="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
    """

    save_folder_name = os.path.join(save_dir, 'result_images')
    
    # minmax_scaler = MinMaxScaler()
    # result = minmax_scaler.fit_transform(stacked_img[0].cpu())
    # result[result < 0.5] = 0
    # result[result >= 0.5] = 255

    palette = [0,0,0, 255,255,255]
    out = Image.fromarray(result.astype(np.uint8), mode='P')
    out.putpalette(palette)
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name[0])
    out.save(desired_path + export_name)
    return out


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img