import os
import csv
from tkinter.filedialog import SaveFileDialog
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
