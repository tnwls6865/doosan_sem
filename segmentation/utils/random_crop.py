import os
from PIL import Image
import pickle
from random import randrange
from pathlib import Path

data_root_path = '/HDD/dataset/doosan/CM939W/'
save_img = '/HDD/dataset/doosan/CM939W/img_random/'
save_msk = '/HDD/dataset/doosan/CM939W/img_seg_random'

with open(os.path.join(data_root_path, 'anno_train.txt'), 'rb') as f:
    data_list = pickle.load(f)

img_path = os.path.join(data_root_path, 'images')
mask_path = os.path.join(data_root_path, 'segmentation_images')

save_img = Path(save_img)
save_img.mkdir(parents=True, exist_ok=True)

save_msk = Path(save_msk)
save_msk.mkdir(parents=True, exist_ok=True)


new_img_list = []
for name in data_list:
    name = name.split(' ')[0]
    img = Image.open(os.path.join(img_path, name))
    mask = Image.open(os.path.join(mask_path, name))

    x, y = img.size
    
    matrix_w = 640
    matrix_h = 448
    sample = 200
    img_list = []
    mask_list = []

    for i in range(sample):
        x1 = randrange(0, x - matrix_w)
        y1 = randrange(0, y - matrix_h)
        new_img = img.crop((x1, y1, x1 + matrix_w, y1 + matrix_h))
        new_msk = mask.crop((x1, y1, x1 + matrix_w, y1 + matrix_h))
        new_name = name[:-4] + '_' + str(i) + '.png'
        new_img_list.append(new_name)
        new_img.save(os.path.join(save_img, new_name))
        new_msk.save(os.path.join(save_msk, new_name))

with open(os.path.join(data_root_path, 'images_random.txt'), 'wb') as f:
    pickle.dump(new_img_list, f)