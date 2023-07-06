import os
import argparse
from PIL import Image
import pickle
from random import randrange
from pathlib import Path

def main(args): 
    
    with open(os.path.join(args.data_root_path, 'anno_train.txt'), 'rb') as f:
        data_list = pickle.load(f)

    img_path = os.path.join(args.data_root_path, 'images')
    mask_path = os.path.join(args.data_root_path, 'train')

    save_img = Path(args.save_img)
    save_img.mkdir(parents=True, exist_ok=True)

    save_msk = Path(args.save_msk)
    save_msk.mkdir(parents=True, exist_ok=True)


    new_img_list = []
    hard_sample_name = ['900_g0411_1_1.png', '900_g0411_1_2.png', '900_g0411_1_3.png', '900_g0411_1_4.png',
                        '900_g0411_1_5.png', '900_g0411_1_6.png', '900_g0411_1_7.png', '900_g0411_1_8.png', 
                        '900_g0411_1_9.png', '900_g0411_1_10.png', '900_g0411_1_11.png', '900_g0411_4_6.png']
    for name in data_list:
        name = name.split(' ')[0]
        img = Image.open(os.path.join(img_path, name))
        mask = Image.open(os.path.join(mask_path, name))

        x, y = img.size
        
        matrix_w = 640
        matrix_h = 448
        
        img_list = []
        mask_list = []

        if name in hard_sample_name:
            sample = 50 
        else:
            sample = 20

        for i in range(sample):
            x1 = randrange(0, x - matrix_w)
            y1 = randrange(0, y - matrix_h)
            new_img = img.crop((x1, y1, x1 + matrix_w, y1 + matrix_h))
            new_msk = mask.crop((x1, y1, x1 + matrix_w, y1 + matrix_h))
            new_name = name[:-4] + '_' + str(i) + '.png'
            new_img_list.append(new_name)
            new_img.save(os.path.join(save_img, new_name))
            new_msk.save(os.path.join(save_msk, new_name))

    with open(os.path.join(args.data_root_path, 'images_random.txt'), 'wb') as f:
        pickle.dump(new_img_list, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data random cropping')

    parser.add_argument('--data', type=str, default='CM939W', help=" IN792sx | IN792sx_inter | CM939W")
    parser.add_argument('--data_root_path', type=str, default='/HDD/dataset/doosan/CM939W/')
    parser.add_argument('--save_img', type=str, default='/HDD/dataset/doosan/CM939W/img_random')
    parser.add_argument('--save_msk', type=str, default='/HDD/dataset/doosan/CM939W/img_seg_random')

    args = parser.parse_args()

    main(args)