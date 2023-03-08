import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import pi
import glob
import pandas as pd

def cal_rect(contours_xy):
    cnt = contours_xy
    
    rect = cv2.minAreaRect(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box, x,y,w,h

def cal_distrib(gamma_prime):
    distrib = None

    if gamma_prime <= 2538:
        distrib = 0
    elif 2538 < gamma_prime and gamma_prime <=7614:
        distrib = 1
    elif 7614 < gamma_prime and gamma_prime <=25380:
        distrib = 2
    elif 25380 < gamma_prime and gamma_prime <=76142:
        distrib = 3
    elif 76142 < gamma_prime:
        distrib = 4            
    return distrib

def cal_width(img, gamma_prime, b):
    h, w, _ = img.shape
    return b*gamma_prime/(h*w)

def cal_circle(img, gamma_prime):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = len(contour[0])
    
#     img3 = cv2.drawContours(img1, [cnt], 0, (0,255,0), 1)    
    
    return (4*pi*gamma_prime)/(perimeter*perimeter)

def main(args):
    img_list = glob.glob(args.pred_path+'/*.*')

    name_list = []
    aspect_list = []
    avg_width_list = []
    avg_cir_list = []
    area_list = []
    gamma_phase_list = []
    gammaP_phase_list = []
    for n in img_list:
        name = n.split('/')[-1]
        name_list.append(name)

        name_ = name[:-4]
        
        save_path = './' + name_ + '/'

        img = cv2.imread(n)
        img_cp = img.copy()
        imgray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
        contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        img_h, img_w, _  = img.shape
        img_flatten = imgray.flatten()
        gamma = len(np.where(img_flatten==255)[0])
        gamma_prime = len(np.where(img_flatten==0)[0])

        trims_coord = []
        gamma_size_distrib = []
        gamma_ratios = []
        gamma_bboxes=[]
        gamma_W= []
        gamma_circle = []
        gamma_size = []
        # img_h, img_w, _ = img.shape
        for i in range(len(contour)):

            contours_xy = np.array(contour[i])
            box, x, y, w, h = cal_rect(contours_xy)

            img_rect = cv2.drawContours(img_cp,[box],0,(0,0,255),2)
            img_tmp = np.ones((img_h, img_w, 3), dtype=np.uint8)*255

            trim_fill_img = cv2.fillPoly(img_tmp,[contours_xy], color = (0,0,0))
            img_trim = trim_fill_img[y:y+h, x:x+w]

            trim_gray = cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY)
            trim_flatten = trim_gray.flatten()
            trim_gamma_prime = len(np.where(trim_flatten==0)[0])
            gamma_size.append(trim_gamma_prime)


            distrib = cal_distrib(trim_gamma_prime)
            gamma_size_distrib.append(distrib)


            trim_h, trim_w, _ = img_trim.shape
            if trim_h > trim_w:
                a = trim_h
                b = trim_w
            else:
                a = trim_w
                b = trim_h

            trim_ratio = a/b

            gamma_bboxes.append(b*(trim_gamma_prime/(trim_h*trim_w)))
            gamma_ratios.append(trim_ratio*trim_gamma_prime)

            trimW = cal_width(img_trim, trim_gamma_prime, b)     
            gamma_W.append(trimW*trim_gamma_prime)

            trimC = cal_circle(img_trim, trim_gamma_prime)
            gamma_circle.append(trimC*trim_gamma_prime)

        
        Aspect_ratio = 0
        avg_width = 0
        avg_cir = 0
        area_0, area_1, area_2, area_3, area_4 = 0, 0, 0, 0, 0
        for k in range(len(gamma_size_distrib)):
            Aspect_ratio += gamma_ratios[k]
            avg_width += gamma_W[k]
            avg_cir += gamma_circle[k]

            if gamma_size_distrib[k] == 0:
                area_0 += 1
            elif gamma_size_distrib[k] == 1:
                area_1 += 1
            elif gamma_size_distrib[k] == 2:
                area_2 += 1
            elif gamma_size_distrib[k] == 3:
                area_3 += 1
            else: area_4+= 1
                
        gamma_phase_list.append((gamma/len(img_flatten))*100)
        gammaP_phase_list.append((gamma_prime/len(img_flatten))*100)

        if len(gamma_size_distrib) == 0:
            area_0 = None
            area_1 = None
            area_2 = None
            area_3 = None
            area_4 = None
        else:
            area_0 = round(area_0/len(gamma_size_distrib)*100,3)
            area_1 = round(area_1/len(gamma_size_distrib)*100,3)
            area_2 = round(area_2/len(gamma_size_distrib)*100,3)
            area_3 = round(area_3/len(gamma_size_distrib)*100,3)
            area_4 = round(area_4/len(gamma_size_distrib)*100,3)
        area_list.append([area_0, area_1, area_2, area_3, area_4])

        if gamma_prime == 0:
            aspect_list.append(None)
        else:
            aspect_list.append(Aspect_ratio/gamma_prime)
        if gamma_prime == 0:
            avg_width_list.append(None)
        else:
            avg_width_list.append(avg_width/gamma_prime)
        if gamma_prime == 0:
            avg_cir_list.append(None)
        else:
            avg_cir_list.append(avg_cir/gamma_prime)

    
    df = pd.DataFrame((zip(name_list, gamma_phase_list, gammaP_phase_list, area_list, aspect_list, avg_width_list, avg_cir_list)), columns = ['Name', 'gamma', 'gammaP', 'gammaP_distrib', 'gammaP_aspect', 'gammaP_width', 'gammaP_circle'])
    print(df)
    df.to_csv(os.path.join(args.save_path, 'features.csv'), header = None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='prediction features extraction')

    parser.add_argument('--pred_path', type=str, default='./')
    parser.add_argument('--save_apth', type=str, default='./')
    
    args = parser.parse_args()

    main(args)

    
            