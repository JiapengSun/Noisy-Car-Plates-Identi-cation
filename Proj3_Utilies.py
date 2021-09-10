import numpy as np
import cv2
import random

###########################################
# Salt and pepper noise function
###########################################
def sp_noise(img, proportion):
    height, width = img.shape[0], img.shape[1]
    num = int(height * width * proportion)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            img[h, w] = 0
        else:
            img[h, w] = 255
    return img


###########################################
# Gaussian noise function
###########################################
def gaussian_noise(img, mean, sigma):
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out


###########################################
# Get aligned car plate from prediction
###########################################
def plate_alignment(img,y_p):

    img = (img*255).astype('uint8')

    y_p = (y_p*255).astype('uint8')
    ret,y_p = cv2.threshold(y_p,127,255,cv2.THRESH_BINARY)
    y_p_mask = np.array(y_p)

    contours, hierarchy = cv2.findContours(y_p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    flag = 0
    aligned_plate = img
    unaligned_plate = img
    box_cont = []


    if len(contours):
        flag = 1
        cont = contours[0]

        x, y, w, h = cv2.boundingRect(cont)  
        unaligned_plate = img[y:y+h,x:x+w]
    
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect).astype(np.int32)
    
        cont_temp = cont.reshape(len(cont),2)
        for b in box:
            min_dist = 100000
            for p in cont_temp:
                d = (b[0]-p[0])**2 + (b[1]-p[1])**2
                if d < min_dist:
                    min_dist = d
                    pick = p
            box_cont.append(pick)
            
        p = np.array(box_cont)
        p = p[np.argsort(p[:,0])]
        p_l = p[:2]
        p_r = p[2:]
        p_lu,p_ld = p_l[np.argsort(p_l[:,1])]
        p_ru,p_rd = p_r[np.argsort(p_r[:,1])]
        
        box_from = np.float32([p_lu,p_ld,p_ru,p_rd])
        box_to = np.float32([(0,0),(0,80),(240,0),(240,80)])
        trans_mat=cv2.getPerspectiveTransform(box_from,box_to)
        aligned_plate=cv2.warpPerspective(img,trans_mat,(240,80))

    return aligned_plate, unaligned_plate, box_cont, flag

