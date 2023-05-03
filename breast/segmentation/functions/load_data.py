import numpy as np
import cv2
import os
import random

def compatible_img(img, d_size=(224, 224)):
  img = cv2.resize(img, d_size, interpolation = cv2.INTER_AREA)
  img = np.array(img, dtype=np.float32).reshape(d_size[0], d_size[1], 1)
  return img

def load_images_with_masks_from_folder(directory, train=True):
    images = []
    masks = []

    if 'Brain' in directory:
        types = ['Tumor']
    elif 'Breast' in directory:
        types = ['benign', 'malignant']

    if train:
        subdir = 'Train'
        mask_subdir = 'TRAIN_masks'
    else:
        subdir = 'Test'
        mask_subdir = 'TEST_masks'

    for type_ in types:
        print(type_)
        folder = directory + '/' + type_
        subdir_path = os.path.join(folder, subdir)
        mask_subdir_path = os.path.join(folder, mask_subdir)
        print(subdir_path)
        
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if ('Breast' in directory and filename.endswith(').png')) or 'Brain' in directory:
                    img_path = os.path.join(subdir_path, filename)
                    if 'Brain' in directory:
                        mask_img_path = os.path.join(mask_subdir_path, filename[:-3]+'png')
                        img = cv2.imread(img_path)
                        mask = cv2.imread(mask_img_path)
                    elif 'Breast' in directory:
                        mask_img_path = os.path.join(subdir_path, filename[:-4]+'_mask'+filename[-4:])
                        mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                          img = compatible_img(img)
                          mask = compatible_img(mask)
                          
                    if mask is not None:
                      images.append(img)
                      masks.append(mask)
    return images, masks

def shuffle_data(images, masks):
  temp = list(zip(images, masks))
  random.shuffle(temp)
  res1, res2 = zip(*temp)
  res1, res2 = list(res1), list(res2)
  return res1, res2