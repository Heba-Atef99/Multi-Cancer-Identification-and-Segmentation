# Multi-Cancer-Identification-and-Segmentation
This project is focused on developing a model to classify and segment multi-cancer scans. The dataset contains images of brain and breast scans, where the brain scans are labeled as "tumor" or "no tumor", and the breast scans are labeled as "malignant", "normal", or "benign". 

# Project Objectives
1. Image Classification:
      - Classify each image as either a "Brain" or "Breast" scan using a convolutional neural network (CNN) model.
2. Brain Tumor Classification: 
      - For brain scans, classify each image as "tumor" or "no tumor". <br>
3. Breast Cancer Classification:
      - For breast scans, classify each image into one of the three classes - "malignant", "normal", or "benign". <br>
4. Image Segmentation:
      - Segment the injured part of each image using the U-Net architecture, which is a convolutional neural network designed for image segmentation.<br>

We used popular machine learning libraries like TensorFlow and Keras. The detailed implementation and experimental results are provided in the notebook located in the project repository.

# Image Preprocessing
- Preprocessing is a very important and main step in image processing, as scanned images are usually displayed in gray scale or color and also they suffer from noise.
- Noise removal and edge detection are the two most important steps in processing any digital images to improve the information in the picture so that it can be easily understood and to make it suitable and readable for any machine working on those images.
- That's why, some preprocessing steps are performed on the image.

## Brain Images Preprocessing steps
![Cropping-process-of-MR-images](https://user-images.githubusercontent.com/43891138/236048765-0e436f33-2dd1-495c-b817-d63a725c538d.jpg)

# Image Segmentation
## Brain Segmentation

## Breast Segmentation

# Results
