# Multi-Cancer-Identification-and-Segmentation
This project is focused on developing a model to classify and segment multi-cancer scans. The dataset contains images of brain and breast scans, where the brain scans are labeled as "tumor" or "no tumor", and the breast scans are labeled as "malignant", "normal", or "benign". 

# Project Pipeline
![Mind Maps - Multi-Cancer Classification  amp; Segmentation Pipeline](https://user-images.githubusercontent.com/54477107/236575716-d463153f-c12a-46cf-858c-31f1403b5e42.jpg)

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
For the Brain Tumor segmentation, the dataset consisted of Images and their corresponding masks that show where the tumor is.

![image](https://user-images.githubusercontent.com/54477107/236577626-fd8c6580-40e4-426d-afe2-1f50257fb8da.png)


we tried DeepLabv3 which is a semantic segmentation architecture that improves upon DeepLabv2 with several modifications. To handle the problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates.

First, the input image goes through the network with the use of dilated convolutions. Then the output from the network is bilinearly interpolated and goes through the fully connected CRF to fine-tune the result we obtain the final predictions.

![image](https://user-images.githubusercontent.com/54477107/236577419-348f3c38-6312-4328-9259-8f118fff718a.png)

### Results
|Metric         |Dice Loss|Dice Coefficient|
|---------------|:----|:----|
|Train set      |0.231 |0.81 |
|Validation set    |0.40 |0.63 |
|Test set     |0.599 |0.562 |

## Breast Segmentation

# Results
