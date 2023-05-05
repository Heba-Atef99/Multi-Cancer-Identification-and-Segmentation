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
For the breast tumor segmentation problem we used the selective kernel U-Net which is a novel variation of the U-Net model.

Standard U-Nets utilize convolutions of the fixed receptive field.
The segmentation method proposed in the proposed study is based on selective kernels (SKs) that can automatically adjust the network’s receptive field via an attention mechanism, and mix feature maps extracted with both dilated and conventional convolutions

### SK-U-Net architecture

![image](https://user-images.githubusercontent.com/54477107/236578445-c6bc2077-c191-4ef7-924a-343379e427d9.png)


The SK-U-Net architecture was based on the U-Net architecture, with conventional convolution/batch normalization/activation function blocks replaced with SK blocks.

The aim of each SK block was to adaptively adjust the network's receptive field and mix feature maps determined using different convolutions, effectively addressing the problem of large variations of breast mass sizes.

Each SK block included two branches:
- The first one was based on convolutions with dilation size equal to 2 and 3x3 kernels filters
- The second one utilized 3 × 3 kernel filters with no dilation.

The resulting feature maps were summed, and global average pooling was applied to convert feature maps to a single feature vector.

Next, the vector was compressed using a fully connected layer, with compression ratio equal to 0.5 (the number of features was halved).

Compressed feature vector was decompressed with a fully connected layer and processed with a sigmoid activation function to determine attention coefficients for each feature map.

Next, the obtained attention coefficients were used to weight the feature maps and calculate output of the SK block, using:

![image](https://user-images.githubusercontent.com/54477107/236578504-10c3fa29-7373-4d0e-b618-8f097a9e0129.png)

### Transfer Learning
We applied transfer learning in this problem using the pre-trained weights of the sk-u-net model and started training the model on the train dataset for 100 epochs.

### Data Augmentation
Because of the small size of the data we used data augmentation (rotation, rescale, width shift, height shift, sheat, zoom, horizontal flip) which was applied to the image and its mask.

### Loss Function
We chose the dice coefficient loss. The Dice coefficient is a measure of overlap between two sets, and it is defined as twice the size of the intersection of the two sets divided by the sum of the sizes of the two sets.
In the context of image segmentation, the sets correspond to the ground truth segmentation of the image and the predicted segmentation produced by the model.
The  Dice coefficient loss is a measure of the dissimilarity between the predicted segmentation and the ground truth segmentation, and it is defined as 1 minus the Dice coefficient. The loss function penalizes the model for producing segmentations that deviate from the ground truth segmentation, with higher penalties for larger deviations.

### Results
|Metric         |Test set|
|---------------|:----|
|Dice Loss     |0.038 |
|IoU    |0.55 |
|Dice Coefficient    |0.962 |

![image](https://user-images.githubusercontent.com/54477107/236578711-2877a586-7e5a-4b2e-9316-843de591a07f.png)
