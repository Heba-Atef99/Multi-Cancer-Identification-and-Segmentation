#### 'plots.py' module is used to: 
####	1. Plot "training curve" of the network/model(for various metrics as given in 'titles' below)  
####	2. Show for images- model's segmentation prediction, obtained "Binary mask" and "Ground Truth" segmentation

from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt, random, numpy as np, cv2
from PIL import Image
from keras import backend as K

def training_history_plot(results):
	"""Plots "training curve" for the network/model for metrics listed below:
    		1. Dice loss
    		2. Pixel-wise accuracy 
    		3. Intersection Over Union(IOU)
    		4. F1 score
    		5. Recall
    		6. Precision

    Args:
        results (History): Output of 'model.fit_generator()', 'History.history' attribute is a record of metrics
        					values as described above(from 1-6)

    Returns:
        None
	"""
	titles = ['Dice Loss','Accuracy','IOU','F1','Recall','Precision'] 
	metric = ['loss', 'accuracy', 'iou','F1','recall','precision'] # Metrics we're keeping track off

	# Define specification of our plot
	fig, axs = plt.subplots(3,2, figsize=(15, 15), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = 0.5, wspace=0.2)
	axs = axs.ravel()

	for i in range(6):
		axs[i].plot(results.history[metric[i]]) # Calls from 'History.history'- 'metric[i]', note 'results' is 
		axs[i].set_title(titles[i])				# a 'History' object
		axs[i].set_xlabel('epoch')  
		axs[i].set_ylabel(metric[i])
		axs[i].legend(['train'], loc='upper left')

def plot_masks(train_model, breast_images_test_arr, breast_masks_test):
  for i in range(breast_images_test_arr.shape[0]):
    img = breast_images_test_arr[i]
    img = np.expand_dims(img, 0)
    mask_predicted = train_model.predict(img/255.0).squeeze().round()
    plot_one_mask(breast_images_test_arr[i], mask_predicted)

def plot_one_mask(image, mask_predicted):
  image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
  overlay = image.copy()
  overlay[mask_predicted.astype(int)==1] = (255,36,12)
  alpha = 0.2  
  image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

  cv2_imshow(image_new)

def plot_prediction(train_model, breast_images_test_arr, breast_masks_test, dice_coef_np):
  for i in range(breast_images_test_arr.shape[0]):
    img = breast_images_test_arr[i]
    img = np.expand_dims(img, 0)
    mask_predicted = train_model.predict(img/255.0).squeeze().round()
    plot_one_prediction(img, breast_masks_test[i], mask_predicted, dice_coef_np)

def plot_one_prediction(img,mask_actual, mask_predicted, dice_coef_np):
  fig = plt.figure()
  ax1 = fig.add_subplot(131)
  ax2 = fig.add_subplot(132)
  ax3 = fig.add_subplot(133)
  ax1.title.set_text('Original Image')
  ax2.title.set_text('Original Mask')
  ax3.title.set_text('Predicted Mask')

  plt.subplot(131)
  plt.imshow(np.round(img.squeeze()), cmap='gray')

  plt.subplot(132)
  plt.imshow(mask_actual, cmap='gray')

  plt.subplot(133)
  plt.imshow(mask_predicted, cmap='gray')

  plt.show()
  print('Dice score:', dice_coef_np(mask_predicted, mask_actual).round(3))

def model_prediction_plot(results, x_test, y_test, t=0.2):
  """Displays:
        1. Original test image  
        2. Network's predicted segmentation mask 
        3. Binary mask obtained from 2
        4. Ground truth segmentation for the test image

    Args:
        results (numpy.array): Numpy array of shape (17,255,255,1)- 17 predicted segmentation mask, each of size
                    (255,255,1)
        t (float)(Default=0.2): Threshold used to convert predicted mask to binary mask
        x_test: a numpy array of the test images
        y_test: a numpy array of the test masks

    Returns:
        None
  """
  bin_result = (results >= t) * 255 # Convert predicted segmentation mask to binary mask on threshold 't'
  titles=['Image','Predicted Mask','Binary Mask','Ground Truth']
  r=random.sample(range(17),4) # Random sample for test images to display

  # Define specification of our plot
  fig, axs = plt.subplots(4, 4, figsize=(15, 15), facecolor='w', edgecolor='k')
  fig.subplots_adjust(hspace = 0.5, wspace=0.2)
  axs = axs.ravel()

  for i in range(4):# 1 iteration for each selected test image
    # Displays test image 
    axs[(i*4)+0].set_title(titles[0])
    arr = x_test[i]
    axs[(i*4)+0].imshow(arr/255, cmap='gray')

    # Displays predicted segmentation mask
    axs[(i*4)+1].set_title(titles[1])
    I=np.squeeze(results[r[i],:,:,:])
    axs[(i*4)+1].imshow(I, cmap="gray")

    # Displays binary mask
    axs[(i*4)+2].set_title(titles[2])
    I=np.squeeze(bin_result[r[i],:,:,:])
    axs[(i*4)+2].imshow(I, cmap="gray")

    # Displays Ground truth segmentation mask 
    axs[(i*4)+3].set_title(titles[3])
    arr = y_test[i]
    axs[(i*4)+3].imshow(arr/255, cmap='gray')
