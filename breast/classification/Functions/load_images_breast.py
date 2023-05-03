import os
import cv2


def load_images_breast(data_path):
  # Define the classes
  classes = ['benign', 'malignant', 'normal']

  # Create empty lists to store the images and labels
  images = []
  labels = []

  # Loop over the subfolders to load the data
  for class_id, class_name in enumerate(classes):
    for subfolder in ['Train', 'Test']:
      subfolder_path = os.path.join(data_path, class_name, subfolder)

      for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        # print(image_path)
        if "mask" not in image_path:
          # load the image
          image = cv2.imread(image_path)

          # Preprocess the image
          # image = image_preprocessing(image_path)[0]

          # resize image
          image = cv2.resize(image, (150,150), interpolation=cv2.INTER_LINEAR)
          blurred = cv2.GaussianBlur(image, (5,5), 0)

          # Append the image and label to the lists
          images.append(blurred)
          labels.append(class_id)

  return images, labels