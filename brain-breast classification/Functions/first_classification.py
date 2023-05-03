import os
import cv2
import random

def load_images_from_folder(folder,train=True,test=False):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        #print(subdir_path)
        if os.path.isdir(subdir_path):
          if train:
            for filename in os.listdir(os.path.join(subdir_path,'Train')):
                #print(filename)
                img_path = os.path.join(subdir_path,'Train', filename)
                #print(img_path)
                if os.path.isfile(img_path) and "mask" not in img_path:
                    # Extract the label based on the parent directory name
                    if "Brain" in folder:
                        label = "brain" 
                    elif "Breast" in folder:
                        label = "breast" 
                    if label is not None:
                        # Load the image and append it and its label to the lists
                        image = cv2.imread(img_path)
                        images.append(image)
                        labels.append(label)
          elif test:
              for filename in os.listdir(os.path.join(subdir_path,'Test')):
                #print(filename)
                img_path = os.path.join(subdir_path,'Test', filename)
                print(img_path)
                if os.path.isfile(img_path) and "mask" not in img_path:
                    # Extract the label based on the parent directory name
                    if "Brain" in folder:
                        label = "brain" #if subdir in ["tumer", "no tumer"] else None
                    elif "Breast" in folder:
                        label = "breast" #if subdir in ["malignent", "benign", "normal"] else None
                    if label is not None:
                        # Load the image and append it and its label to the lists
                        image = cv2.imread(img_path)
                        images.append(image)
                        labels.append(label)

    return images, labels


def pipeline(brain_images,brain_labels,breast_images,breast_labels):
  # Combine the brain and breast images and labels into a single dataset
  images = brain_images + breast_images
  labels = brain_labels + breast_labels
  print('Length of Images = ',len(images))
  print('Length of Labels = ',len(labels))


  # Combine the images and labels into a single list of tuples
  data = list(zip(images, labels))

  # Shuffle the list of tuples
  random.shuffle(data)

  # Separate the shuffled images and labels into separate lists
  shuffled_images, shuffled_labels = zip(*data)

  print('Size of images before Resizing: ',shuffled_images[0].shape)

  resized_images = []
  for image in shuffled_images:
      # Resize the image to have a fixed shape of (150, 150, 3)
      resized_image = cv2.resize(image, (150, 150))
      # Append the resized image to the list
      resized_images.append(resized_image)
  
  print('Size of images after Resizing: ',resized_images[0].shape)

  # Create a dictionary that maps label strings to integer labels
  label_dict = {"brain": 0, "breast": 1}

  # Convert the original string labels to integer labels using the label_dict
  integer_labels = [label_dict[label] for label in shuffled_labels]

  return resized_images,integer_labels
