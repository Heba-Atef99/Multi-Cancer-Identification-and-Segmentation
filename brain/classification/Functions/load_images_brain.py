from image_preprocessing_brain import image_preprocessing
import cv2
import os


def load_images(data_path):
    # Define the classes
    classes = ['No tumor', 'Tumor']

    # Create empty lists to store the images and labels
    images = []
    labels = []

    # Loop over the subfolders to load the data
    for subfolder in ['Train', 'Test', 'TRAIN', 'TEST']:
        if subfolder == 'Train' or subfolder == 'Test':
            subfolder_path = os.path.join(data_path, classes[0], subfolder)
        else:
            subfolder_path = os.path.join(data_path, classes[1], subfolder)

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            # print(image_path)

            if image_name.startswith(('y', 'Y')):
                labels.append(1)
            elif image_name.startswith(('no', 'No')):
                labels.append(0)

            # Preprocess the image
            image = image_preprocessing(image_path)

            # resize image
            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)

            # Append the image and label to the lists
            images.append(image)

    return images, labels


def read_images_from_dataFrame(data):
    images = []
    for path in data.images:
        # apply preprocessing in each image
        image = image_preprocessing(path)

        # resize image
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)

        # Append the image and label to the lists
        images.append(image)

    return images

