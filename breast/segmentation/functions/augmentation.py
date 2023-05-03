#### 'augmentation.py' module is used to perform Data Augmentation and/or return a Generator used in 
#### 'model.fit()' and 'model.evaluate()'

from keras.preprocessing.image import ImageDataGenerator

def train_data_aug(x_train, y_train):
    """Peforms real-time Data Augmentation on the Training dataset used in 'model.fit_generator()'

    Args:
        x_train , y_train both are numpy arrays of the images

    Returns:
        iterator: Single generator "zipped" to maintain mapping of train image and corresponding mask 
            after augmentation
    """    
    seed = 42 # To ensure correct mapping of train images to corresponding mask, otherwise ordering  
             # of images and masks aren't consistent 

    # Define what augmentation should take place
    image_datagen = ImageDataGenerator(rotation_range=0.2,rescale=1./255, width_shift_range=0.05, 
                    height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                    horizontal_flip=True, fill_mode='nearest') 
    mask_datagen = ImageDataGenerator(rotation_range=0.2,rescale=1./255 , width_shift_range=0.05,
                    height_shift_range=0.05, shear_range=0.05, zoom_range=0.05,
                    horizontal_flip=True, fill_mode='nearest')  
    
    
    image_generator =image_datagen.flow(x_train.astype('float32'),seed=seed, batch_size=20)
    mask_generator = mask_datagen.flow(y_train.astype('float32'), seed=seed, batch_size=20)

    train_generator = zip(image_generator, mask_generator) # Zipped to ensure correct ordering/mapping of images
    return train_generator
    
def test_data_aug(x_test, y_test):
    """Peforms real-time Data Augmentation on the Test/Validation dataset used in 'model.evaluate_generator()'

    Args:
        x_test , y_test both are numpy arrays of the images

    Returns:
        iterator: Single generator "zipped" to maintain mapping of test image and corresponding mask 
            after augmentation
    """ 
    seed = 42 # To ensure correct mapping, same as above
    image_datagen1 = ImageDataGenerator(rescale=1./255) # Required only rescaling as we're testing here thus 
                                                        # no augmentation
    mask_datagen1 = ImageDataGenerator(rescale=1./255)  

    image_generator1 =image_datagen1.flow(x_test.astype('float32'), shuffle=False, seed=seed, batch_size=10)
    mask_generator1 = mask_datagen1.flow(y_test.astype('float32'), shuffle=False, seed=seed, batch_size=10)

    test_generator = zip(image_generator1, mask_generator1) # Zipped to ensure correct ordering/mapping of images
                                                            # and corresponding masks
    return test_generator