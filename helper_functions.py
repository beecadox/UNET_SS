import os
import cv2
import numpy as np
from sklearn import preprocessing
import keras


# load images from folder and return two dictionaries (train-validation-test) that hold all images category by category
def load_images_from_folder(folder_name, image_size):
    print("Reading Dataset\n------------------------------")
    category_images = []
    category_labels = []
    le = preprocessing.LabelEncoder()
    for each_category in os.listdir(folder_name):
        path = folder_name + "/" + each_category
        print('\tCategory..', each_category, '\tparsing images..')
        for image in os.listdir(path):
            img = cv2.imread(path + "/" + image)
            if img is not None:
                #  resize it to the given dimensions
                img = cv2.resize(img, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
                category_images.append(img)
                category_labels.append(each_category)
    category_labels = le.fit_transform(category_labels)
    return np.array(category_images), np.array(category_labels)


# loading images from folder with images and folder with masks.in case of a mask/image not having a pair stop loading
def load_images_and_masks(folder_name, image_size):
    print("Reading Dataset\n------------------------------")
    original_images = []
    masked_labels = []
    path1 = folder_name + "/images"
    path2 = folder_name + "/masks"
    for image_name in os.listdir(path1):
        img = cv2.imread(path1 + "/" + image_name)
        if img is not None:
            #  resize it to the given dimensions
            img = resize_image(img, image_size)
            mask = cv2.imread(path2 + "/" + image_name.rsplit('.')[0] + "_seg0.png", 0)
            if mask is None:
                print(' .. unable to load the corresponding mask')
                break
            else:
                mask = resize_image(mask, image_size)
                mask[mask < 1] = 0
                mask[mask >= 1] = 1
                # convert mask labels to binarize vectors. Here we know that we have two classes
                mask = keras.utils.to_categorical(mask, 2)
                original_images.append(img)
                masked_labels.append(mask)
        else:
            print(' .. unable to load the image')
            break
    return np.array(original_images), np.array(masked_labels)


# resize image to given size
def resize_image(image, image_size):
    dimensions = image.shape
    width_size_check = image.shape[1] - image_size[1]
    height_size_check = image.shape[0] - image_size[0]
    if len(dimensions) == 3:
        num_of_channels_check = image.shape[2] - image_size[2]
    else:
        num_of_channels_check = 0
    if (width_size_check == 0) & (height_size_check == 0) & (num_of_channels_check == 0):
        print(' ... mask has the correct size')
    else:
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    return image



