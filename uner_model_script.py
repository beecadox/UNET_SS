# -------------------------
# needed imports
# ---------------------------
import cv2
import numpy as np
import random
from helper_functions import load_images_and_masks
from matplotlib import pyplot as plt
from openpyxl.workbook import Workbook
from openpyxl.styles import Font
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split as split
from keras.utils.vis_utils import plot_model  # pip install pydot or conda install -c anaconda pydot
from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization, Input, UpSampling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import tensorflow as tf

# Allowing GPU memory growth
# in case of using normal tensorflow and not tensorflow-gpu comment out these two lines
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# -------------------------
# used functions
# ---------------------------

# 2 Conv2D Relu Layers
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# function used for fitting the unet model
def fitting_unet_model(image_size, baseNumOfFilters, batch_size, epochs, model_name, train_x, val_x, train_y, val_y):
    inputs = Input((image_size[0], image_size[1], image_size[2]))
    conv1 = conv2d_block(inputs, n_filters=baseNumOfFilters * 1, kernel_size=3, batchnorm=True)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d_block(pool1, n_filters=baseNumOfFilters * 2, kernel_size=3, batchnorm=True)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d_block(pool2, n_filters=baseNumOfFilters * 4, kernel_size=3, batchnorm=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d_block(pool3, n_filters=baseNumOfFilters * 8, kernel_size=3, batchnorm=True)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv2d_block(pool4, n_filters=baseNumOfFilters * 16, kernel_size=3, batchnorm=True)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(baseNumOfFilters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = conv2d_block(merge6, n_filters=baseNumOfFilters * 8, kernel_size=3, batchnorm=True)

    up7 = Conv2D(baseNumOfFilters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv2d_block(merge7, n_filters=baseNumOfFilters * 4, kernel_size=3, batchnorm=True)

    up8 = Conv2D(baseNumOfFilters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv2d_block(merge8, n_filters=baseNumOfFilters * 2, kernel_size=3, batchnorm=True)

    up9 = Conv2D(baseNumOfFilters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv2d_block(merge9, n_filters=baseNumOfFilters, kernel_size=3, batchnorm=True)

    outputs = Conv2D(2, (1, 1), padding='same', activation='sigmoid')(conv9)

    UnetCustMod = Model(inputs=[inputs], outputs=[outputs])
    UnetCustMod.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    UnetCustMod.summary()
    # train the model
    callbacksOptions = [
        EarlyStopping(patience=15, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
        ModelCheckpoint('OutputFiles/Models/tmpUnet.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    UnetCustMod.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacksOptions,
                    validation_data=(val_x, val_y))
    # save the final model, once training is completed
    UnetCustMod.save(model_name)
    # UNET model architecture
    print("Saving Unet Model Architecture\n")
    plot_model(UnetCustMod, to_file='OutputFiles/Models/model.png', show_shapes=True)


# function used for predicting labels with a cnn model based on images
def predict_scores(x, model):
    y_train_predictions_vectorized = model.predict(x)
    return y_train_predictions_vectorized


# plot of original image, original mask and estimated mask
def show_masked_image(im1, im2, im3, title):
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.suptitle("Original Image - Mask - Estimated Mask")
    plt.show()
    plt.savefig("OutputFiles/OutputImages/" +title + ".png")
    plt.close()


# converting y to appropriate type
def convert_y(y):
    return np.argmax(y, axis=3).astype(np.uint8)


# showing 6 random plots
def show_random_results(test_x, test_y, pred_y):
    pred_y = convert_y(pred_y)*255
    test_y = convert_y(test_y)*255
    # for shuffling each of the 3 arrays
    map_index_position = list(zip(test_x, test_y, pred_y))
    random.shuffle(map_index_position)
    test_x, test_y, pred_y = zip(*map_index_position)
    for i in range(6):
        title = "fig_" + str(i)
        show_masked_image(test_x[i], test_y[i], pred_y[i], title)


# function that saves all the results in an excel
def scores_excel_file(train_y, y_pred_train, test_y, y_pred_test):
    train_y = convert_y(train_y)
    y_pred_train = convert_y(y_pred_train)
    test_y = convert_y(test_y)
    y_pred_test = convert_y(y_pred_test)
    # calculate the scores
    acc_train = accuracy_score(train_y.flatten(), y_pred_train.flatten())
    acc_test = accuracy_score(test_y.flatten(), y_pred_test.flatten())
    pre_train = precision_score(train_y.flatten(), y_pred_train.flatten())
    pre_test = precision_score(test_y.flatten(), y_pred_test.flatten())
    rec_train = recall_score(train_y.flatten(), y_pred_train.flatten())
    rec_test = recall_score(test_y.flatten(), y_pred_test.flatten())
    f1_train = f1_score(train_y.flatten(), y_pred_train.flatten())
    f1_test = f1_score(test_y.flatten(), y_pred_test.flatten())
    data_for_xls = [["Train", str(acc_train), str(pre_train), str(rec_train), str(f1_train)],
                    ["Test", str(acc_test), str(pre_test), str(rec_test), str(f1_test)]]
    for data in data_for_xls:
        sheet.append(data)

    print('Accuracy scores of UNET model are:',
          'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
    print('Precision scores of UNET model are:',
          'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
    print('Recall scores of UNET model are:',
          'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
    print('F1 scores of UNET model are:',
          'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
    print('')


# Excel file info - headers - name of the sheet etc
headers = ['Train/Test', 'Accuracy', ' Precision', 'Recall', 'F1 score']
workbook_name = 'OutputFiles/Results/unet_scores.xls'
wb = Workbook()
sheet = wb.active
sheet.title = 'Semantic Scores'
sheet.append(headers)
row = sheet.row_dimensions[1]
row.font = Font(bold=True)

input_folder = 'M:/Datasets/Butterflies'
input_image_size = [128, 128, 3]
num_of_filters = 16
batch = 60
epoch = 20
test_size = 0.2
val_size = 0.05
model_name = 'OutputFiles/Models/butterflyUNET.h5'
images, labels = load_images_and_masks(input_folder, input_image_size)
print("Dataset Loading done!")
print("Dataset split")
xx, x_test, yy, y_test = split(images, labels, test_size=test_size, random_state=3)
x_train, x_val, y_train, y_val = split(xx, yy, test_size=val_size, random_state=4)

fitting_unet_model(input_image_size, num_of_filters, batch, epoch, model_name, x_train, x_val, y_train, y_val)
unet_model = load_model(model_name)
y_pred_train = predict_scores(x_train, unet_model)
y_pred_test = predict_scores(x_test, unet_model)
show_random_results(x_test, y_test, y_pred_test)
scores_excel_file(y_train, y_pred_train, y_test, y_pred_test)
wb.save(filename=workbook_name)
