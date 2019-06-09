from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
import pickle
import utils
import cv2
from skimage.color import rgba2rgb


# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = r"C:\Users\bramn\Documents\MAI\CompVision\VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)

def build_classification_dataset(list_of_files, gray=False):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    # train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            # label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            # train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    if gray:
        resize_shape = (image_size, image_size)
    else:
        resize_shape = (image_size, image_size, 3)
    x = np.array([resize(io.imread(img_f, as_gray=gray), resize_shape) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y

def build_segmentation_dataset(list_of_files, gray=False):
    temp = []
    # train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines])
            # label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            # train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    if gray:
        resize_shape = (image_size, image_size)
    else:
        resize_shape = (image_size, image_size, 3)
    x = np.array([resize(io.imread(img_f, as_gray=gray), resize_shape) for img_f in image_filenames]).astype(
        'float32')

    segment_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass/")
    segment_filenames = [os.path.join(segment_folder, file) for f in train_filter for file in os.listdir(segment_folder) if
                       f in file]
    resize_shape = (image_size, image_size, 3)
    y = np.array([resize(rgba2rgb(io.imread(img_f)), resize_shape) for img_f in segment_filenames]).astype(
        'float32')

    return x, y

''' ------------------ Classification ------------------ '''
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]

x_train, y_train = build_classification_dataset(train_files, gray=True)
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
utils.disp_images(x_train[-5:], y_train[-5:], title="disp", cmap='gray')

x_val, y_val = build_classification_dataset(val_files, gray=True)
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))
pickle.dump({"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}, open(r"pickles/classification_gray.p", "wb"))
print('Pickle file saved')


''' ------------------ Segmentation ------------------ '''
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for c_f in classes_files if 'train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for c_f in classes_files if 'val.txt' in c_f]

x_train, y_train = build_segmentation_dataset(train_files)
print('{} training images with shapes: {}'.format(x_train.shape[0], x_train.shape))
utils.disp_images(np.concatenate([x_train[-5:], y_train[-5:]]), title="disp", cols=5)

x_val, y_val = build_segmentation_dataset(val_files)
print('{} validation images with shapes: {}'.format(x_val.shape[0], y_val.shape))
pickle.dump({"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}, open(r"pickles/segmentation.p", "wb"))
print('Pickle file saved')


