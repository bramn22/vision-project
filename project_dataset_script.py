from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle


# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = r"C:\Users\bramn\Documents\MAI\CompVision\VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)
# power of 2 - easy for convnet

# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

# step2 - build (x,y) for TRAIN/VAL (classification)
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for c_f in classes_files for filt in filter if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for c_f in classes_files for filt in filter if filt in c_f and '_val.txt' in c_f]


def build_classification_dataset(list_of_files, gray=False):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, )
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        print('Building classification dataset -', f_cf)
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]
    y = np.array([item for l in train_labels for item in l])

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for file in os.listdir(image_folder) for f in train_filter if
                       f in file]
    if gray:
        resize_shape = (image_size, image_size)
    else:
        resize_shape = (image_size, image_size, 3)
    x = np.array([resize(io.imread(img_f, as_gray=gray), resize_shape) for img_f in image_filenames]).astype(
        'float32')
    return x, y


x_train, y_train = build_classification_dataset(train_files, gray=True)
print('%i training images from %i classes' %(x_train.shape[0], len(np.unique(y_train))))
x_val, y_val = build_classification_dataset(val_files, gray=True)
print('%i validation images from %i classes' %(x_val.shape[0], len(np.unique(y_val))))
pickle.dump({"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}, open("data_gray.p", "wb"))
print('Pickle file saved')
# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)