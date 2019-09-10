import gdal
import numpy as np
from gdalconst import GA_ReadOnly
import keras.backend as K
import skimage

def read_tif(filename):
    """
    reads a tif file to numpy array and returns
    :param filename: file to read
    :return: numpy array
    """
    raster = gdal.Open(filename, GA_ReadOnly)
    imarray = np.array(raster.ReadAsArray())
    return imarray


def zero_one(im):
    """
    scales the input image between 0 and 1
    :param im: input image
    :return: scaled image
    """
    m = im.min()
    im = (im - m) / (im.max() - m)
    return im


def new_mrcnn(semantic_label_file, output_label_file):
    """
    Takes image label tif file "semantic_label_file" and creates mask rcnn usable labels and save it to
    output_label_file with .npy extension.
    :param semantic_label_file: tif file containing pixel level labels for an image.
    :param output_label_file: .npy extension file name to save mrcnn label.
    :return:
    """
    img = skimage.io.imread(semantic_label_file)
    img = img[64:192, 64:192]
    img_labeled = skimage.measure.label(img, connectivity=1)
    idx = [np.where(img_labeled == label) for label in np.unique(img_labeled) if label]

    list_of_all_mask_indices = []
    list_of_all_class_ids = []
    for i in range(len(idx)):
        tmp = np.zeros(img.shape)
        tmp[idx[i]] = img[idx[i]]
        cur_class_id = np.unique(tmp)[1].astype(int)
        list_of_all_mask_indices.append(idx[i])
        list_of_all_class_ids.append(cur_class_id)
    np.save(output_label_file, [list_of_all_mask_indices, list_of_all_class_ids, len(list_of_all_class_ids)])



