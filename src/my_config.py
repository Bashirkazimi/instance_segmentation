"""
Mask R-CNN
Train on the harz digital elevation dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified by Bashir Kazimi

"""

import os
import sys
import numpy as np
import skimage.draw
from glob import glob

from src.config import Config
import src.model as modellib
import src.utils as utils
from src import my_utils


############################################################
#  Configurations
############################################################


class MyConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "Harz"  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 4

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.

    EPOCHS = 100
    STEPS_PER_EPOCH = 1250

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 160

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1+4  # background, bomb craters, charcoal kilns, barrows, mine shafts

    CLASSES = ["bombs", "meiler", "barrows", "pinge"]

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 1000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 200

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # IMAGE_CHANNEL_COUNT = 3
    IMAGE_CHANNEL_COUNT = 1

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL = np.array([0]) # we have one channel, and we do not want any mean subtraction.

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # NEW ARGUMENTS

    # input images for training
    dataset_dirX = "/media/kazimi/Data/data/bmbp_data/x"

    # input labels for training
    dataset_dirY = "/run/user/1000/gvfs/smb-share:server=koko,share=tmp/kazimi/data/bmbp_MRCNN_DATA/ys/masks"

    # input images for validation
    val_dataset_dirX = "/media/kazimi/Data/data/bmbp_data/validation/x"

    # input labels for validation
    val_dataset_dirY = "/run/user/1000/gvfs/smb-share:server=koko,share=tmp/kazimi/data/bmbp_MRCNN_DATA/ys/masks"

    # input images for test
    test_dataset_dirX = "/media/kazimi/Data/data/bmbp_data/test/x"

    # input labels for test
    test_dataset_dirY = "/run/user/1000/gvfs/smb-share:server=koko,share=tmp/kazimi/data/bmbp_MRCNN_DATA/ys/masks"

    # height and width of training instances
    height = 128
    width = 128

    # path to saved weight file
    weights = None  # set it to 'last' if loading saved weights, or set it to weights path

    # Logs and checkpoints directory
    logs = "./log"

    # path to example image to detect objects at
    image = 'my_image.tif'

    # preprocess input image or not?
    preprocess = False

    # iou_threshold to calculate average precision
    iou_threshold = 0.5

############################################################
#  Dataset
############################################################


class MyDataset(utils.Dataset):

    def load_my_data(self, dataset_dirX, dataset_dirY, height, width, classnames = ['bomb', 'meiler', 'barrow',
                                                                                 'pinge'], preprocess=False):
        """Load a subset of the my dataset.
        dataset_dirX: directory of the dataset images.
        dataset_dirY: directory of masks and instances
        """
        # Add classes.
        # self.add_class("balloon", 1, "bomb")
        # self.add_class("balloon", 2, "meiler")
        self.classnames = classnames
        for i,cn in enumerate(classnames):
            self.add_class("harz", i+1, cn)

        self.preprocess = preprocess
        # Add images
        images = glob(os.path.join(dataset_dirX, '*.tif'))
        for im in images:
            _, name = os.path.split(im)
            just_name, ext = os.path.splitext(name)
            niedersachsen_num = just_name.split('_')[0]
            just_num = niedersachsen_num[13:]
            self.add_image(
                "harz",
                image_id=name,  # use file name as a unique image id
                path=im,
                mask_path=os.path.join(dataset_dirY, just_num+'.npy'),
                width=width, height=height)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if self.preprocess:
            image = my_utils.zero_one(image)

        image = image[64:192, 64:192]
        return np.expand_dims(image, -1)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        # if image_info["source"] != "balloon":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_indices, class_ids, num_instances = np.load(info['mask_path'],allow_pickle=True)

        mask = np.zeros([info["height"], info["width"], num_instances],
                        dtype=np.uint8)
        # id_array = []
        instance_looper = 0
        for rr, cc in mask_indices:
            mask[rr, cc, instance_looper] = 1
            instance_looper += 1

        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "harz":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, config):
    """Train the model."""
    # Training dataset.
    dataset_train = MyDataset()
    dataset_train.load_my_data(config.dataset_dirX, config.dataset_dirY, config.height, config.width, config.CLASSES,
                               config.preprocess)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MyDataset()
    dataset_val.load_my_data(config.val_dataset_dirX, config.val_dataset_dirY, config.height, config.width,
                             config.CLASSES, config.preprocess)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers='all')


def evaluate(model, config):
    dataset_test = MyDataset()
    dataset_test.load_my_data(config.test_dataset_dirX, config.test_dataset_dirY, config.height, config.width,
                              config.CLASSES, config.preprocess)
    dataset_test.prepare()
    image_ids = np.copy(dataset_test.image_ids)
    APS = []
    for image_id in image_ids:
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, image_id,
                                                                         use_mini_mask=False)
        results = model.detect([image], verbose=0)
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'],
                                                             r['class_ids'], r['scores'], r['masks'],
                                                             iou_threshold=config.iou_threshold)
        # print(AP, precisions, recalls, overlaps)
        APS.append(AP)
    mAP = np.mean(APS)
    print('map @ IoU = {}: {}'.format(config.iou_threshold, mAP))
    return mAP


def detect_and_return(model, config):
    image_path = config.image
    if os.path.isfile(image_path):
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, -1)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        return r
    else:
        sys.exit("No such image found: {}".format(image_path))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments. Any provided command line arg will override the default variables defined in config
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or 'detect'")
    parser.add_argument('--dataset_dirX', required=False,
                        default="/media/kazimi/Data/data/bmbp_data/x",
                        help='Directory of the harz input dataset')
    parser.add_argument('--dataset_dirY', required=False,
                        default="path/to/trainlabels",
                        help='Directory of the harz label dataset')
    parser.add_argument('--val_dataset_dirX', required=False,
                        default="/media/kazimi/Data/data/bmbp_data/validation/x",
                        help='Directory of the harz validation input dataset')
    parser.add_argument('--val_dataset_dirY', required=False,
                        default="path/to/validationlabels",
                        help='Directory of the harz validation label dataset')
    parser.add_argument('--test_dataset_dirX', required=False,
                        default="/media/kazimi/Data/data/bmbp_data/test/x",
                        help='Directory of the harz test input dataset')
    parser.add_argument('--test_dataset_dirY', required=False,
                        default="path/to/testlabels",
                        help='Directory of the harz test label dataset')
    parser.add_argument('--height', type=int, default=128, help='Image height')
    parser.add_argument('--width', type=int, default=128, help='Image width')
    parser.add_argument('--NAME', type=str, default="Harz", help='Project Name')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold')

    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default='./log',
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--preprocess', type=bool, default=False,
                        help='True if input image should be preprocessed, do not add flag if no preprocessing.')
    args = parser.parse_args()

    if args.command == "train":
        config = MyConfig()
    else:
        class InferenceConfig(MyConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5
        config = InferenceConfig()
    config.display()

    for k, v in vars(args).items():
        setattr(config, k, getattr(args, k))

    config.display()
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config)

    if config.weights:
        # Find last trained weights
        print("Loading model from previous weight file!")
        if config.weights == "last":
            weights_path = model.find_last()
            model.load_weights(weights_path, by_name=True)
        else:
            model.load_weights(config.weights, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config)
    elif args.command == "detect":
        detect_and_return(model, config)
    elif args.command == "evaluate":
        evaluate(model, config)
    else:
        print("'{}' is not recognized. Use 'train' or 'detect' ".format(args.command))