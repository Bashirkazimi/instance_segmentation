"""
Mask R-CNN
Train on the harz dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

modified by Bashir Kazimi
------------------------------------------------------------

"""


# Import Mask RCNN
import src.model as modellib
from src import my_config


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
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--NAME', type=str, default="Harz", help='Project Name')

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
        config = my_config.MyConfig()
    else:
        class InferenceConfig(my_config.MyConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
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
        my_config.train(model, config)
    elif args.command == "detect":
        my_config.detect_and_return(model, config)
    elif args.command == "evaluate":
        my_config.evaluate(model, config)
    else:
        print("'{}' is not recognized. Use 'train' or 'detect' ".format(args.command))
