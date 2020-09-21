import os
import time
import cv2
import numpy as np
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
                            IAAEmboss, RGBShift, IAASharpen, GaussianBlur, IAAAdditiveGaussianNoise, RandomBrightnessContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded, OneOf, Compose
                           )
from sklearn.model_selection import train_test_split
import argparse


def generate_skip_ganomaly_dataset(train_normal, test_normal, test_abnormal, dataset_name, img_shape=(128,128)):
    if not os.path.isdir(os.path.join("data", dataset_name)):
         os.mkdir(os.path.join("data", dataset_name))
    for folder in ["train", "test"]:
        if not os.path.isdir(os.path.join("data", dataset_name, folder)):
            os.mkdir(os.path.join("data", dataset_name, folder))
    if not os.path.isdir(os.path.join("data", dataset_name,  "train", "0.normal")):
        os.mkdir(os.path.join("data", dataset_name,  "train", "0.normal"))
    for folder in ["0.normal", "1.abnormal"]:
        if not os.path.isdir(os.path.join("data", dataset_name, "test", folder)):
            os.mkdir(os.path.join("data", dataset_name, "test", folder))
    for dataset, path in zip([train_normal, test_normal, test_abnormal], [["train", "0.normal"], ["test", "0.normal"], ["test","1.abnormal"]]):
        for image in dataset:
            image = cv2.resize(image, img_shape)
            cv2.imwrite(os.path.join(os.getcwd(), "data", dataset_name, path[0], path[1], str(time.time())+".png"), image)

def augment_image(p=0.7):
    return Compose([
      VerticalFlip(),
      HorizontalFlip(),
      OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise()
      ]),
      OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ]),
  ], p=p)


def global_standardization(image):
    image = image.astype('float32')
    #calculate pixel mean and standard deviation
    mean, std = image.mean(), image.std()
    #standardize th pixel images
    image = (image - mean) / std
    return image


def local_standardization(image, n_channel):
    image = image.astype('float32')
    means = []
    stds = []
    #Calculate pixel mean and standard deviation per channel
    for i in range(n_channel):
        means.append(image[:,:,i].mean())
        stds.append(image[:,:,i].std())
    # standardize the pixel images by each channel
    for i in range(n_channel):
        image[:,:,i] = (image[:,:,i] - means[i]) / stds[i]
    # clip pixel values to [-1,1]
    image = np.clip(image, -1.0, 1.0)
    #shift from [-1,1] to [0,1] with 0.5 mean
    image = (image + 1.0) / 2.0
    #transform back t [0,255] pixel range
    image = (image*255).astype("uint8")
    return image

def rotate_image(image):
    return RandomRotate90(always_apply=True).apply(image, factor=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-nip","--normal_images_path", action="store", help="absolute path to images without defect (dont use backslash, use normal slash instead)",
                        required=True)
    parser.add_argument("-aip","--abnormal_images_path", action="store", help="absolute path to images with defect(dont use backslash, use normal slash instead)",
                        required=True)
    parser.add_argument("-is", "--image_size", action="store", help="wanted size of images. Will be quadratic so just wirte a single number for in pixel", default="256")
    parser.add_argument("-dn", "--dataset_name", action="store", help="name of the resulting dataset", default="custom_dataset")
    parser.add_argument("-aug", "--augment", action="store_true", help="determine if the images should be augmented")
    parser.add_argument("-r", "--rotate", action="store_true", help="determine if the images should be rotated by 90 degrees")
    parser.add_argument("-s", "--standardize", action="store_true", help="determine if the images should be locally standardized")

    args = parser.parse_args()
    normal_images_path = os.path.abspath(args.normal_images_path)
    abnormal_images_path = os.path.abspath(args.abnormal_images_path)
    print("Trying to get all images from {0} as normal images and images from {1} as abnormal images!"
          .format(str(normal_images_path), str(abnormal_images_path)))
    normal_images = []
    abnormal_images = []
    normal_run=True
    for path in [normal_images_path, abnormal_images_path]:
        file_path = [os.path.join(path, o) for o in os.listdir(path) if (".png" in o or ".jpg" in o or ".bmp" in o)]
        for file in file_path:
            image = cv2.imread(file)
            if normal_run:
                normal_images.append(image)
            else:
                abnormal_images.append(image)
        normal_run=False
    print("Collected {0} normal images and {1} abnormal images!".format(str(len(normal_images)),
                                                                        str(len(abnormal_images))))
    if args.standardize:
        print("Doing local standardization!")
        for images in [normal_images, abnormal_images]:
            for i in range(len(images)):
                images[i] = local_standardization(images[i], 3)

    print("Doing train test split")
    train_normal, test_normal = train_test_split(normal_images, test_size=0.2)
    if args.rotate:
        print("Doing all image rotation")
        rotated_images = []
        for images in [normal_images, abnormal_images]:
            for image in images:
                rotated_image = rotate_image(image)
                rotated_images.append(rotated_image)
            images = np.concatenate((images, rotated_images))
    if args.augment:
        print("Augmenting normal images")
        augmentation = augment_image(p=0.7)
        augmented_images = []
        for image in train_normal:
            augmented_image = augmentation(image=image)["image"]
            augmented_images.append(augmented_image)
        train_normal = np.concatenate((train_normal, augmented_images))

    np.random.shuffle(train_normal)
    np.random.shuffle(test_normal)
    np.random.shuffle(abnormal_images)

    generate_skip_ganomaly_dataset(train_normal, test_normal, abnormal_images, args.dataset_name,
                                   (int(args.image_size),int(args.image_size)))

