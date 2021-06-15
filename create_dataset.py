import os
import time
import cv2
import numpy as np
import yaml
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
                            IAAEmboss, RGBShift, IAASharpen, GaussianBlur, IAAAdditiveGaussianNoise, RandomBrightnessContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded, OneOf, Compose
                           )
from sklearn.model_selection import train_test_split
import argparse
from lib.data.datasets import is_image_file


def generate_dataset(train_normal, test_normal, test_abnormal, dataset_name, train_abnormal=None, img_shape=(128,128)):
    if not os.path.isdir(os.path.join("data")): os.mkdir(os.path.join("data"))
    if not os.path.isdir(os.path.join("data", dataset_name)): os.mkdir(os.path.join("data", dataset_name))
    else: 
        raise ValueError("The Dataset_name {} already exists. Please choose another name or delete the existing one.".format(dataset_name))
    for folder in ["train", "test"]:
        for sub_folder in ["0.normal", "1.abnormal"]:
            if not train_abnormal and folder == "train" and sub_folder == "1.abnormal":
                continue
            if not os.path.isdir(os.path.join("data", dataset_name, folder, sub_folder)): os.makedirs(os.path.join("data", dataset_name, folder, sub_folder))
    zipped_stuff = zip([train_normal, test_normal, test_abnormal], [["train", "0.normal"], ["test", "0.normal"], ["test","1.abnormal"]])
    if train_abnormal:
        zipped_stuff = zip([train_normal, train_abnormal, test_normal, test_abnormal], [["train", "0.normal"], ["train", "1.abnormal"], ["test", "0.normal"], ["test", "1.abnormal"]])
    for dataset, path in zipped_stuff:
        for image in dataset:
            image = cv2.resize(image, img_shape)
            file_name = str(time.time())+".png"
            cv2.imwrite(os.path.join(os.getcwd(), "data", dataset_name, path[0], path[1], file_name), image)
            time.sleep(0.001)

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
    if n_channel == 1:
        mean = image.mean()
        std = image.std()
        image = image - mean / std
    else:
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
    parser.add_argument("-c", "--config", action="store", help="path to config.yaml... Does not have to been set in the file is in root of skip-ganomaly", default="../config.yaml")
    parser.add_argument("-p", "--path", action="store", default="..\\", help="prepath tp paths in config.yaml. i.e. '..\\' if the folder is in the parent directory.")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("No File named {} found!".format(args.config))
    #config = {**cfg, **args.overwrite_config} if args.overwrite_config is not None else cfg
    config = cfg
    img_shape=(config["image_size"], config["image_size"])
    print("Trying to get all images from {0} as normal images and images from {1} as abnormal images!".format(str(config["normal_images_paths"]), str(config["normal_images_paths"])))
    normal_images = []
    abnormal_images = []
    blacklist_path = []
    for path in config["blacklist"]:
        blacklist_path.append(os.path.relpath(args.path + path))
    for flag, paths in {"normal":config["normal_images_paths"], "abnormal":config["abnormal_images_paths"]}.items():
        for path in paths:
            path = os.path.relpath(args.path + path)
            
            file_path = [os.path.join(path, o) for o in os.listdir(path) if is_image_file(o)]
            for file in file_path:
                
                if file in blacklist_path:
                    print("{} is a blacklist image".format(str(file)))
                    continue
                else:
                    image = cv2.imread(file)                        
                    image = cv2.resize(image, img_shape)
                    

                    if flag == "normal":
                        normal_images.append(image)
                    else:
                        abnormal_images.append(image)
    print("Collected {0} normal images and {1} abnormal images!".format(str(len(normal_images)),
                                                                        str(len(abnormal_images))))
    if config["standardize"]:
        print("Doing local standardization!")
        for images in [normal_images, abnormal_images]:
            for i in range(len(images)):
                images[i] = local_standardization(images[i], 3)
                if config["greyscale"]:
                        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        print("After standardization:")
        print(len(normal_images))
        print(len(abnormal_images))
        
    if config["rotate"]:
        print("Doing all image rotation")
        
        for key, images in {"normal":normal_images, "abnormal":abnormal_images}.items():
            rotated_images = []
            for image in images:
                rotated_image = rotate_image(image)
                rotated_images.append(rotated_image)
            if key == "normal":
                normal_images = np.concatenate((images, rotated_images))
            else:
                abnormal_images = np.concatenate((images, rotated_images))
        print("After rotation:")
        print(len(normal_images))
        print(len(abnormal_images))
    print("Doing train test split")
    train_normal, test_normal = train_test_split(normal_images, test_size=config["test_data_size"])
    train_abnormal, test_abnormal = train_test_split(abnormal_images, test_size=config["test_data_size"])
    
    if config["augment"]:
        print("Augmenting normal images")
        augmentation = augment_image(p=0.7)
        augmented_images = []
        for image in train_normal:
            augmented_image = augmentation(image=image)["image"]
            augmented_images.append(augmented_image)
        train_normal = np.concatenate((train_normal, augmented_images))
    
    

    np.random.shuffle(train_normal)
    np.random.shuffle(test_normal)
    np.random.shuffle(train_abnormal)
    np.random.shuffle(test_abnormal)
    print("train_normal", len(train_normal))
    print("test_normal", len(test_normal))
    print("train_abnormal", len(train_abnormal))
    print("test_abnormal", len(test_abnormal))
    if "all" in config["model"]:
        generate_dataset(train_normal=train_normal, test_normal=test_normal, train_abnormal=train_abnormal, test_abnormal=test_abnormal, dataset_name=config["dataset_name"] + "_deep", img_shape=img_shape)
        generate_dataset(train_normal=train_normal, test_normal=test_normal, test_abnormal=np.concatenate((train_abnormal, test_abnormal)), dataset_name=config["dataset_name"] + "_skip", img_shape=img_shape)
    elif "deep" in config["model"]:
        generate_dataset(train_normal=train_normal, test_normal=test_normal, train_abnormal=train_abnormal, test_abnormal=test_abnormal, dataset_name=config["dataset_name"] + "_" + config["model"], img_shape=img_shape)
    elif "skip" in config["model"]:
        generate_dataset(train_normal=train_normal, test_normal=test_normal, test_abnormal=np.concatenate((train_abnormal, test_abnormal)), dataset_name=config["dataset_name"] + "_" + config["model"], img_shape=img_shape)
    else:
        raise NotImplementedError("Can't create dataset for not implemented Model")

