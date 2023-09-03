# Source code of augmentation
# Authored by X. X.

import cv2
import numpy as np
from random import randint
from numpy import random

from aug_utils import random_augmentation

batch_size = 16
input_shape = (64, 64)

def read_input(path):
    img = cv2.imread(path)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    return x

def read_gt(path):
    img = cv2.imread(path)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  / 255.
    x = np.expand_dims(x, axis=2)
    return x

def random_crop(img, mask, crop_size=input_shape[0]):
    imgheight= img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight-crop_size)
    j = randint(0, imgwidth-crop_size)

    return img[i:(i+crop_size), j:(j+crop_size), :], mask[i:(i+crop_size), j:(j+crop_size)]


def gen(data, au=False, inv=False):
    while True:
        repeat = 4
        index= random.choice(list(range(len(data))), batch_size//repeat)
        index = list(map(int, index))
        list_images_base = [read_input(data[i][0]) for i in index]
        list_gt_base = [read_gt(data[i][1]) for i in index]

        list_images = []
        list_gt = []

        for image, gt in zip(list_images_base, list_gt_base):

            for _ in range(repeat):
                image_, gt_ = random_crop(image.copy(), gt.copy())
                list_images.append(image_)
                list_gt.append(gt_)

        list_images_aug = []
        list_gt_aug = []

        for image, gt in zip(list_images, list_gt):
            if au:
                image, gt = random_augmentation(image, gt)
            list_images_aug.append(image)
            list_gt_aug.append(gt)

        yield np.array(list_images_aug), np.array(list_gt_aug)