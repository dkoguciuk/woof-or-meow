#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2018 Daniel Koguciuk <daniel.koguciuk@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 04.07.2018
'''


import os
import sys
import cv2
import time
import tqdm
import shutil
import argparse
import numpy as np
from utils import data_generator as gen


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
HOGF_DIR = os.path.join(FEAT_DIR, 'hog')
if not os.path.exists(FEAT_DIR):
    os.mkdir(FEAT_DIR)
if os.path.exists(HOGF_DIR):
    shutil.rmtree(HOGF_DIR)
os.mkdir(HOGF_DIR)

def __HOG(img, cell_size=(8,8), nbins=9):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin = np.int32(nbins*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx, celly = cell_size

    for i in range(0,int(img.shape[0]/celly)):
        for j in range(0,int(img.shape[1]/cellx)):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])   

    hists = [np.bincount(b.ravel(), m.ravel(), nbins) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= cv2.norm(hist) + eps

    return hist

def _HOG(images, image_size):
    """
    Calc HOG features for grayscale images.

    Args:
        images (ndarray of size [images, some_size, some_size]): Grayscale images.

    Returns:
        (ndarray of size [images, features_no]): HOG features for each image.
    """

    NBINS = 9
    CELL_SIZE = (int(image_size/16), int(image_size/16))
    hog_features = [__HOG(image, cell_size=CELL_SIZE, nbins=NBINS) for image in images]
    return np.stack(hog_features, axis=0)

def extract_HOG(generator, category='species', image_size=256, train=True, verbose=True):
    """
    Extract HOG features for specified dataset (train/test).

    Args:
        generator (Generator class object): Generator class object.
        train (bool): Am I working with train or test data?
        verbose (bool): Should I print some additional info?
        category (str): What category do you want: species or breeds?

    Returns:
        (ndarray of size [images, features_no], ndarray of size [images]) Features and labels. 
    """
    all_featrs = []
    all_labels = []
    batch_size = 64
    start_time = time.time()
    batches = generator.images_count(train=train) / batch_size
    print("Calculating HOG featues..")
    for images, labels in tqdm.tqdm(generator.generate_batch(train=train, batch_size=batch_size, category=category, image_size=image_size), total=batches):
        all_featrs.append(_HOG(images, image_size))
        all_labels.append(labels)
    all_featrs = np.concatenate(all_featrs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    hog_time = time.time()
    if verbose:
        print ("Features calculated in ", hog_time - start_time, " seconds")
    return all_featrs, all_labels


def main(argv):

    # Parser ==================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--image_size", help="images size (defaults: 256)", type=int, default=256)
    args = vars(parser.parse_args())

    # Extract features ========================================================
    generator = gen.OxfordIIITPets(colorspace='GRAY', train_size=0.8)
    features_train, labels_train = extract_HOG(generator, category='species', image_size=args['image_size'], train=True, verbose=True)
    features_test, labels_test = extract_HOG(generator, category='species', image_size=args['image_size'], train=False, verbose=True)

    # Save ====================================================================
    np.save(os.path.join(HOGF_DIR, "features_train.npy"), features_train)
    np.save(os.path.join(HOGF_DIR, "labels_train.npy"), labels_train)
    np.save(os.path.join(HOGF_DIR, "features_test.npy"), features_test)
    np.save(os.path.join(HOGF_DIR, "labels_test.npy"), labels_test)

if __name__ == "__main__":
    main(sys.argv[1:])
