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
import cv2
import sys
import numpy as np
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PETS_DIR = os.path.join(BASE_DIR, 'the-oxford-iiit-pet-dataset')
PETS_DATA_DIR = os.path.join(PETS_DIR, 'images')
PETS_ANNO_DIR = os.path.join(PETS_DIR, 'annotations')
PETS_ANNO_FILE = os.path.join(PETS_ANNO_DIR, 'list.txt')


class OxfordIIITPets(object):

    def __init__(self, colorspace='BGR', train_size=0.8):
        """
        Default constructor of the pets data generator.

        Args:
            colorspace (str): Supported colorspaces so far: BGR and GRAY.
            train_test_split (float): 
        """
        assert colorspace == 'BGR' or colorspace == 'GRAY' or colorspace == 'RGB', 'Not supported colorspace, please use: BGR, RGB or GRAY.'
        assert train_size >= 0. and train_size <= 1., 'Split should be between 0 and 1.'
        self.colorspace = colorspace
        
        # Get filepaths =======================================================
        image_filepaths = [os.path.join(PETS_DATA_DIR, filename) for filename in os.listdir(PETS_DATA_DIR) if 'jpg' in filename]
        
        # Load annotations ====================================================
        labels = {}
        with open(PETS_ANNO_FILE) as anno_file:
            for line in anno_file:
                if line[0] == '#':
                    continue
                else:
                    line_parts = line.strip().split(' ')
                    assert len(line_parts) == 4, "Annotation file corrupted.."
                    labels[line_parts[0]] = [int(line_parts[1]), int(line_parts[2]), int(line_parts[3])]

        # Split train/test ====================================================
        train_fpaths, test_fpaths = train_test_split(image_filepaths, train_size=train_size, shuffle=True)
        self.train_fnames = [os.path.splitext(os.path.basename(fpath))[0] for fpath in train_fpaths]
        self.train_labels = {k:v for k, v in labels.items() if k in self.train_fnames}
        self.train_fnames = [fname for fname in self.train_fnames if fname in self.train_labels.keys()]
        train_fpaths = [os.path.join(PETS_DATA_DIR, fname + '.jpg') for fname in self.train_fnames]
        self.test_fnames = [os.path.splitext(os.path.basename(fpath))[0] for fpath in test_fpaths]
        self.test_labels = {k:v for k, v in labels.items() if k in self.test_fnames}
        self.test_fnames = [fname for fname in self.test_fnames if fname in self.test_labels.keys()]
        test_fpaths = [os.path.join(PETS_DATA_DIR, fname + '.jpg') for fname in self.test_fnames]

    @staticmethod
    def _get_fpaths_from_fnames(fnames):
        return [os.path.join(PETS_DATA_DIR, fname + '.jpg') for fname in fnames]

    @staticmethod
    def _imread_and_resize(fpaths, colorspace='BGR', image_size=64):
        
        images = [cv2.imread(fpath) for fpath in fpaths]
        
        # Resize image ====================================================
        for image_idx in range(len(images)):

            # Reshape with aspect ratio ===================================
            image = images[image_idx]
            old_size = image.shape[:2]
            ratio = float(image_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            image = cv2.resize(image, (new_size[1], new_size[0]), cv2.INTER_AREA)
            
            # Pad the image to be square size =============================                
            delta_w = image_size - new_size[1]
            delta_h = image_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            images[image_idx] = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if colorspace == 'GRAY':
                images[image_idx] = cv2.cvtColor(images[image_idx], cv2.COLOR_BGR2GRAY)
            elif colorspace == 'RGB':
                images[image_idx] = cv2.cvtColor(images[image_idx], cv2.COLOR_BGR2RGB)
        return images

    def get_random_images(self, train=True, instance_number=10, category='species', image_size=64):
        """
        Take random images for each class with how_many_instances examples of each class.

        Args:
            train (bool): Should I provide you with train or test examples?
            instance_number (int): How many examples of each class do you want?
            category (str): What category do you want: species or breeds?
            image_size (int): Output size of images.

        Returns:
            (list of np.ndarrays of size [how_many_instances, 32, 32, 3]): Images.
        """
        assert category=='species' or category=='breeds', "You can categorise with species or breeds only."
        assert image_size>0, "Output size should be a positive number"
        
        if train:
            fnames = self.train_fnames
            labels = self.train_labels
        else:
            fnames = self.test_fnames
            labels = self.test_labels
        
        # What category do we aim for =========================================
        ret = []
        if category == 'species':
            value_index = 1
        elif category == 'breeds':
            value_index = 0
        category_labels = {k:v[value_index] for k, v in labels.items()}
        
        # Get instance_number images ==========================================
        for class_idx in range(1, np.max(list(category_labels.values()))+1):
            class_fnames = [k for k, v in category_labels.items() if v == class_idx]
            class_fnames_random = np.random.choice(class_fnames, size=instance_number)
            class_fpaths = self._get_fpaths_from_fnames(class_fnames_random)
            class_images = self._imread_and_resize(class_fpaths, colorspace='BGR', image_size=image_size)
            ret.append(class_images)
        
        return ret        

    def generate_batch(self, train, batch_size, category='species', image_size=64):
        """
        Generate representative batch of images.

        Args:
            train (bool): Should I provide you with train or test examples?
            batch_szie (int): How many samples do you want?
            category (str): What category do you want: species or breeds?
            image_size (int): Output image size of images.

        Returns:
            (np.ndarray of size [batch_size, image_size, image_size, 3],
             np.ndarray of size [batch_size, 1]): Images and their labels.
        """
        assert category=='species' or category=='breeds', "You can categorise with species or breeds only."
        assert image_size>0, "Output size should be a positive number"
        
        if train:
            fnames = self.train_fnames
            labels = self.train_labels
        else:
            fnames = self.test_fnames
            labels = self.test_labels
            
        # What category do we aim for =========================================
        ret = []
        if category == 'species':
            value_index = 1
        elif category == 'breeds':
            value_index = 0
        category_labels = {k:v[value_index] for k, v in labels.items()}
        category_fnames = list(category_labels.keys())
        np.random.shuffle(category_fnames)

        # What category do we aim for =========================================
        for batch_idx in range(int(len(category_fnames)/batch_size)):
            batch_fnames = category_fnames[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_fpaths = self._get_fpaths_from_fnames(batch_fnames)
            batch_images = self._imread_and_resize(batch_fpaths, colorspace=self.colorspace, image_size=image_size)
            batch_labels = [v-1 for k,v in category_labels.items() if k in batch_fnames]
            yield batch_images, batch_labels

    def images_count(self, train=True):
        if train:
            count = len(self.train_fnames)
        else:
            count = len(self.test_fnames)
            
        return count
