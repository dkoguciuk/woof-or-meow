#!/usr/bin/env python
# -*- coding: utf-8 -*

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
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import data_generator as gen


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
CNNF_DIR = os.path.join(FEAT_DIR, 'cnn')
MDLS_DIR = os.path.join(BASE_DIR, 'models')
RSRC_DIR = os.path.join(MDLS_DIR, 'research')
SLIM_DIR = os.path.join(RSRC_DIR, 'slim')
sys.path.append(SLIM_DIR)
import nets.inception as inception_model

IMAGE_SIZE = 299
BATCH_SIZE = 4
INCEPTION_MODEL_PATH = 'inception/inception_v3.ckpt'


class InceptionV3(object):
    def __init__(self, sess):
        # placeholder
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
        # inception features extractor
        device = tf.device('device:CPU:0')
        if tf.test.is_gpu_available():
            device = tf.device('device:GPU:0')
        with device:
            with tf.contrib.slim.arg_scope(inception_model.inception_v3_arg_scope()):
                self.features_extractor, _ = inception_model.inception_v3(self.input_placeholder,
                                                                          num_classes=0, is_training=False)
        # init
        init_fn = tf.contrib.slim.assign_from_checkpoint_fn(INCEPTION_MODEL_PATH,
                                                            tf.contrib.slim.get_model_variables("InceptionV3"))
        init_fn(sess)

def augment_data_mirror(images, labels):
    """
    Augment images with mirroring.

    Args:
        images (list of ndarrays of shape [IMAGE_SIZE, IMAGE_SIZE, 3]): Original images to be augmented.

    Returns:
        (list of ndarrays of shape [IMAGE_SIZE, IMAGE_SIZE, 3]): Augmented images.
    """

    # flip images horizontaly =================================================
    images_flipped = []
    labels_flipped = []
    for idx in range(len(images)):
        images_flipped.append(cv2.flip(images[idx], 1))
        labels_flipped.append(labels[idx])
        images_flipped.append(images[idx])
        labels_flipped.append(labels[idx])

    # return
    return images_flipped, labels_flipped

def calc_features(train, multiplications, augmentation, inception, sess, category='species'):
    """
    Calculate cnn features with inception v3.

    Args:
        train (bool): Am I working with train or test data?
        multiplications (int): How many times should I augment data (should be 1 for
            test dataset).
        augmentation (int): Augmentation level, either 0, 1 or 2.
        inception (InceptionV3 object): Inception model.
        sess (Tensorflow session): Tensorflow session.
    """

    # counter
    instances_counter = { str(idx) : 0 for idx in range(10)}

    # for every batch
    generator = gen.OxfordIIITPets(colorspace='RGB', train_size=0.8)
    batches = generator.images_count(train) / BATCH_SIZE
    all_featrs = []
    all_labels = []

    for _ in range(multiplications):
        for batch_images, batch_labels in tqdm(generator.generate_batch(train=train, batch_size=BATCH_SIZE, category=category, image_size=IMAGE_SIZE), total=batches):

            # augmentation
            if augmentation == 1:
                images_augmented, labels_augmented = augment_data_mirror(batch_images, batch_labels)
            else:
                images_augmented, labels_augmented = batch_images, batch_labels

            # norm
            images_normalized = [cv2.normalize(src=image, dst=None, alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                 for image in images_augmented]

            # to tf format
            np_images_data = [np.asarray(image_normalized) for image_normalized in images_normalized]

            # stack
            tf_images_data = np.stack(np_images_data, axis=0)

            # calc features
            features = sess.run(inception.features_extractor, feed_dict={inception.input_placeholder : tf_images_data})
            all_featrs.append(np.squeeze(features))
            all_labels.append(labels_augmented)

    # Save it!
    all_featrs = np.concatenate(all_featrs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_featrs, all_labels


def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--augmentation_level", help="augment dataset: option 0, 1 or 2, 3", type=int, default=0)
    parser.add_argument("-m", "--multiplications", help="how many times should I augment training dataset?", type=int, default=1)
    args = vars(parser.parse_args())

    # assert aug level
    if args['augmentation_level'] != 0 and args['augmentation_level'] != 1:
        assert False, "augmentation level can be one of: 0, 1" 

    # assert features dir
    out_dir = CNNF_DIR + "-" + str(args['augmentation_level'])
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # tf session & graph
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    graph.as_default()

    # InceptionV3
    inception = InceptionV3(sess)

    # train
    print("Calculating features for train dir...")
    features_train, labels_train = calc_features(True, args['multiplications'], args['augmentation_level'], inception, sess)
    np.save(os.path.join(out_dir, "features_train.npy"), features_train)
    np.save(os.path.join(out_dir, "labels_train.npy"), labels_train)

    # test
    print("Calculating features for test dir...")
    features_test, labels_test = calc_features(False, 1, 0, inception, sess)
    np.save(os.path.join(out_dir, "features_test.npy"), features_test)
    np.save(os.path.join(out_dir, "labels_test.npy"), labels_test)

if __name__ == "__main__":
    main(sys.argv[1:])
