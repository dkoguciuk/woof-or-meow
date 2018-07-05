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
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE
from utils import data_generator as gen
from utils.models import MLPClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, 'features')


def hparam_search(features_train, labels_train, features_test, labels_test):
    """
    Search best C param for SVM classifier but first reduce dimension of the features.

    Args:
        features_train (ndarray of size [images, features_no]): Features of the train dataset.
        labels_train (ndarray of size [images]): Labels of the train dataset.
        features_test (ndarray of size [images, features_no]): Features of the test dataset.
        labels_test (ndarray of size [images]): Labels of the test dataset.
    """

    VARIANCE = 0.60
    pca_hparam = decomposition.PCA(VARIANCE)
    pca_hparam.fit(features_train)
    features_hparam_train = pca_hparam.transform(features_train)

    print("Componenst with ", VARIANCE * 100, "% of variance: ", pca_hparam.n_components_)
    for C in [0.001, 0.01, 0.1, 1.0, 1.2, 1.5, 2.0, 10.0]:
        classifier_svm = LinearSVC(C=C, verbose=False)
        classifier_svm.fit(features_hparam_train, labels_train)
        print("======= C:", C, "=======")
        print("TRAIN SCORE = ", classifier_svm.score(features_hparam_train, labels_train))
        features_hparam_test = pca_hparam.transform(features_test)
        print("TEST  SCORE = ", classifier_svm.score(features_hparam_test, labels_test))

def classify_svm(features_train, labels_train, features_test, labels_test, C=1.0, verbose=False):
    """
    Train SVM classifier and eval train and test scores.

    Args:
        features_train (ndarray of size [images, features_no]): Features of the train dataset.
        labels_train (ndarray of size [images]): Labels of the train dataset.
        features_test (ndarray of size [images, features_no]): Features of the test dataset.
        labels_test (ndarray of size [images]): Labels of the test dataset.
        C (float): C parameter of SVM classifier.
        verbose (bool): Should I print some additional info?
    """
    print("Learning started, please wait...")
    svm_time_start = time.time()
    classifier_svm = LinearSVC(C=C, verbose=verbose, dual=True, max_iter=1000)
    classifier_svm.fit(features_train, labels_train)
    svm_time_fit = time.time()
    print("SVM fit in ", svm_time_fit - svm_time_start, " seconds\n\n")
    print("TRAIN SCORE = ", classifier_svm.score(features_train, labels_train))
    print("TEST  SCORE = ", classifier_svm.score(features_test, labels_test))

def classify_mlp(features_train, labels_train, features_test, labels_test, epochs, learning_rate):

    BATCH_SIZE = 200
    CLASSES_COUNT = np.max(labels_train) + 1
   
    # Reset
    tf.reset_default_graph()
    
    # Define model
    with tf.device("/device:GPU:0"):
        model_classifier = MLPClassifier([features_train.shape[-1], 1024, 512, 256, 128, 64, CLASSES_COUNT], BATCH_SIZE, learning_rate)

    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:
        
        # Run the initialization
        sess.run(tf.global_variables_initializer())

        # Logs
        log_model_dir = os.path.join("logs", model_classifier.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())))
#         writer.add_graph(sess.graph)

        # Do the training loop
        global_batch_idx = 1
        print("Learning started, please wait...")
        for epoch in range(epochs):

            indices = np.arange(features_train.shape[0])
            features_shuffled = features_train.copy()[indices]
            labels_shuffled = labels_train.copy()[indices]
            for index in range(len(features_train)/BATCH_SIZE):
                features = features_shuffled[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                labels = labels_shuffled[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                
                # zero mean
                #features = np.clip((features - 0.38) / 10, -1.0, 1.0)
                
                # run optimizer
                labels_one_hot = sess.run(tf.one_hot(labels, CLASSES_COUNT))
                _, loss, pred, summary = sess.run([model_classifier.optimizer, model_classifier.loss,
                                                   model_classifier.get_classification_prediction(), model_classifier.summary],
                                                   feed_dict={model_classifier.placeholder_embed: features,
                                                              model_classifier.placeholder_label: labels_one_hot})
                
                # train acc
                acc = float(sum(np.argmax(pred, axis=-1) == labels)) / labels.shape[0]
                
                # summ
                writer.add_summary(summary, global_batch_idx)
                global_batch_idx += 1

                # Info
                print("Epoch: %06d batch: %03d loss: %06f train acc: %03f" % (epoch + 1, index, loss, acc))
                index += 1
        
        accs = []
        #features_test = np.clip((features_test - 0.38) / 10, -1.0, 1.0)
        for index in range(len(features_test)/BATCH_SIZE):
            features = features_test[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            labels = labels_test[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            pred = sess.run(model_classifier.get_classification_prediction(),
                            feed_dict={model_classifier.placeholder_embed: features})
            acc = float(sum(np.argmax(pred, axis=-1) == labels)) / labels.shape[0]
            accs.append(acc)
        print("TEST ACC = ", np.mean(accs))


def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", help="features to decompose (hog, cnn-0, cnn-1, ...)", type=str, default='hog')
    parser.add_argument("-c", "--classifier", help="supported image features: svm, mlp", type=str, default='svm')
    parser.add_argument("-e", "--epochs", help="training epochs for mlp classifier", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", help="training learning_rate for mlp classifier", type=float, default=0.0005)
    parser.add_argument('-s', '--hparam_search', help='search for best C param for SVM classifier', action='store_true')
    args = vars(parser.parse_args())  
 
     # Load features ===========================================================
    data_dir = os.path.join(FEAT_DIR, args['features'])
    if not os.path.exists(data_dir):
        print ("There is no such features calculated...")
 
    features_train = np.load(os.path.join(data_dir, "features_train.npy"))
    labels_train = np.load(os.path.join(data_dir, "labels_train.npy"))
    features_test = np.load(os.path.join(data_dir, "features_test.npy"))
    labels_test = np.load(os.path.join(data_dir, "labels_test.npy"))

    ######################################################################################
    #################################### HPARAM SEARCH ###################################
    ######################################################################################
 
    if args['hparam_search']:
        hparam_search(features_train, labels_train, features_test, labels_test)
 
    ######################################################################################
    ################################ FINAL CLASSIFICATION ################################
    ######################################################################################
 
    if args['classifier'] == 'svm':
        classify_svm(features_train, labels_train, features_test, labels_test)
    elif args['classifier'] == 'mlp':
        classify_mlp(features_train, labels_train, features_test, labels_test, args['epochs'], args['learning_rate'])

if __name__ == "__main__":
    main(sys.argv[1:])
