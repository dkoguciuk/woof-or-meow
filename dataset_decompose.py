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
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, 'features')


def decompose(features, labels):
    """
    Decompose features with some dimensionality reduction techniques.

    Args:
        features (ndarray of size [images, features_no]): Features of the dataset.
        labels (ndarray of size [images]): Labels of the dataset.
    """

    # PCA decomposition 
    start_time = time.time()
    pca = decomposition.PCA(n_components=2)
    pca.fit(features)
    features_pca = pca.transform(features)
    pca_time = time.time()
    print("PCA features calculated in ", pca_time - start_time, " seconds with variance ", pca.explained_variance_ratio_)

    # t-SNE decomposition
    elems = 5000
    tsne = TSNE(n_components=2, verbose=True, perplexity=40, n_iter=300)
    features_tsne = tsne.fit_transform(features[:elems], labels[:elems])
    tsne_time = time.time()
    print("t-SNE features calculated in ", tsne_time - pca_time, " seconds ")

    # plots
    plt.figure(figsize=(15, 15))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    plt.figure(figsize=(15, 15))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels[:elems])
    plt.show()

def main(argv):

    # Parser ==================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", help="features to decompose (hog, cnn-0, cnn-1, ...)", type=str, default='hog')
    args = vars(parser.parse_args())
    
    # Load features ===========================================================
    data_dir = os.path.join(FEAT_DIR, args['features'])
    if not os.path.exists(data_dir):
        print ("There is no such features calculated...")
    
    features_train = np.load(os.path.join(data_dir, "features_train.npy"))
    labels_train = np.load(os.path.join(data_dir, "labels_train.npy"))
    decompose(features_train, labels_train)


if __name__ == "__main__":
    main(sys.argv[1:])
