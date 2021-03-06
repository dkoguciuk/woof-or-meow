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
import time
import numpy as np
import tensorflow as tf


class MLPClassifier(object):

    MODEL_NAME = "MLPClassifier"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    CLASSES_COUNT = 10
    """
    How many classes do we have.
    """

    def __init__(self, mlp_layers_sizes, batch_size, learning_rate):
        """
        Build a model.
        Args:
            input_features_sizes (int): 
            mlp_layers_sizes (list of ints): List of hidden units of mlp, where the first
                value is considered to be a size of input feature vector.
            batch_size (int): Batch size of SGD.
            learning_rate (float): Learning rate of SGD.
        """        
        # Save params
        self.mlp_layers_sizes = mlp_layers_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.summaries = []
        
        with tf.name_scope(self.MODEL_NAME):
        
            # Placeholders
            with tf.name_scope("placeholders"):
                self.placeholder_embed = tf.placeholder(tf.float32, [self.batch_size, self.mlp_layers_sizes[0]], name="input_embedding")
                self.placeholder_label = tf.placeholder(tf.float32, [self.batch_size, self.mlp_layers_sizes[-1]], name="true_labels")
    
            # init MLP params
            with tf.name_scope("params"):
                self._init_params()
            
            # calculate loss & optimizer
            with tf.name_scope("train"):
                self.classification_pred = self._define_classifier(self.placeholder_embed)
                self.loss = self._calculate_loss(self.classification_pred, self.placeholder_label)
                self.optimizer = self._define_optimizer(self.loss)
                self.summaries.append(tf.summary.scalar('loss', self.loss))
            
            # merge summaries and write        
            self.summary = tf.summary.merge(self.summaries)

    def get_classification_prediction(self):
        """
        Get classification prediction.
        """
        return tf.nn.softmax(self.classification_pred)

    def get_model_name(self):
        return self.MODEL_NAME

    def save_model(self, session):
        """
        Save the model in the model dir.

        Args:
            session (tf.Session): Session which one want to save model.
        """
        saver = tf.train.Saver()
        name = self.MODEL_NAME + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + ".ckpt"
        return saver.save(session, os.path.join("models_classifier", name))      

    def _init_params(self):
        """
        Initialize params for MLP network classifier.
        """
        # Define MLP params
        self.mlp_params = {}
        for layer_idx in range(1, len(self.mlp_layers_sizes)):
            self.mlp_params['W' + str(layer_idx)] = tf.get_variable('W' + str(layer_idx),
                                                                    [self.mlp_layers_sizes[layer_idx - 1], self.mlp_layers_sizes[layer_idx]],
                                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.mlp_params["b" + str(layer_idx)] = tf.get_variable("b" + str(layer_idx),
                                                                    [self.mlp_layers_sizes[layer_idx]],
                                                                    initializer=tf.zeros_initializer())

    def _define_classifier(self, embeddings):
        """
        Define classifier on embeddings vector.

        Args:
            embeddings (np.ndaray of shape [B, E]): embedding of each image, where
                B: batch_size, E: size of an embedding of an image
        Returns:
            (np.ndarray of shape [B, C]): Prediction probability for each class.
        """
        with tf.name_scope("mlp"):
            AX = embeddings
            for layer_idx in range(1, len(self.mlp_layers_sizes) - 1):
                with tf.name_scope("layer_" + str(layer_idx)):
                    AX = tf.nn.tanh(tf.matmul(AX, self.mlp_params['W' + str(layer_idx)]) + self.mlp_params['b' + str(layer_idx)])            

            with tf.name_scope("layer_" + str(len(self.mlp_layers_sizes) - 1)):
                return tf.matmul(AX, self.mlp_params['W' + str(len(self.mlp_layers_sizes) - 1)]) + self.mlp_params['b' + str(len(self.mlp_layers_sizes) - 1)]

    def _calculate_loss(self, predictions, true_labels):
        """
        Calculate loss.

        Args:
            predictions (np.ndaray of shape [B, C]): prediction of a class for each image, where:
                B: batch_size, C: prediction probability for each class.
            true_labels (np.ndaray of shape [B, C]): true labels of for each image, where:
                B: batch_size, C: true label
        Returns:
            (float): Loss of current batch.
        """
        with tf.name_scope("loss"):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=predictions))

    def _define_optimizer(self, loss_function):
        """
        Define optimizer operation.
        """
        with tf.name_scope("optimizer"):
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_function)
