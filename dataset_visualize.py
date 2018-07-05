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


import sys
import cv2
import numpy as np
from utils import data_generator as gen


def main(argv):

    # Some variables ==========================================================
    BORDER = 5
    INSTANCES = 5
    MAX_CLASSES = 5
    IMAGE_SIZE = 128
    CATEGORY = 'breeds' #species / breeds
    
    # Load images =============================================================
    generator = gen.OxfordIIITPets(colorspace='BGR', train_size=0.8)
    images = generator.get_random_images(train=True, instance_number=INSTANCES, category=CATEGORY, image_size=IMAGE_SIZE)
    class_count = np.min([len(images), MAX_CLASSES])
    
#     # Optional image transformations ==========================================
#     for class_idx in range(class_count):
#         for instance_idx in range(INSTANCES):
#             images[class_idx][instance_idx] = cv2.medianBlur(images[class_idx][instance_idx], 5)
    
    # Create canvas and fill it ===============================================
    canvas = np.ones(shape=(class_count * IMAGE_SIZE + (class_count + 1) * BORDER,
                            INSTANCES * IMAGE_SIZE + (INSTANCES + 1) * BORDER, 3), dtype=np.uint8) * 255
    for class_idx in range(class_count):
        row = class_idx * IMAGE_SIZE + (class_idx + 1) * BORDER
        for instance_idx in range(INSTANCES):
            col = instance_idx * IMAGE_SIZE + (instance_idx + 1) * BORDER
            canvas[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE, :] = images[class_idx][instance_idx]

    # Show images =============================================================
    cv2.imshow('examples', canvas)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
