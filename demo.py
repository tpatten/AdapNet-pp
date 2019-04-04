''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import argparse
import importlib
import os
import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import cv2
import NYU40_colors

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/scannet_test.config')
PARSER.add_argument('-f', '--file', default='example.jpg')

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = np.array(NYU40_colors.create_color_palette_legend())

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def test_func(config, image_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    resnet_name = 'resnet_v2_50'

    with tf.variable_scope(resnet_name):
        model = model_func(num_classes=config['num_classes'], training=False)
        images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        model.build_graph(images_pl)

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:', len(import_variables)
    saver = tf.train.Saver(import_variables)
    print 'checkpoint', config['checkpoint']
    saver.restore(sess, config['checkpoint'])

    input_img = cv2.imread(image_file)
    img = np.zeros(shape=(1, 480, 640, 3))
    img[0][:] = input_img
    img = img.astype(float)
    feed_dict = {images_pl : img}
    probabilities = sess.run([model.softmax], feed_dict=feed_dict)
    prediction = np.argmax(probabilities[0], 3)
    colored_prediction = label_to_color_image(prediction[0])
    #print 'input_img', input_img.shape, type(input_img), input_img.dtype
    #print 'img', img.shape, type(img), img.dtype
    #print 'prediction', prediction.shape, type(prediction), prediction.dtype
    #print 'colored_prediction', colored_prediction.shape, type(colored_prediction), colored_prediction.dtype
    vis_img = np.vstack((img[0].astype(np.uint8), colored_prediction))
    plt.imshow(vis_img)
    plt.show()

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print '--config config_file_address missing'
        return
    if args.file:
        image_file = args.file
    else:
        print '--file image file missing'
        return
    print 'config_file', args.config
    print 'image_file', image_file
    test_func(config, image_file)

if __name__ == '__main__':
    main()
