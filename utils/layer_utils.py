# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net
    
    # first two conv2d layers
    net1 = conv2d(inputs, 32,  3, strides=1)   # 1
    net2 = conv2d(net1, 64,  3, strides=2)      # 2

    # res_block * 1
    net3 = res_block(net2, 32)                  # 3

    net4 = conv2d(net3, 128, 3, strides=2)      # 4

    # res_block * 2                           # 5
    net = net4
    for i in range(2):
        net = res_block(net, 64)

    net5 = net
    net6 = conv2d(net5, 256, 3, strides=2)      # 6
 
    # res_block * 8                           # 7
    net = net6
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net                      
    net7 = net
    net8 = conv2d(net7, 512, 3, strides=2)     # 8

    # res_block * 8                          # 9
    net = net8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net9 = net
    net10 = conv2d(net9, 1024, 3, strides=2)    # 10

    # res_block * 4                          # 11
    net = net10
    for i in range(4):
        net = res_block(net, 512)
    
    route_3 = net
    net11 = net

    return route_1, route_2, route_3, net1, net3, net5


def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs



