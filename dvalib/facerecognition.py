"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from collections import defaultdict
import time, os,logging
import tensorflow as tf
import numpy as np
from .facenet import facenet
from .facenet.align import detect_face
import random, json
from time import sleep


def align(image_paths, output_dir, image_size=182, margin=44, gpu_memory_fraction=0.2):
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    aligned_paths = defaultdict(list)
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    output_class_dir = output_dir
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    for image_path in image_paths:
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        logging.info(image_path)
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            logging.info(errorMessage)
        else:
            if img.ndim < 2:
                logging.info('Unable to align "%s"' % image_path)
                continue
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det_all = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                for boxindex in range(nrof_faces):
                    det = np.squeeze(det_all[boxindex, :])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    nrof_successfully_aligned += 1
                    output_filename = os.path.join(output_class_dir,"face_"+filename+'_'+str(boxindex)+'.jpg')
                    misc.imsave(output_filename, scaled)
                    aligned_paths[image_path].append((output_filename,bb))
            else:
                logging.info('Unable to align "%s"' % image_path)
    logging.info('Total number of images: %d' % nrof_images_total)
    logging.info('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    return aligned_paths


