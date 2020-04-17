#!/usr/bin/env python

"""
Script to process the data from a resolution of 800x600 to 200x88
"""

import json
import numpy as np
import glob
import re
import scipy
import multiprocessing

import argparse
from PIL import Image
import math

import time
import os
from collections import deque
import scipy.ndimage

""" Position to cut the image before reshapping """
""" This is used to cut out the sky (Kind of useless for learning) """
IMAGE_CUT = [90, 485]


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def join_classes(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for key, value in join_dic.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


def join_classes_for(labels_image, join_dic):
    compressed_labels_image = np.copy(labels_image)
    for i in range(labels_image.shape[0]):
        for j in range(labels_image.shape[1]):
            compressed_labels_image[i, j, 0] = join_dic[labels_image[i, j, 0]]

    return compressed_labels_image


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]


def sort_nicely(l):
    """ 
    Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def reshape_images(image_type, episode, data_point_number):
    """
    Function for reshaping all the images of an episode and save it again in the same location

    Args:
        image_type: The type of images that is going to be reshaped - Depth / RGB / SemanticSeg
        episode: current episode number
        data_point_number: current frame number

    """

    if image_type == 'SemanticSeg':
        interp_type = 'nearest'
    else:
        interp_type = 'bicubic'

    center_name = 'Central' + image_type + '_' + data_point_number + '.png'
    left_name = 'Left' + image_type + '_' + data_point_number + '.png'
    right_name = 'Right' + image_type + '_' + data_point_number + '.png'

    center = scipy.ndimage.imread(os.path.join(episode, center_name))
    left = scipy.ndimage.imread(os.path.join(episode, left_name))
    right = scipy.ndimage.imread(os.path.join(episode, right_name))
    
    if center.shape[0] == 600:
        center = center[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
        center = scipy.misc.imresize(center, (88, 200), interp=interp_type)
        scipy.misc.imsave(os.path.join(episode, center_name), center)

    if left.shape[0] == 600:
        left = left[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
        left = scipy.misc.imresize(left, (88, 200), interp=interp_type)
        scipy.misc.imsave(os.path.join(episode, left_name), left)

    if right.shape[0] == 600:
        right = right[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
        right = scipy.misc.imresize(right, (88, 200), interp=interp_type)
        scipy.misc.imsave(os.path.join(episode, right_name), right)

    
# multiprocessing module
def process_fn(args, episode):

    if os.path.exists(os.path.join(episode, "checked")) or os.path.exists(
            os.path.join(episode, "processed2")) \
            or os.path.exists(os.path.join(episode, "bad_episode")):
        # Episode was not checked. So we dont load it.
        print(" This episode was already checked ")
        return
    
    # Take all the measurements from a list
    try:
        measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
        sort_nicely(measurements_list)
        print (" Purging other data")
        print ("Lidar")
        purge(episode, "Lidar*")

        # print (episode)
        if args.delete_depth:
            print ("***Depth***")
            purge(episode, "CentralDepth*")
            purge(episode, "LeftDepth*")
            purge(episode, "RightDepth*")

        if args.delete_semantic_segmentation:
            print ("***Purging SemanticSeg***")
            purge(episode, "CentralSemanticSeg*")
            purge(episode, "LeftSemanticSeg*")
            purge(episode, "RightSemanticSeg*")

        bad_episode = False
        if len(measurements_list) <= 1:
            print (" Episode is empty")
            purge(episode, '.')
            bad_episode = True
            return

        for measurement in measurements_list[:-3]:

            data_point_number = measurement.split('_')[-1].split('.')[0]
            
            with open(measurement) as f:
                measurement_data = json.load(f)

            reshape_images("RGB", episode, data_point_number)
            if not args.delete_depth:
                reshape_images("SemanticSeg", episode, data_point_number)

            if not args.delete_depth:
                reshape_images("Depth", episode, data_point_number)

            if 'forwardSpeed' in  measurement_data['playerMeasurements']:
                speed = measurement_data['playerMeasurements']['forwardSpeed']
            else:
                speed = 0


        for measurement in measurements_list[-3:]:
            data_point_number = measurement.split('_')[-1].split('.')[0]
            purge(episode, "CentralRGB_" + data_point_number + '.png')
            purge(episode, "LeftRGB_" + data_point_number + '.png')
            purge(episode, "RightRGB_" + data_point_number + '.png')
            purge(episode, "CentralSemanticSeg_" + data_point_number + '.png')
            purge(episode, "LeftSemanticSeg_" + data_point_number + '.png')
            purge(episode, "RightSemanticSeg_" + data_point_number + '.png')
            purge(episode, "CentralDepth_" + data_point_number + '.png')
            purge(episode, "LeftDepth_" + data_point_number + '.png')
            purge(episode, "RightDepth_" + data_point_number + '.png')
            os.remove(measurement)

        if not bad_episode:
            done_file = open(os.path.join(episode, "processed2"), 'w')
            done_file.close()

    except:
        import traceback
        traceback.print_exc()
        print (" Error on processing")
        done_file = open(os.path.join(episode, "bad"), 'w')
        done_file.close()

        return


    # The last is deleted
    data_point_number = measurements_list[-1].split('_')[-1].split('.')[0]

    purge(episode, "CentralRGB_" + data_point_number + '.png')
    purge(episode, "LeftRGB_" + data_point_number + '.png')
    purge(episode, "RightRGB_" + data_point_number + '.png')
    


# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')

    parser.add_argument('-pt', '--path', default="", help='path of the folder containing the data')
    parser.add_argument('--episodes', nargs='+', dest='episodes', type=str, default='all')
    parser.add_argument('-s', '--start_episode', default=0, type=int, help=' the first episode') # don't modify this
    parser.add_argument('-e', '--episode_st', type=int, help='episode number from where to start')
    parser.add_argument('-t', '--terminate',type=int, help='episode number where to terminate')
    """ Pass this extra arguments to delete the semantic segmenation labels"""
    parser.add_argument('-ds', '--delete-semantic-segmentation', dest='delete_semantic_segmentation', action='store_true',)
    """ Pass this extra arguments to delete the depth labels"""
    parser.add_argument('-dd', '--delete-depth', dest='delete_depth', action='store_true',)

    args = parser.parse_args()

    # By setting episodes as all, it means that all episodes should be visualized
    if args.episodes == 'all':
        episodes_list = glob.glob(os.path.join(args.path, 'episode_*'))
        sort_nicely(episodes_list)
    else:
        episodes_list = args.episodes

    data_configuration_name = 'coil_training_dataset'
    print ( data_configuration_name)
    print ('dataset_configurations.' + (data_configuration_name) )
    settings_module = __import__('dataset_configurations.' + (data_configuration_name),
                                 fromlist=['dataset_configurations'] )

    # st = time.time()
    
    pool = multiprocessing.Pool()
    for episode in episodes_list[args.start_episode:]:
        
        if 'episode' not in episode:
            episode = 'episode_' + episode

        episode_number = int(episode.split('_')[-1])
        # only process the episodes in the specified range
        if episode_number < args.episode_st or episode_number > args.terminate:
            continue

        print ('Episode ', episode)
        
        p = multiprocessing.Process(target=process_fn, args=(args, episode))
        p.start()
    
