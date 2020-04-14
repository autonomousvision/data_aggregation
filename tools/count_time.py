"""
Script to compute total time traversed in the collected data assuming 10 fps at generation
"""

import glob
import re
import argparse

import os
from collections import deque

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

# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Path viewer')
    parser.add_argument('-pt', '--path', default="")
    parser.add_argument('-e', '--st_episode', type=int, default=0, type='starting episode number')
    parser.add_argument('-t', '--end_episode', type=int, default=1e10, type='last episode number')
    parser.add_argument('--episodes', nargs='+', dest='episodes', type=str, default ='all')

    args = parser.parse_args()
    path = args.path

    # By setting episodes as all, it means that all episodes should be visualized
    if args.episodes == 'all':
        episodes_list = glob.glob(os.path.join(path, 'episode_*'))
    else:
        episodes_list = args.episodes
    sort_nicely(episodes_list)

    total_number_of_seconds = 0

    for episode in episodes_list:
        
        if 'episode' not in episode:
            episode = 'episode_' + episode

        episode_number = int(episode.split('_')[-1])
        
        # only count the episodes in the specified range
        if episode_number < int(args.st_episode) or episode_number > int(args.end_episode):
            continue
        
        # Take all the measurements from a list
        measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
        sort_nicely(measurements_list)

        # time is computed assuming 10 fps generation
        if len (measurements_list) > 0:
            data_point_number = len(measurements_list) # total number of frames
            total_number_of_seconds += float(data_point_number)/10.0

    print( 'Total Hours = ',  total_number_of_seconds/3600.0)

    # save_gta_surface(gta_surface)
