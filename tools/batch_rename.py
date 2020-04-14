"""
    Script to rename all the files from a folder. it forces the numbering to start on 0
"""

from __future__ import print_function

import os
import re
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Name changer')
    parser.add_argument('-pt', '--path', default="")
    args = parser.parse_args()
    path = args.path

    count = 0
    file_list = sorted(os.listdir(path))

    print (file_list)

    for filename in file_list:
            print (filename)
            newfilename = os.path.join(path, 'episode_'+str(count+60).zfill(5))

            os.rename(os.path.join(path, filename), newfilename )
            count +=1

