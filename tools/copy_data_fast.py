"""
Script to copy data fast using multiprocessing and shutil

Usage: python copy_data_fast.py <source_dir> <target_dir> <start_episode_num> <end_episode_num>
"""

import os
import sys
import numpy as np
import shutil
import time
import multiprocessing

source_dir = sys.argv[1]
target_dir = sys.argv[2]

def copy_episodes(episode):
	if not os.path.isdir(os.path.join(target_dir, episode)):
		os.mkdir(os.path.join(target_dir, episode))

	episode_data = sorted(os.listdir(os.path.join(source_dir, episode)))

	for file_name in episode_data:
		
		if 'metadata' in file_name:
			shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))

		if 'processed2' in file_name:
			shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))
		
		if 'measurements_' in file_name:
				
			# copy the data to other directory
			episode_number = file_name.split('.')[0].split('_')[-1]

			central_image = os.path.join(source_dir, episode, 'CentralRGB_%s.png'%episode_number)
			left_image = os.path.join(source_dir, episode, 'LeftRGB_%s.png'%episode_number)
			right_image = os.path.join(source_dir, episode, 'RightRGB_%s.png'%episode_number)

			shutil.copy2(central_image, os.path.join(target_dir, episode, 'CentralRGB_%s.png'%episode_number))
			shutil.copy2(left_image, os.path.join(target_dir, episode, 'LeftRGB_%s.png'%episode_number))
			shutil.copy2(right_image, os.path.join(target_dir, episode, 'RightRGB_%s.png'%episode_number))

			shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))
			



if __name__ == '__main__':

	episodes_list = sorted(os.listdir(source_dir))

	jobs = []

	st = time.time()
	for episode in episodes_list:
		if os.path.isdir(os.path.join(source_dir, episode)):
			episode_number = int(episode.split('_')[-1])

			if episode_number >= int(sys.argv[3]) and episode_number <= int(sys.argv[4]):
				print (episode)
				p = multiprocessing.Process(target=copy_episodes, args=(episode,))
				jobs.append(p)
				p.start()


	for process in jobs:
		process.join()

	print ('total time taken: ', time.time()-st)
