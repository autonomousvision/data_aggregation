"""
Script to sample critical states from the on-policy data (Sampling Methods - AE(brake), AE(all)), Rank, IT (task-based)

Usage: python filter_dagger_data.py <source_dir> <target_dir> <start_episode_num> <end_episode_num>
"""

import os
import sys
import time
import json
import shutil
import argparse
import multiprocessing
import numpy as np

DATA_DIR = '/is/rg/avg/aprakash'

source_dir = os.path.join(DATA_DIR, sys.argv[1])
target_dir = os.path.join(DATA_DIR, sys.argv[2])

def filter_data(episode, return_dict):
	if not os.path.isdir(os.path.join(target_dir, episode)):
		os.mkdir(os.path.join(target_dir, episode))
	count = 0
	total = 0
	episode_data = sorted(os.listdir(os.path.join(DATA_DIR, sys.argv[1], episode)))
			
	for file_name in episode_data:
		
		if 'metadata' in file_name:
			shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))

		if 'processed2' in file_name:
			shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))
		
		if 'measurements_' in file_name:
			total += 1
			with open(os.path.join(DATA_DIR, sys.argv[1], episode, file_name)) as file:
				measurement_data = json.load(file)
			file.close()

			expert_throttle = float(measurement_data['throttle'])
			agent_throttle = float(measurement_data['throttle_noise'])
			throttle_change = abs(expert_throttle - agent_throttle)
			
			expert_steer = float(measurement_data['steer'])
			agent_steer = float(measurement_data['steer_noise'])
			steer_change = abs(expert_steer - agent_steer)

			expert_brake = float(measurement_data['brake'])
			agent_brake = float(measurement_data['brake_noise'])
			brake_change = abs(expert_brake - agent_brake)

			# specify the sampling condition here - AE(brake) or AE(all) or Rank or IT (task-based)
			# for task-based sampling, condition is measurement_data['directions'] == 3 or 4 or 5
			# set the threshold for the first iteration based on the proportion of the data to be sampled
			# in the first iteration, then keep the threshold fixed for subsequent iterations
			if brake_change > 0.2: # set the sampling condition and threshold here as required, this is just an example
				count += 1
				
				# copy the data to other directory
				episode_number = file_name.split('.')[0].split('_')[-1]

				central_image = os.path.join(DATA_DIR, sys.argv[1], episode, 'CentralRGB_%s.png'%episode_number)
				left_image = os.path.join(DATA_DIR, sys.argv[1], episode, 'LeftRGB_%s.png'%episode_number)
				right_image = os.path.join(DATA_DIR, sys.argv[1], episode, 'RightRGB_%s.png'%episode_number)

				shutil.copy2(central_image, os.path.join(target_dir, episode, 'CentralRGB_%s.png'%episode_number))
				shutil.copy2(left_image, os.path.join(target_dir, episode, 'LeftRGB_%s.png'%episode_number))
				shutil.copy2(right_image, os.path.join(target_dir, episode, 'RightRGB_%s.png'%episode_number))

				# central_seg = os.path.join(DATA_DIR, sys.argv[1], episode, 'CentralSemanticSeg_%s.png'%episode_number)
				# left_seg = os.path.join(DATA_DIR, sys.argv[1], episode, 'LeftSemanticSeg_%s.png'%episode_number)
				# right_seg = os.path.join(DATA_DIR, sys.argv[1], episode, 'RightSemanticSeg_%s.png'%episode_number)

				# shutil.copy2(central_seg, os.path.join(target_dir, episode, 'CentralSemanticSeg_%s.png'%episode_number))
				# shutil.copy2(left_seg, os.path.join(target_dir, episode, 'LeftSemanticSeg_%s.png'%episode_number))
				# shutil.copy2(right_seg, os.path.join(target_dir, episode, 'RightSemanticSeg_%s.png'%episode_number))

				# central_depth = os.path.join(DATA_DIR, sys.argv[1], episode, 'CentralDepth_%s.png'%episode_number)
				# left_depth = os.path.join(DATA_DIR, sys.argv[1], episode, 'LeftDepth_%s.png'%episode_number)
				# right_depth = os.path.join(DATA_DIR, sys.argv[1], episode, 'RightDepth_%s.png'%episode_number)

				# shutil.copy2(central_depth, os.path.join(target_dir, episode, 'CentralDepth_%s.png'%episode_number))
				# shutil.copy2(left_depth, os.path.join(target_dir, episode, 'LeftDepth_%s.png'%episode_number))
				# shutil.copy2(right_depth, os.path.join(target_dir, episode, 'RightDepth_%s.png'%episode_number))

				shutil.copy2(os.path.join(source_dir, episode, file_name), os.path.join(target_dir, episode, file_name))
				

	return_dict[episode] = (count, total)

def main():

	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []

	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

	episodes_list = sorted(os.listdir(os.path.join(DATA_DIR, sys.argv[1])))

	if 'metadata.json' in episodes_list:
		shutil.copy2(os.path.join(source_dir, 'metadata.json'), os.path.join(target_dir, 'metadata.json'))

	st = time.time()
	for episode in episodes_list:
		# print ('episode: ', episode)
		episode_num = int(episode.split('_')[-1])
		if os.path.isdir(os.path.join(DATA_DIR, sys.argv[1], episode)) \
				 and episode_num >= int(sys.argv[3]) and episode_num <= int(sys.argv[4]):
			print ('episode: ', episode)
			p = multiprocessing.Process(target=filter_data, args=(episode, return_dict))
			jobs.append(p)
			p.start()

	for process in jobs:
		process.join()
	print ('time for processing episodes: ', time.time()-st)

	print ('total episodes: ', len(return_dict))

	# compute total sampled data
	total_deviant_frames = 0.0
	total_frames = 0.0
	for key, value in return_dict.items():
		deviation = float(value[0])/float(value[1])
		print (key, value, deviation)
		total_deviant_frames += value[0]
		total_frames += value[1]

	print (total_deviant_frames, total_frames, total_deviant_frames/total_frames)

if __name__ == '__main__':
	
	main()