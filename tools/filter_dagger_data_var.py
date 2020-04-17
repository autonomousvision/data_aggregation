"""
Script for policy-based sampling using uncertainty estimate.
Uncertainty is measured by computing the variance in the predicted controls
of 100 runs of model with test time dropout

Requires:
	var_file: computed variance in the controls with test time dropout
	preload: npy preload file for the entire on-policy data
"""

import os
import sys
import time
import json
import shutil
import argparse
import multiprocessing
import numpy as np

def filter_episode(episode, episode_data):
	if not os.path.isdir(os.path.join(args.target_dir, episode)):
		os.mkdir(os.path.join(args.target_dir, episode))
	
	files = sorted(os.listdir(os.path.join(args.source_dir, episode)))
	if 'metadata.json' in files:
		shutil.copy2(os.path.join(args.source_dir, episode, 'metadata.json'), os.path.join(args.target_dir, episode, 'metadata.json'))

	if 'processed2' in files:
		shutil.copy2(os.path.join(args.source_dir, episode, 'processed2'), os.path.join(args.target_dir, episode, 'processed2'))
			
	for filename in episode_data:

		episode_number = filename.split('_')[-1].split('.')[0]

		central_image = os.path.join(args.source_dir, episode, 'CentralRGB_%s.png'%episode_number)
		left_image = os.path.join(args.source_dir, episode, 'LeftRGB_%s.png'%episode_number)
		right_image = os.path.join(args.source_dir, episode, 'RightRGB_%s.png'%episode_number)

		shutil.copy2(central_image, os.path.join(args.target_dir, episode, 'CentralRGB_%s.png'%episode_number))
		shutil.copy2(left_image, os.path.join(args.target_dir, episode, 'LeftRGB_%s.png'%episode_number))
		shutil.copy2(right_image, os.path.join(args.target_dir, episode, 'RightRGB_%s.png'%episode_number))
		
		measurements_file = os.path.join(args.source_dir, episode, 'measurements_%s.json'%episode_number)

		shutil.copy2(measurements_file, os.path.join(args.target_dir, episode, 'measurements_%s.json'%episode_number))


# this function is used to get the sampled episodes in the first iteration
def get_required_episodes():
	computed_var = np.load(args.var_file)
	preload = np.load(args.preload)

	# take the max variance out of steer, throtte and brake
	max_var = np.max(computed_var, axis=1)
	print (max_var.shape)
	indices_var = np.argsort(max_var)
	required_var = max_var[indices_var[::-1]]
	threshold_index = 72507 # this is selected based on the proportion of data to be sampled in the first iteration
	threshold_var = required_var[threshold_index]
	print (threshold_var)
	new_preload = preload[0][indices_var[::-1]]
	required_preload = new_preload[:threshold_index]

	required_episodes = {}
	for i in range(len(required_preload)):
	    curr_episode, curr_frame = required_preload[i].split('/')
	    if curr_episode in required_episodes:
	        required_episodes[curr_episode].append(curr_frame)
	    else:
	        required_episodes[curr_episode] = [curr_frame]

	print (len(required_episodes))

	return required_episodes

# once the threshold is fixed after the first iteration, use this function for sampling
def get_required_episodes_thres():
	computed_var = np.load(args.var_file)
	preload = np.load(args.preload)
	max_var = np.max(computed_var, axis=1)
	thres = 0.00963
	required_preload = preload[0][max_var>thres]

	'''
	#indices_var = np.argsort(max_var)
	#required_var = max_var[indices_var[::-1]]
	#threshold_index = 72507
	#threshold_var = required_var[threshold_index]
	#print (threshold_var)
	#new_preload = preload[0][indices_var[::-1]]
	#required_preload = new_preload[:threshold_index]
	#print (required_preload)
	'''

	required_episodes = {}
	for i in range(len(required_preload)):
	    curr_episode, curr_frame = required_preload[i].split('/')
	    if curr_episode in required_episodes:
	        required_episodes[curr_episode].append(curr_frame)
	    else:
	        required_episodes[curr_episode] = [curr_frame]

	print (len(required_episodes))

	return required_episodes

def main():

	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []

	if not os.path.isdir(args.target_dir):
		os.mkdir(args.target_dir)

	# episodes_dict = get_required_episodes() # this is used for the first iteration
	episodes_dict = get_required_episodes_thres() # this is used for the sunsequent iteration

	# if 'metadata.json' in episodes_list:
	# 	shutil.copy2(os.path.join(source_dir, 'metadata.json'), os.path.join(target_dir, 'metadata.json'))

	st = time.time()
	for episode in episodes_dict:
		print ('episode: ', episode)
		p = multiprocessing.Process(target=filter_episode, args=(episode, episodes_dict[episode]))
		jobs.append(p)
		p.start()

	for process in jobs:
		process.join()
	print ('time for processing episodes: ', time.time()-st)

	for episode in episodes_dict:
		print (episode, len(episodes_dict[episode]))

if __name__ == '__main__':
	global args
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', type=str, required=True, help='source directory')
	parser.add_argument('--target_dir', type=str, required=True, help='target directory')
	parser.add_argument('--preload', type=str, required=True, help='preload path of required dataset')
	parser.add_argument('--var_file', type=str, required=True, help='path of variance file')

	args = parser.parse_args()
	

	main()