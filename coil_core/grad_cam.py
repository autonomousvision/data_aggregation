"""
This script computes gradcam attentions maps for the images stored as per the standard training data format
"""

import os
import time
import sys
import random
import argparse

import torch
import traceback
import dlib

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint

import matplotlib.pyplot as plt
import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--gpus', type=str, required=True, help='gpu id')
parser.add_argument('--dataset_path', type=str, required=True, help='path to carla dataset')
parser.add_argument('--preload_name', type=str, required=True, help='preload file name')
parser.add_argument('--config', type=str, required=True, help='configuration file')
parser.add_argument('--checkpoint', type=str, required=True, help='saved model checkpoint')
parser.add_argument('--gradcam_path', type=str, required=True, help='path to save gradcam heatmap')
parser.add_argument('--type', type=str, required=True, help='type of evaluation')

args = parser.parse_args()

merge_with_yaml(args.config)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

augmenter = Augmenter(None)
dataset = CoILDataset(args.dataset_path, transform=augmenter, preload_name=args.preload_name)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE, shuffle=False,
										 num_workers=g_conf.NUMBER_OF_LOADING_WORKERS, pin_memory=True)

model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
model = model.cuda()

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
print (len(dataset))

save_dir = os.path.join(args.gradcam_path, args.type)
if not os.path.isdir(save_dir):
	os.mkdir(save_dir)

count = 0
for data in dataloader:

	for i in range(g_conf.BATCH_SIZE):
		controls = data['directions']
		output = model.forward_branch(torch.squeeze(data['rgb']).cuda(), 
									  dataset.extract_inputs(data).cuda(), 
									  controls)
		activations = model.get_perception_activations(torch.squeeze(data['rgb']).cuda())[4].detach()
		# gradcam results in the suppmat are computed using brake
		output[i,2].backward() # backprop from the steer (0), throttle (1) or brake (2)
		gradients = model.get_perception_gradients()
		pooled_gradients = torch.mean(torch.mean(torch.mean(gradients, 3), 2), 0)
		
		for j in range(512): # number of feature maps = 512 for conv4, 256 for conv3
			activations[:,j,:,:] *= pooled_gradients[j]

		
		heatmap = torch.mean(activations, dim=1).squeeze()
		heatmap = np.maximum(heatmap, 0)
		heatmap /= torch.max(heatmap)
		curr_heatmap = heatmap[i]
		curr_heatmap = curr_heatmap.cpu().numpy()

		img = data['rgb'][i].numpy().transpose(1, 2, 0)
		img = np.uint8(255*img)
		curr_heatmap = cv2.resize(curr_heatmap, (img.shape[1], img.shape[0]))
		curr_heatmap = np.uint8(255*curr_heatmap)
		curr_heatmap = cv2.applyColorMap(curr_heatmap, cv2.COLORMAP_JET)
		superimposed_img = np.uint8(curr_heatmap*0.4 + img)

		# plt.imshow(superimposed_img)
		# plt.show()
		cv2.imwrite(os.path.join(save_dir, 'img_%d.jpg'%count), superimposed_img)
		count += 1
		if count%100 == 0:
			print (count)
	
	# specify the number of images to be saved
	if count>=20000:
		break
