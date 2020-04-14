"""
Script to generate video from a set of images in an episode

data_type: folder containing the data
episode_number: specific episode number containing the images
"""

from PIL import Image, ImageFont, ImageDraw
import csv
import os
import sys
import json
import numpy as np
import skvideo.io

DATA_DIR = '/is/rg/avg/aprakash'

if __name__ == '__main__':
	data_type = sys.argv[1]
	episode_number = sys.argv[2]

	episode_path = os.path.join(DATA_DIR, data_type, 'episode_%s'%(episode_number))

	episode_data = sorted(os.listdir(episode_path))

	measurements_data = []
	expert_steer = []
	expert_throttle = []
	expert_brake = []
	agent_steer = []
	agent_throttle = []
	agent_brake = []
	directions = []
	speed_module = []
	central_image_path = []

	for file_name in episode_data:
		if 'measurements_' in file_name:
			# print (file_name)
			frame_number = file_name.split('_')[-1].split('.')[0]
			with open(os.path.join(episode_path, file_name)) as file:
				json_data = json.load(file)
			file.close()

			measurements_data.append(json_data)

			expert_steer.append(float(json_data['steer']))
			expert_throttle.append(float(json_data['throttle']))
			expert_brake.append(float(json_data['brake']))

			agent_steer.append(float(json_data['steer_noise']))
			agent_throttle.append(float(json_data['throttle_noise']))
			agent_brake.append(float(json_data['brake_noise']))

			# directions.append(json_data['directions'])
			if 'playerMeasurements' in json_data and 'forwardSpeed' in json_data['playerMeasurements']:
				speed_module.append(float(json_data['playerMeasurements']['forwardSpeed']))
			else:
				speed_module.append(0)

			image_path = os.path.join(episode_path, 'CentralRGB_%s.png'%(frame_number))
			central_image_path.append(image_path)


	# print (len(measurements_data), len(central_image_path), len(expert_steer), len(expert_throttle), len(expert_brake),
	# 	   len(agent_steer), len(agent_throttle), len(agent_brake), len(directions), len(speed_module))
	if not os.path.isdir(os.path.join(DATA_DIR, 'videos')):
		os.mkdir(os.path.join(DATA_DIR, 'videos'))

	writer = skvideo.io.FFmpegWriter(os.path.join(DATA_DIR, 'videos', '%s_episode_%s.mp4'%(data_type, episode_number)), inputdict={'-r': '10'}, outputdict={'-r': '10'})
	for i in range(1, len(measurements_data)):
		img = Image.open(central_image_path[i])
		helvetica = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", size=20)
		d = ImageDraw.Draw(img)
		text_color = (255, 255, 255)

		location = (40, 10)
		d.text(location, "agent_steer = %0.4f"%agent_steer[i], font=helvetica, fill=text_color)

		location = (40, 35)
		d.text(location, "agent_throttle = %0.4f"%agent_throttle[i], font=helvetica, fill=text_color)

		location = (40, 60)
		d.text(location, "agent_brake = %0.4f"%agent_brake[i], font=helvetica, fill=text_color)

		location = (300, 10)
		d.text(location, "expert_steer = %0.4f"%expert_steer[i], font=helvetica, fill=text_color)

		location = (300, 35)
		d.text(location, "expert_throttle = %0.4f"%expert_throttle[i], font=helvetica, fill=text_color)

		location = (300, 60)
		d.text(location, "expert_brake = %0.4f"%expert_brake[i], font=helvetica, fill=text_color)

		location = (40, 85)
		d.text(location, "vehicle_speed = %0.4f"%speed_module[i], font=helvetica, fill=text_color)

		writer.writeFrame(img)

	writer.close()
