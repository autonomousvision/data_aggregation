"""
Script to plot the infraction locations on the town map

Usage: python plot_infractions.py <town_name> <data_type> <data_path>

town_name: Town01 / Town02
data_type: name of the output folder is maps_<data_type>
data_path: folder consisting of measurements.csv and summary.csv files
"""

import math
import os
import sys

from skimage.transform import rescale
from carla.planner import map
from PIL import Image
import numpy as np

DATA_DIR = '/is/sg2/aprakash/Downloads'


def plot_test_image(image, name):
    image_to_plot = Image.fromarray(image.astype("uint8"))
    image_to_plot.save(name)


def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def plot_point(map_image, x, y, colour):
    if (x <map_image.shape[1]  and x > 0) and (y <map_image.shape[0]  and y > 0):
        map_image[x, y] = colour


def plot_on_map(map_image, position, color, size):
    def plot_square(map_image, position, color, size):
        for i in range(0, size):
            for j in range(0, size):
                map_image[int(position[1]) + i, int(position[0]) + j] = color

    for i in range(size):
        plot_square(map_image, position, color, i)


metrics_parameters ={ 'intersection_offroad': {'frames_skip': 10,
	                                     'frames_recount': 20,
	                                     'threshold': 0.3,
	                                     'count': 0
	                                     },
			         'intersection_otherlane': {'frames_skip': 10,
			                                       'frames_recount': 20,
			                                       'threshold': 0.4,
			                                       'count': 0
			                                       },
			         'collision_pedestrians': {'frames_skip': 5,
			                                      'frames_recount': 100,
			                                      'threshold': 300,
			                                      'count': 0 
			                                      },
			         'collision_vehicles': {'frames_skip': 10,
			                                   'frames_recount': 30,
			                                   'threshold': 400,
			                                   'count': 0
			                                   },
			         'collision_other': {'frames_skip': 10,
			                                'frames_recount': 20,
			                                'threshold': 400,
			                                'count': 0
			                                },
		            }

def update_metrics_parameters(color_palete):

	metrics_parameters['intersection_offroad'].update({'color': color_palete[0]})
	metrics_parameters['intersection_otherlane'].update({'color': color_palete[1]})
	metrics_parameters['collision_pedestrians'].update({'color': color_palete[2]})
	metrics_parameters['collision_vehicles'].update({'color': color_palete[3]})
	metrics_parameters['collision_other'].update({'color': color_palete[4]})

	print (metrics_parameters)


def store_infractions(details_matrix, metric, header_details, infraction_points):

	i = metrics_parameters[metric]['frames_skip']

	while i < details_matrix.shape[0]:
		if (details_matrix[i, header_details.index(metric)] - \
			details_matrix[i - metrics_parameters[metric]['frames_skip'], header_details.index(metric)]) > \
			metrics_parameters[metric]['threshold']:

			pos_x = details_matrix[i, header_details.index('pos_x')]
			pos_y = details_matrix[i, header_details.index('pos_y')]
			point = [pos_x, pos_y]
			point[1] = point[1] - 3
			point[0] = point[0] - 2
			point.append(0.0)

			infraction_points.append(point)

			# update the count
			metrics_parameters[metric]['count'] += 1

			i += metrics_parameters[metric]['frames_recount']
		i += 1


def map_infractions(map_image_dots, infraction_points, carla_map, color):
	for point in infraction_points:
        plot_on_map(map_image_dots, carla_map.convert_to_pixel(point), color, 16)


def plot_episodes_tracks(town_name, data_type, data_path):

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(town_name, 0.164, 50)

    meas_file = os.path.join(data_path, 'measurements.csv')

    f = open(meas_file, 'rU')
    header_details = f.readline().strip().split(',')
    f.close()
    print (header_details)
    details_matrix = np.loadtxt(open(meas_file, 'rb'), delimiter=',', skiprows=1)
    print ('got binarized matrix: ', len(details_matrix))
    
    paths_dir = '/is/sg2/aprakash/Documents/maps_%s'%(data_type)

    # Create the paths just in case they don't exist.
    if not os.path.exists(paths_dir):
        os.makedirs(paths_dir)

    # Color pallet for the causes of episodes to end
    color_palete = [
        [255, 0, 0, 255],  # Red for infraction_offroad
        [0, 255, 0, 255],  # Green for infraction_otherlane
        [0, 0, 255, 255],  # Blue for collision_pedestrians
        [255, 255, 0, 255],  # Yellow for collision_vehicles
        [255, 0, 255, 255],  # Magenta for collision_other

    ]

    # updpate the metric parameters to include color_palete
    update_metrics_parameters(color_palete)

    # We instance an image that is going to have all the final position plots
    map_image_dots = Image.open(os.path.join(image_location, town_name + '.png'))
    map_image_dots.load()
    map_image_dots = np.asarray(map_image_dots, dtype="int32")

    for metric in metrics_parameters.keys():
    	infraction_points = []
    	store_infractions(details_matrix, metric, header_details, infraction_points)
    	# print (metric, len(infraction_points))
    	map_infractions(map_image_dots, infraction_points, carla_map, metrics_parameters[metric]['color'])

    map_image_dots = rescale(map_image_dots.astype('float'), 1.0 / 4.0)
    plot_test_image(map_image_dots, os.path.join(paths_dir, 'all_dots_infractions.png'))

    for metric in metrics_parameters.keys():
    	print (metric, metrics_parameters[metric]['count'])


if __name__ == '__main__':

	plot_episodes_tracks(sys.argv[1], sys.argv[2], sys.argv[3])
            
