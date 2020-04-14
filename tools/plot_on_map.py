"""
Script to plot the tracks covered by the driving policy on the town map

Usage: python plot_on_map.py <town_name> <data_type> <data_path>

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



def split_episodes(meas_file):
    """
    The idea is to split the positions assumed by the ego vehicle on every episode.
    Args:
        meas_file: the file containing the measurements.

    Returns:
        a matrix where each vector is a vector of points from the episodes.
        a vector with the travelled distance on each episode

    """
    f = open(meas_file, "rU")
    header_details = f.readline()

    header_details = header_details.strip().split(',')
    # header_details[-1] = header_details[-1][:-2]
    f.close()

    print (header_details)

    details_matrix = np.loadtxt(open(meas_file, "rb"), delimiter=",", skiprows=1)

    #
    #print (details_matrix)
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    #

    episode_positions_matrix = []
    positions_vector = []
    travelled_distances = []
    travel_this_episode = 0
    previous_start_point = details_matrix[0, header_details.index('start_point')]
    previous_end_point = details_matrix[0, header_details.index('end_point')]
    previous_repetition = details_matrix[0, header_details.index('rep')]
    for i in range(1, len(details_matrix)):
        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]

        start_point = details_matrix[i, header_details.index('start_point')]
        end_point = details_matrix[i, header_details.index('end_point')]
        repetition = details_matrix[i, header_details.index('rep')]

        positions_vector.append(point)
        if (previous_start_point != start_point and end_point != previous_end_point) or \
                repetition != previous_repetition:

            travelled_distances.append(travel_this_episode)
            travel_this_episode = 0
            positions_vector.pop()
            episode_positions_matrix.append(positions_vector)
            positions_vector = []

        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point

        previous_start_point = start_point
        previous_end_point = end_point
        previous_repetition = repetition

    return episode_positions_matrix, travelled_distances


def get_causes_of_end(summary_file):
    """
        The dot that finalizes the printing is codified differently depending on the
        cause ( pedestrian, vehicle, timeout, other)

    """
    f = open(summary_file, "rU")
    header_summary = f.readline()

    header_summary = header_summary.strip().split(',')
    # header_summary[-1] = header_summary[-1][:-2]
    f.close()

    summary_matrix = np.loadtxt(open(summary_file, "rb"), delimiter=",", skiprows=1)

    success = summary_matrix[:, header_summary.index('result')]
    end_pedestrian = summary_matrix[:, header_summary.index('end_pedestrian_collision')]
    end_vehicle = summary_matrix[:, header_summary.index('end_vehicle_collision')]
    end_other = summary_matrix[:, header_summary.index('end_other_collision')]

    # print ("end peds ", end_pedestrian, len(end_pedestrian))
    # print ("success ", success, len(success))
    
    all_ends = np.concatenate((np.expand_dims(success, axis=1),
                               np.expand_dims(end_pedestrian, axis=1),
                               np.expand_dims(end_vehicle, axis=1),
                               np.expand_dims(end_other, axis=1)),
                              axis=1)
    no_timeout_pos, end_cause = np.where(all_ends == 1)
    final_end_cause = np.zeros((len(success)))
    final_end_cause[no_timeout_pos] = end_cause + 1

    return final_end_cause


def plot_episodes_tracks(town_name, data_type, data_path):
    meas_file = os.path.join(data_path, 'measurements.csv')

    image_location = map.__file__[:-7]
    carla_map = map.CarlaMap(town_name, 0.164, 50)

    # Split the measurements for each of the episodes
    episodes_positions, travelled_distances = split_episodes(meas_file)

    summary_file = os.path.join(data_path, 'summary.csv')

    # Get causes of end
    end_cause = get_causes_of_end(summary_file)

    print ("End casues ", len(end_cause))
    print (end_cause)

    # Prepare the folder where the results are going to be written
    paths_dir = '/is/rg/avg/aprakash/visualization/maps_%s'%(data_type)

    # Create the paths just in case they don't exist.
    if not os.path.exists(paths_dir):
        os.makedirs(paths_dir)

    count = 0
    
    end_color_palete = [
        [0, 0, 0, 255],  # Black for timeout
        [0, 255, 0, 255],  # Green for success
        [255, 0, 0, 255],  # Red for End pedestrian
        [0, 0, 255, 255],  # Blue for end car
        [0, 255, 255, 255],  # Magenta for end other

    ]
    print ("Number of episodes ", len(episodes_positions))

    # We instance an image that is going to have all the final position plots
    map_image_dots = Image.open(os.path.join(image_location, town_name + '.png'))
    map_image_dots.load()
    map_image_dots = np.asarray(map_image_dots, dtype="int32")

    for episode_vec in episodes_positions:

        map_image = Image.open(os.path.join(image_location, town_name + '.png'))
        map_image.load()
        map_image = np.asarray(map_image, dtype="int32")

        travel_this_episode = 0
        previous_pos = episode_vec[0]
        
        # This is for plotting the path driven by the car.
        for point in episode_vec[1:]:

            travel_this_episode += sldist(point, previous_pos)
            previous_pos = point
            point[1] = point[1] - 3
            point[0] = point[0] - 2
            value = travel_this_episode / travelled_distances[count]

            color_palate_inst = [0 + (value * x) for x in [255, 0, 0]]
            color_palate_inst.append(255)

            point.append(0.0)

            plot_on_map(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 8)

        # print ('episode: ', count)
        if count >= len(end_cause):
            break
        # Plot the end point on the path map
        plot_on_map(map_image, carla_map.convert_to_pixel(point),
                    end_color_palete[int(end_cause[count])], 16)
        # Plot the end point on the map just showing the dots
        plot_on_map(map_image_dots, carla_map.convert_to_pixel(point),
                    end_color_palete[int(end_cause[count])], 16)

        count += 1
        map_image = rescale(map_image.astype('float'), 1.0 / 4.0)
        plot_test_image(map_image, os.path.join(paths_dir, str(count) + '.png'))

    map_image_dots = rescale(map_image_dots.astype('float'), 1.0 / 4.0)
    plot_test_image(map_image_dots, os.path.join(paths_dir, 'all_dots_end.png'))

if __name__ == '__main__':

    plot_episodes_tracks(sys.argv[1], sys.argv[2], sys.argv[3])