#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import argparse
import logging
import random
import time

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla.client import make_carla_client

from carla.tcp import TCPConnectionError
from carla.carla_game import CarlaGame
from carla.planner import Planner
from carla.agent import HumanAgent, ForwardAgent, CommandFollower, LaneFollower

import modules.data_writer as writer
from modules.noiser import Noiser
from modules.collision_checker import CollisionChecker

# for DAgger and DART data collection
from coil_core import validate_single_model
from drive import coil_agent

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
# This is the number of frames that the car takes to fall from the ground
NUMBER_OF_FRAMES_CAR_FLIES = 25  # multiply by ten


def make_controlling_agent(args, town_name):
    """ 
    Make the controlling agent object depending on what was selected.
    Options:
    Forward Agent: Trivial agent that just accelerate forward.
    Human Agent: Agent controlled by a human driver, currently only by keyboard.
    Command Follower: A* planner followed by a PID controller (used as expert for data collection)
    """

    if args.controlling_agent == "ForwardAgent":
        return ForwardAgent()
    elif args.controlling_agent == "HumanAgent":
        return HumanAgent()
    elif args.controlling_agent == "CommandFollower":
        return CommandFollower(town_name)
    else:
        raise ValueError("Selected Agent Does not exist")


def get_directions(measurements, target_transform, planner):
    """ 
    Function to get the high level commands and the waypoints.
    The waypoints correspond to the local planning, the near path the car has to follow.
    
    Args:
        measurements: measurements file containing global information
        target_transform: target location
        planner: to get the shortest path from current location to target location

    Returns:
        directions: high level navigational commands to follow
    """

    # Get the current position from the measurements
    current_point = measurements.player_measurements.transform

    directions = planner.get_next_command(
        (current_point.location.x,
         current_point.location.y, 0.22),
        (current_point.orientation.x,
         current_point.orientation.y,
         current_point.orientation.z),
        (target_transform.location.x, target_transform.location.y, 0.22),
        (target_transform.orientation.x, target_transform.orientation.y,
         target_transform.orientation.z)
    )

    return directions



def new_episode(client, carla_settings, position, vehicle_pair, pedestrian_pair, set_of_weathers):
    """
    Start a CARLA new episode and generate a target to be pursued on this episode

    Args:
        client: the client connected to CARLA now
        carla_settings: a carla settings object to be used
        position: start position in the town
        vehicle_pair: range of permissible number of vehicles
        pedestrian_pair: range of permissible number of pedestrians
        set_of_weathers: permissible set of weathers

    Returns:
        Returns the configuration of the current episode to be used for data collection
    """

    # Every time the seeds for the episode are different
    number_of_vehicles = random.randint(vehicle_pair[0], vehicle_pair[1])
    number_of_pedestrians = random.randint(pedestrian_pair[0], pedestrian_pair[1])
    weather = random.choice(set_of_weathers)
    carla_settings.set(
        NumberOfVehicles=number_of_vehicles,
        NumberOfPedestrians=number_of_pedestrians,
        WeatherId=weather
    )
    scene = client.load_settings(carla_settings)
    client.start_episode(position)

    return scene.map_name, scene.player_start_spots, weather, number_of_vehicles, number_of_pedestrians, \
           carla_settings.SeedVehicles, carla_settings.SeedPedestrians


def check_episode_has_noise(lat_noise_percent, long_noise_percent):
    """
    Checks if the noise is to be added to the current episode (only for inject triangular perturbations)

    Args:
        lat_noise_percent: % noise to inject in steering
        long_noise_percent: % noise to inject in throttle / brake

    Returns:
        lat_noise: bool value whether to inject noise in the current episode
        long_noise: bool value whether to inject noise in the current episode
    """
    lat_noise = False
    long_noise = False
    if random.randint(0, 101) < lat_noise_percent:
        lat_noise = True

    if random.randint(0, 101) < long_noise_percent:
        long_noise = True
    return lat_noise, long_noise


def reach_timeout(current_time, timeout_period):
    if current_time > timeout_period:
        return True

    return False


def calculate_timeout(start_point, end_point, planner):
    """
    Calucaltes the time limit between start_point and end_point considering a fixed speed of 5 km/hr.
    
    Args:
        start_point: initial position
        end_point: target_position
        planner: to get the shortest part between start_point and end_point

    Returns:
        time limit considering a fixed speed of 5 km/hr
    """
    path_distance = planner.get_shortest_path_distance(
        [start_point.location.x, start_point.location.y, 0.22], [
            start_point.orientation.x, start_point.orientation.y, 0.22], [
            end_point.location.x, end_point.location.y, end_point.location.z], [
            end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    return ((path_distance / 1000.0) / 5.0) * 3600.0 + 10.0


def reset_episode(client, carla_game, settings_module, show_render):
    """
    Reset the episode to a random configuration from the permissible settings

    Args:
        client: carla client object
        carla_game: for visualization
        settings_module: permissible configuration values
        show_render: for visualization

    Returns:
        episode_characteristics: configuration for the current episode
    """

    random_pose = random.choice(settings_module.POSITIONS)
    town_name, player_start_spots, weather, number_of_vehicles, number_of_pedestrians, \
        seeds_vehicles, seeds_pedestrians = new_episode(client,
                                                        settings_module.make_carla_settings(),
                                                        random_pose[0],
                                                        settings_module.NumberOfVehicles,
                                                        settings_module.NumberOfPedestrians,
                                                        settings_module.set_of_weathers)

    # Here when verbose is activated we also show the rendering window.
    carla_game.initialize_game(town_name, render_mode=show_render)
    carla_game.start_timer()

    # An extra planner is needed in order to calculate timeouts
    planner = Planner(town_name)

    carla_game.set_objective(player_start_spots[random_pose[1]])

    player_target_transform = player_start_spots[random_pose[1]]

    last_episode_time = time.time()

    timeout = calculate_timeout(player_start_spots[random_pose[0]],
                                player_target_transform, planner)
    episode_characteristics = {
        "town_name": town_name,
        "player_target_transform": player_target_transform,
        "last_episode_time": last_episode_time,
        "timeout": timeout,
        "weather": weather,
        "number_of_vehicles": number_of_vehicles,
        "number_of_pedestrians": number_of_pedestrians,
        "seeds_vehicles": seeds_vehicles,
        "seeds_pedestrians": seeds_pedestrians,
        "start_pose": random_pose[0],
        "end_pose": random_pose[1]
    }

    return episode_characteristics


def suppress_logs(episode_number):
    if not os.path.exists('_output_logs'):
        os.mkdir('_output_logs')
    sys.stdout = open(os.path.join('_output_logs', 'collect_' + str(os.getpid()) + '_' + str(episode_number) + ".out"), "a", buffering=1)
    sys.stderr = open(os.path.join('_output_logs', 'err_collect_' + str(os.getpid()) + '_' + str(episode_number) + ".out"),"a", buffering=1)


def inject_dart_noise(action, covariance_mat):
    """
    Inject gaussian noise with the computed covariance matrix for DART
    Compute the noisy control as per DART code - https://github.com/BerkeleyAutomation/DART

    Args:
        control: control values to be perturbed
        covariance_mat: covariance mat computed as per DART code

    Returns:
        dart_control: perturbed control values as per DART
    """

    dart_control = np.random.multivariate_normal(action, covariance_mat)

    return dart_control


def get_normalized_covariance_mat(args, settings_module):
    """
    Run the agent to get the covariance matrix for the DART noise model
    
    Args:
        args: current configuration settings

    Returns:
        normalized_covariance_mat: computed covariance matrix
    """

    normalized_covariance_mat = validate_single_model.execute(gpu=args.gpu, yaml_file=settings_module.yaml_config_file)

    return normalized_covariance_mat


def update_controls(control, perturbations, brake=False):
    """
    Update controls with the perturbed values

    Args:
        control: original control values to be updated
        perturbations: perturbed control values
        brake: whether to update brake or not

    Returns:
        control: updated perturbed controls
    """
    control.steer = perturbations.steer
    control.throttle = perturbations.throttle
    if brake:
        control.brake = perturbations.brake

    return control


def collect(client, args):
    """
    The main loop for the data collection process.
    Args:
        client: carla client object
        args: arguments passed on the data collection main.

    Returns:
        None

    """
    # Here we instantiate a sample carla settings. The name of the configuration should be passed as a parameter.
    settings_module = __import__('dataset_configurations.' + (args.data_configuration_name), fromlist=['dataset_configurations'])

    # Suppress output to some logfile, that is useful when running a massive number of collectors
    if not args.verbose:
        suppress_logs(args.episode_number)

    # Instatiate the carlagame debug screen. This is basically an interface to visualize the data collection process
    carla_game = CarlaGame(False, args.debug, WINDOW_WIDTH, WINDOW_HEIGHT, MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)

    # The collision checker , checks for collision at any moment.
    collision_checker = CollisionChecker()

    ##### Start the episode #####
    # This returns all the aspects from the episodes.
    episode_aspects = reset_episode(client, carla_game, settings_module, args.debug)
    planner = Planner(episode_aspects["town_name"])

    # Instantiate the expert agent (CommandFollower), depending on the parameter
    controlling_agent_expert = make_controlling_agent(args, episode_aspects["town_name"])

    if args.mode == 'dagger':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        checkpoint = torch.load(settings_module.model_checkpoint)
        controlling_agent = coil_agent.CoILAgent(checkpoint, episode_aspects['town_name'])

    if args.mode == 'dart':
        normalized_covariance_mat = get_normalized_covariance_mat(args, settings_module)

    # The noise object to add triangular perturbations (in the expert controls) to some episodes is instanced
    longitudinal_noiser = Noiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0)
    lateral_noiser = Noiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5)

    episode_lateral_noise, episode_longitudinal_noise = check_episode_has_noise(settings_module.lat_noise_percent, settings_module.long_noise_percent)

    ##### DATASET writer initialization #####
    # here we make the full path for the dataset that is going to be created.
    # Make dataset path
    writer.make_dataset_path(args.data_path)
    # We start by writing the  metadata for the entire data collection process.
    # That basically involves writing the configuration that was set on the settings module.
    writer.add_metadata(args.data_path, settings_module)
    # Also write the metadata for the current episode
    writer.add_episode_metadata(args.data_path, str(args.episode_number).zfill(5), episode_aspects)

    # We start the episode number with the one set as parameter
    episode_number = args.episode_number
    try:
        image_count = 0
        # The maximum episode is equal to the current episode plus the number of episodes you want to run
        maximun_episode = int(args.episode_number) + int(args.number_of_episodes)
        while carla_game.is_running() and episode_number < maximun_episode:

            # we add the vehicle and the connection outside of the game.
            measurements, sensor_data = client.read_data()

            # run a step for the expert agent
            control_expert, controller_state = controlling_agent_expert.run_step(measurements,
                                                       sensor_data,
                                                       [],
                                                       episode_aspects['player_target_transform'])
            # Get the directions, also important to save those for future training
            directions = get_directions(measurements,
                                        episode_aspects['player_target_transform'], planner)

            # run a step of the trained policy for on-policy data collection
            if args.mode == 'dagger':
                control = controlling_agent.run_step(measurements,
                                                    sensor_data,
                                                    directions,
                                                    episode_aspects['player_target_transform'])

            controller_state.update({'directions': directions})

            # this noise module should be active only in case of 'expert' mode
            # if this is a noisy episode, add noise to the controls
            if episode_longitudinal_noise:
                control_noise, _, _ = longitudinal_noiser.compute_noise(control_expert,
                                            measurements.player_measurements.forward_speed * 3.6)
            else:
                control_noise = control_expert

            if episode_lateral_noise:
                control_noise_f, _, _ = lateral_noiser.compute_noise(control_noise,
                                            measurements.player_measurements.forward_speed * 3.6)
            else:
                control_noise_f = control_noise


            # Set the player position
            # if you want to debug also render everything
            if args.debug:
                objects_to_render = controller_state.copy()
                objects_to_render['player_transform'] = measurements.player_measurements.transform
                objects_to_render['agents'] = measurements.non_player_agents
                objects_to_render["draw_pedestrians"] = args.draw_pedestrians
                objects_to_render["draw_vehicles"] = args.draw_vehicles
                objects_to_render["draw_traffic_lights"] = args.draw_traffic_lights
                # Comment the following two lines to see the waypoints and routes.
                objects_to_render['waypoints'] = None
                objects_to_render['route'] = None

                # Render with the provided map
                carla_game.render(sensor_data['CameraRGB'], objects_to_render)

            # Check two important conditions for the episode, if it has ended
            # and if the episode was a success
            episode_ended = collision_checker.test_collision(measurements.player_measurements) or \
                            reach_timeout(measurements.game_timestamp / 1000.0,
                                          episode_aspects["timeout"]) or \
                            carla_game.is_reset(measurements.player_measurements.transform.location)
            episode_success = not (collision_checker.test_collision(
                                   measurements.player_measurements) or
                                   reach_timeout(measurements.game_timestamp / 1000.0,
                                                 episode_aspects["timeout"]))

            # Check if there is collision. Start a new episode if there is a collision but repeat the same by not incrementing episode number.
            if episode_ended:
                if episode_success:
                    episode_number += 1
                else:
                    if args.mode == 'expert':
                        # If the episode did go well and we were recording, delete this episode
                        if not args.not_record:
                            writer.delete_episode(args.data_path, str(episode_number-1).zfill(5))
                    else:
                        # also collect failed episodes as they contain critical states not present in expert data
                        episode_number += 1

                episode_lateral_noise, episode_longitudinal_noise = check_episode_has_noise(settings_module.lat_noise_percent, settings_module.long_noise_percent)

                # We reset the episode and receive all the characteristics of this episode.
                episode_aspects = reset_episode(client, carla_game, settings_module, args.debug)

                writer.add_episode_metadata(args.data_path, str(episode_number).zfill(5),episode_aspects)

                # Reset the image count
                image_count = 0
            
            # We do this to avoid the frames that the car is coming from the sky.
            if image_count >= NUMBER_OF_FRAMES_CAR_FLIES and not args.not_record:
                st = time.time()
                writer.add_data_point(measurements, control_expert, control_noise_f, sensor_data,
                                      controller_state,
                                      args.data_path, str(episode_number).zfill(5),
                                      str(image_count - NUMBER_OF_FRAMES_CAR_FLIES),
                                      settings_module.sensors_frequency)
            
            if args.mode == 'dart':
                # inject noise using the normalized_covariance_mat computed for DART
                controls_to_be_perturbed = np.asarray([control_noise_f.steer, control_noise_f.throttle, control_noise_f.brake])
                dart_perturbed_controls = inject_dart_noise(controls_to_be_perturbed, normalized_covariance_mat)
                control_noise_f = update_controls(control_noise_f, dart_perturbed_controls)

            # End the loop by sending controller
            if args.mode == 'expert' or args.mode == 'dart':
                client.send_control(control_noise_f)
            elif args.mode == 'dagger':
                client.send_control(control)

            # Add one more image to the counting
            image_count += 1

    except TCPConnectionError as error:
        """
        If there is any connection error we delete the current episode, 
        This avoid incomplete episodes
        """
        import traceback
        traceback.print_exc()
        if not args.not_record:
            writer.delete_episode(args.data_path, str(episode_number).zfill(5))

        raise error

    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        if not args.not_record:
            writer.delete_episode(args.data_path, str(episode_number).zfill(5))


def main():
    """
    The main function of the data collection process

    """
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-pt','--data-path', metavar='H', default='.', dest='data_path', help=' Where the recorded data will be placed')
    argparser.add_argument('--data-configuration-name', metavar='H', default='coil_training_dataset_singlecamera', dest='data_configuration_name', help=' Name of the data configuration file that should be place on .dataset_configurations')
    argparser.add_argument('-c', '--controlling_agent', default='CommandFollower',
        help='the controller that is going to be used by the main vehicle.'
             ' Options: '
             ' HumanAgent - Control your agent with a keyboard.'
             ' ForwardAgent - A trivial agent that goes forward'
             ' CommandFollower - A lane follower agent that follow commands from the planner')
    argparser.add_argument('-db', '--debug', action='store_true', help='enable the debug screen mode, on this mode a rendering screen will show'
             'information about the agent')
    argparser.add_argument('-dp', '--draw-pedestrians', dest='draw_pedestrians', action='store_true', help='add pedestrians to the debug screen')
    argparser.add_argument('-dv', '--draw-vehicles', dest='draw_vehicles', action='store_true', help='add vehicles dots to the debug screen')
    argparser.add_argument('-dt', '--draw-traffic-lights', dest='draw_traffic_lights', action='store_true', help='add traffic lights dots to the debug screen')
    argparser.add_argument('-nr', '--not-record', action='store_true', default=False, help='flag for not recording the data ( Testing purposes)')
    argparser.add_argument('-e', '--episode-number', metavar='E', dest='episode_number', default=0, type=int, help='The episode number that it will start to record.')
    argparser.add_argument('-n', '--number-episodes', metavar='N', dest='number_of_episodes', default=999999999, help='The number of episodes to run, default infinite.')
    argparser.add_argument('-m', '--mode', default='expert', type=str, help='data collection mode - expert/dagger/dart')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                collect(client, args)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
