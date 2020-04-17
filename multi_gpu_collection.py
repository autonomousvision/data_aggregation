import argparse
import time
import logging
import subprocess
import multiprocessing

from carla.client import make_carla_client
from carla.tcp import TCPConnectionError

from collect import collect


class Arguments():
    def __init__(self, port, number_of_episodes, episode_number, path_name, data_configuration_name, gpu_id, town_name, container_name, mode):
        self.port = port
        self.gpu = gpu_id
        self.host = 'localhost'
        self.number_of_episodes = number_of_episodes
        self.episode_number = episode_number
        self.not_record = False
        self.debug = False
        self.verbose = True
        self.controlling_agent = 'CommandFollower'
        self.data_path = path_name
        self.data_configuration_name = data_configuration_name
        self.town_name = town_name
        self.container_name = container_name
        self.mode = mode


def collect_loop(args):
    try:
        carla_process, out = open_carla(args.port, args.town_name, args.gpu, args.container_name)
        
        while True:
            try:
                with make_carla_client(args.host, args.port) as client:
                    collect(client, args)
                    break

            except TCPConnectionError as error:
                logging.error(error)
                time.sleep(1)

        # KILL CARLA TO AVOID ZOMBIES 
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])

    except KeyboardInterrupt:
        print ('Killed By User')
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])

    except:
        carla_process.kill()
        subprocess.call(['docker', 'stop', out[:-1]])
        
def execute_collector(args):
    p = multiprocessing.Process(target=collect_loop,
                                args=(args,))
    p.start()


# open a carla docker with the container_name
def open_carla(port, town_name, gpu, container_name):
    sp = subprocess.Popen(
        ['docker', 'run', '--rm', '-d', '-p',
         str(port) + '-' + str(port + 2) + ':' + str(port) + '-' + str(port + 2),
         '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + str(gpu), container_name,
         '/bin/bash', 'CarlaUE4.sh', '/Game/Maps/' + town_name, '-windowed',
         '-benchmark', '-fps=10', '-world-port=' + str(port)], shell=False,
        stdout=subprocess.PIPE)

    (out, err) = sp.communicate()

    return sp, out


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Release Data Collectors')
    
    argparser.add_argument('-ids','--ids_gpus', type=str, required=True, help='string containing the gpu ids')
    argparser.add_argument('-n', '--number_collectors', default=1, type=int, help='number of collectors used')
    argparser.add_argument('-e', '--number_episodes', default=200, type=int, help='number of episodes per collector used')
    argparser.add_argument('-g', '--carlas_per_gpu', default=3, type=int, help='number of gpus per carla')
    argparser.add_argument('-s', '--start_episode', default=0, type=int, help='first episode number')
    argparser.add_argument('-d', '--data_configuration_name', default='coil_training_dataset', type=str, help='config file in dataset_configurations')
    argparser.add_argument('-pt', '--data_path', type=str, required=True, help='path used to save the data')
    argparser.add_argument('-ct', '--container_name', default='carlagear', type=str, help='docker container used to collect data')
    argparser.add_argument('-t', '--town_name', default=1, type=int, help='town name (1/2)')
    argparser.add_argument('-m', '--mode', default='expert', type=str, help='data collection mode - expert/dagger/dart')

    args = argparser.parse_args()

    town_name = 'Town0' + str(args.town_name)
    # distribute collectors over the gpus
    for i in range(args.number_collectors):
        port = 10000 + i * 3
        gpu = (int(i / args.carlas_per_gpu))
        gpu_id = args.ids_gpus[gpu % len(args.ids_gpus)]
        print ('using gpu id: ', gpu_id)
        collector_args = Arguments(port, args.number_episodes,
                                   args.start_episode + (args.number_episodes) * (i),
                                   args.data_path,
                                   args.data_configuration_name, 
                                   gpu_id,
                                   town_name,
                                   args.container_name,
                                   args.mode)
        execute_collector(collector_args)
