"""
Code to compute the covariance matrix required for DART
"""

import os
import time
import sys
import random

import numpy as np
import torch
import traceback
import dlib
import argparse

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint


def write_waypoints_output(iteration, output):

    for i in range(g_conf.BATCH_SIZE):
        steer = 0.7 * output[i][3]

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        coil_logger.write_on_csv(iteration, [steer,
                                            output[i][1],
                                            output[i][2]])


def write_regular_output(iteration, output):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i][0],
                                            output[i][1],
                                            output[i][2]])



def execute(gpu, exp_batch='nocrash', exp_alias='resnet34imnet10S1', suppress_output=True, yaml_file=None):
    latest = None
    # try:
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # At this point the log file with the correct naming is created.
    path_to_yaml_file = os.path.join('configs', exp_batch, exp_alias+'.yaml')
    if yaml_file is not None:
        path_to_yaml_file = os.path.join(yaml_file, exp_alias+'.yaml')
    merge_with_yaml(path_to_yaml_file)
    # The validation dataset is always fully loaded, so we fix a very high number of hours
    # g_conf.NUMBER_OF_HOURS = 10000 # removed to simplify code
    
    """
    # commenting this segment to simplify code, uncomment if necessary
    set_type_of_process('validation', dataset_name)

    if not os.path.exists('_output_logs'):
        os.mkdir('_output_logs')

    if suppress_output:
        sys.stdout = open(os.path.join('_output_logs',
                                       exp_alias + '_' + g_conf.PROCESS_NAME + '_'
                                       + str(os.getpid()) + ".out"),
                          "a", buffering=1)
        sys.stderr = open(os.path.join('_output_logs',
                          exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                       + str(os.getpid()) + ".out"),
                          "a", buffering=1)
    """

    # Define the dataset. This structure is has the __get_item__ redefined in a way
    # that you can access the HDFILES positions from the root directory as a in a vector.
    
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.DART_COVMAT_DATA) # dataset used for computing dart covariance matrix

    augmenter = Augmenter(None)

    # Definition of the dataset to be used. Preload name is just the validation data name
    print ('full dataset path: ', full_dataset)
    dataset = CoILDataset(full_dataset, transform=augmenter, preload_name=g_conf.DART_COVMAT_DATA) # specify DART_COVMAT_DATA in the config file


    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                              pin_memory=True)

    model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)

    """ removing this segment to simplify code
    # The window used to keep track of the trainings
    l1_window = []
    latest = get_latest_evaluated_checkpoint()
    if latest is not None:  # When latest is noe
        l1_window = coil_logger.recover_loss_window(g_conf.DART_COVMAT_DATA, None)
    """
    
    model.cuda()

    best_mse = 1000
    best_error = 1000
    best_mse_iter = 0
    best_error_iter = 0

    # modified validation code from here to run a single model checkpoint
    # used for computing the covariance matrix with the DART model checkpoint
    checkpoint = torch.load(g_conf.DART_MODEL_CHECKPOINT) # specify DART_MODEL_CHECKPOINT in the config file
    checkpoint_iteration = checkpoint['iteration']
    print("Validation loaded ", checkpoint_iteration)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    accumulated_mse = 0
    accumulated_error = 0
    iteration_on_checkpoint = 0

    # considering steer, throttle & brake so 3x3 matrix
    normalized_covariate_shift = torch.zeros(3,3)

    print ('data_loader size: ', len(data_loader))
    for data in data_loader:

        # Compute the forward pass on a batch from the validation dataset
        controls = data['directions']
        output = model.forward_branch(torch.squeeze(data['rgb']).cuda(),
                                      dataset.extract_inputs(data).cuda(),
                                      controls)

        """ removing this segment to simplify code
        # It could be either waypoints or direct control
        if 'waypoint1_angle' in g_conf.TARGETS:
            write_waypoints_output(checkpoint_iteration, output)
        else:
            write_regular_output(checkpoint_iteration, output)
        """
        
        mse = torch.mean((output -
                          dataset.extract_targets(data).cuda())**2).data.tolist()
        mean_error = torch.mean(
                        torch.abs(output -
                                  dataset.extract_targets(data).cuda())).data.tolist()

        accumulated_error += mean_error
        accumulated_mse += mse
        error = torch.abs(output - dataset.extract_targets(data).cuda()).data.cpu()
        
        ### covariate shift segment starts
        error = error.unsqueeze(dim=2)
        error_transpose = torch.transpose(error, 1, 2)
        # compute covariate shift
        covariate_shift = torch.matmul(error, error_transpose)
        # expand traj length tensor to Bx3x3 (considering steer, throttle & brake)
        traj_lengths = torch.stack([torch.stack([data['current_traj_length'].squeeze(dim=1)]*3, dim=1)]*3, dim=2)
        covariate_shift = covariate_shift / traj_lengths
        covariate_shift = torch.sum(covariate_shift, dim=0)
        # print ('current covariate shift: ', covariate_shift.shape)

        normalized_covariate_shift += covariate_shift
        ### covariate shift segment ends

        total_episodes = data['episode_count'][-1].data
        iteration_on_checkpoint += 1
        if iteration_on_checkpoint % 50 == 0:
            print ('iteration: ', iteration_on_checkpoint)

    print ('total episodes: ', total_episodes)
    normalized_covariate_shift = normalized_covariate_shift / total_episodes
    print ('normalized covariate shift: ', normalized_covariate_shift.shape, normalized_covariate_shift)
    
    # save the matrix to restart directly from the mat file
    # np.save(os.path.join(g_conf.COVARIANCE_MATRIX_PATH, 'covariance_matrix_%s.npy'%g_conf.DART_COVMATH_DATA), normalized_covariate_shift)
    return normalized_covariate_shift.numpy()

    '''
    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='compute covariance matrix by running a single validation step')
    
    parser.add_argument('--gpus', type=str, help='gpu id required')
    parser.add_argument('--exp_batch', type=str, default='nocrash', help='from which folder to import config file')
    parser.add_argument('--exp_alias', type=str, default='resnet34imnet10S1', help='which resenet model to use')
    # parser.add_argument('--dataset_name', type=str, help='preload name of the required dataset') # not required anymore

    args = parser.parse_args()

    execute(args.gpus, args.exp_batch, args.exp_alias)        