import os
import time
import sys
import random

import numpy as np
import torch
import traceback
import dlib
import argparse
import cv2

from torch.utils.data import Dataset
from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint


def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])



def execute(gpu, exp_batch, exp_alias, dataset_name, suppress_output=True, yaml_file=None):
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
    # check again if this segment is required or not
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
    
    dataset_name = dataset_name.split('_')[-1] # since preload file has '<X>hours_' as prefix whereas dataset folder does not
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name) # original code
    augmenter = Augmenter(None)

    print ('full dataset path: ', full_dataset)
    dataset = CoILDataset(full_dataset, transform=augmenter, preload_name=args.dataset_name)

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
        l1_window = coil_logger.recover_loss_window(dataset_name, None)
    """
    
    model.cuda()

    best_mse = 1000
    best_error = 1000
    best_mse_iter = 0
    best_error_iter = 0

    # modified validation code from here to run a single model
    checkpoint = torch.load(args.checkpoint)
    checkpoint_iteration = checkpoint['iteration']
    print("model loaded ", checkpoint_iteration)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    accumulated_mse = 0
    accumulated_error = 0
    iteration_on_checkpoint = 0

    print ('data_loader size: ', len(data_loader))
    total_error = []
    for data in data_loader:

        # Compute the forward pass on a batch from the loaded dataset
        controls = data['directions']
        branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())
        output = model.extract_branch(torch.stack(branches[0:4]), controls)
        error = torch.abs(output - dataset.extract_targets(data).cuda())
        total_error += error.detach().cpu().tolist()
        
        iteration_on_checkpoint += 1
        if iteration_on_checkpoint % 50 == 0:
            print ('iteration: ', iteration_on_checkpoint)

    total_error = np.array(total_error)
    print (len(total_error), total_error.shape)

    np.save(os.path.join(args.save_path, args.dataset_name, 'computed_error.npy'), total_error)
    '''
    print (len(total_output), len(path_names))
    for act, name in zip(total_output, path_names):
        episode_num = name.split('/')[-2]
        frame_num = name.split('/')[-1].split('_')[-1].split('.')[0]
        if not os.path.isdir(os.path.join(args.save_path, args.dataset_name, episode_num)):
            os.mkdir(os.path.join(args.save_path, args.dataset_name, episode_num))
        file_name = 'Activation_'+frame_num

        torch.save(act, os.path.join(args.save_path, args.dataset_name, episode_num, file_name))
    ''' 
    # total_output = torch.cat(total_output, dim=0)
    # print (total_output.shape)
    # torch.save(total_output, os.path.join(args.save_path, args.dataset_name+'activations'))

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='compute error for 'Rank' based sampling - these values are then sorted in descending order and top 'k' states are sampled')
    
    parser.add_argument('--gpus', type=str, help='gpu id required')
    parser.add_argument('--exp_batch', type=str, default='nocrash', help='from which folder to import config file')
    parser.add_argument('--exp_alias', type=str, default='resnet34imnet10S1', help='which resenet model to use')
    parser.add_argument('--dataset_name', type=str, help='preload name of the required dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint to validate')
    parser.add_argument('--save_path', type=str, required=True, help='path to save computed variances')

    args = parser.parse_args()
    
    if not os.path.isdir(os.path.join(args.save_path, args.dataset_name)):
        os.mkdir(os.path.join(args.save_path, args.dataset_name))

    execute(args.gpus, args.exp_batch, args.exp_alias, args.dataset_name)