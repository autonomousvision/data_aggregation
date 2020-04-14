"""
Code to estimate entropy by computing variance in the predicted control
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
import cv2

from torch.utils.data import Dataset
from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import Augmenter
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


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None):
        # Setting the root directory for this dataset
        self.root_dir = root_dir
        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        print("preload Name ", self.preload_name)

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY ")
            self.sensor_data_names, self.measurements, self.semantic_segmentation_maps = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'))
            print(self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements, self.semantic_segmentation_maps = self._pre_load_image_folders(root_dir)

        print("preload Name ", self.preload_name)

        self.transform = transform
        self.batch_read_number = 0


    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, index):

        try:
            img_path = os.path.join(self.root_dir,
                                    self.sensor_data_names[index].split('/')[-2],
                                    self.sensor_data_names[index].split('/')[-1])
            
            frame_num = self.sensor_data_names[index].split('/')[-1].split('_')[-1].split('.')[0]
            
            # load the prefinal layer activations which were computed
            act_path = os.path.join('/is/rg/avg/aprakash/saved_activations', args.dataset_name, 
                                    self.sensor_data_names[index].split('/')[-2],
                                    'Activation_'+frame_num)
            activation = torch.Tensor(torch.load(act_path))

            measurements = self.measurements[index].copy()
            measurements['activation'] = activation # directly provide the saved activation

            # directions = self.measurements[index]['directions']
            # load the segmentation map
            # seg_map_path = os.path.join(self.root_dir,
            #                             self.semantic_segmentation_maps[index].split('/')[-2],
            #                             self.semantic_segmentation_maps[index].split('/')[-1])

            # load texture here
            # texture_image_path = self.texture_names[np.random.randint(0, len(self.texture_names))]
            # texture = cv2.imread(texture_image_path, cv2.IMREAD_COLOR)
            '''
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # seg = cv2.imread(seg_map_path)
            # Apply the image transformation
            if self.transform is not None:
                boost = 1
                img = self.transform(self.batch_read_number * boost, img)
            else:
                img = img.transpose(2, 0, 1)

            img = img.astype(np.float)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            img = img / 255.

            measurements = self.measurements[index].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()

            measurements['rgb'] = img
            measurements['activation'] = activation

            self.batch_read_number += 1
            '''
        except AttributeError:
            print ("Blank IMAGE")
            '''
            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros(3, 88, 200)
            '''
        return measurements

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INPUTS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)



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



def extract_branch(output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]


# The main function maybe we could call it with a default name
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
    # commenting out this segment to simplify code
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
    
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name) # original code
    augmenter = Augmenter(None)
    # Definition of the dataset to be used. Preload name is just the validation data name
    print ('full dataset path: ', full_dataset)
    dataset = CoILDataset(full_dataset, transform=augmenter,
                          preload_name=dataset_name)

    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.

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
    # checkpoint = torch.load(os.path.join(g_conf.VALIDATION_CHECKPOINT_PATH
    #                        , 'checkpoints', g_conf.VALIDATION_CHECKPOINT_ITERATION + '.pth'))
    checkpoint = torch.load(args.checkpoint)
    checkpoint_iteration = checkpoint['iteration']
    print("model loaded ", checkpoint_iteration)

    model.load_state_dict(checkpoint['state_dict'])

    model.train()
    accumulated_mse = 0
    accumulated_error = 0
    iteration_on_checkpoint = 0

    # considering steer, throttle & brake so 3x3 matrix
    normalized_covariate_shift = torch.zeros(3,3)

    print ('data_loader size: ', len(data_loader))
    total_var = []
    for data in data_loader:
        # dataloader directly loads the saved activations
        # Compute the forward pass on a batch from  the validation dataset
        controls = data['directions']
        curr_var = []
        for i in range(100):
            output = model.branches(data['activation'].cuda())
            output_vec = model.extract_branch(torch.stack(output), controls)
            curr_var.append(output_vec.detach().cpu().numpy())
            
        curr_var = np.array(curr_var)
        compute_var = np.var(curr_var, axis=0)
        total_var += compute_var.tolist()
        
        iteration_on_checkpoint += 1
        if iteration_on_checkpoint % 50 == 0:
            print ('iteration: ', iteration_on_checkpoint)

    total_var = np.array(total_var)
    print (len(total_var), total_var.shape)

    # save the computed variance array, this would be used for uncertainity based sampling in 'tools/filter_dagger_data_var.py'
    np.save(os.path.join(args.save_path, args.dataset_name, 'computed_var.npy'), total_var)
    

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='compute variance for uncertainity based sampling')
    
    parser.add_argument('--gpus', type=str, help='gpu id required')
    parser.add_argument('--exp_batch', type=str, default='nocrash', help='from which folder to import config file')
    parser.add_argument('--exp_alias', type=str, default='resnet34imnet10S1', help='which resenet model to use')
    parser.add_argument('--dataset_name', type=str, help='preload name of the required dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint to validate')
    parser.add_argument('--save_path', type=str, required=True, help='path to save computed variances')

    args = parser.parse_args()
    

    execute(args.gpus, args.exp_batch, args.exp_alias, args.dataset_name)