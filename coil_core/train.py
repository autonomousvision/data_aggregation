import os
import sys
import random
import time
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate_auto, adjust_learning_rate_cosine_annealing
from input import CoILDataset, Augmenter, select_balancing_strategy
from logger import coil_logger
from coilutils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint, \
                                    check_loss_validation_stopped

# additional optimizers
from .adabound import AdaBound
from .adamaio import AdamAIO


def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
    """
        The main training function. This functions loads the latest checkpoint
        for a given, exp_batch (folder) and exp_alias (experiment configuration).
        With this checkpoint it starts from the beginning or continue some training.
    Args:
        gpu: gpus ids for training
        exp_batch: the folder with the experiments
        exp_alias: the alias, experiment name
        suppress_output: if the output are going to be saved on a file
        number_of_workers: the number of threads used for data loading

    Returns:
        None

    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu)
        g_conf.VARIABLE_WEIGHT = {}

        # At this point the log file with the correct naming is created.
        # You merge the yaml file with the global configuration structure.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')

        # Set the process into loading status.
        coil_logger.add_message('Loading', {'GPU': gpu})

        # Put the output to a separate file if it is the case
        if suppress_output:
            if not os.path.exists('_output_logs'):
                os.mkdir('_output_logs')
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a",
                              buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)

        if coil_logger.check_finish('train'):
            coil_logger.add_message('Finished', {})
            return

        # Preload option
        if g_conf.PRELOAD_MODEL_ALIAS is not None:
            checkpoint = torch.load(os.path.join('_logs', g_conf.PRELOAD_MODEL_BATCH,
                                                  g_conf.PRELOAD_MODEL_ALIAS,
                                                 'checkpoints',
                                                 str(g_conf.PRELOAD_MODEL_CHECKPOINT)+'.pth'))


        # Get the latest checkpoint to be loaded
        # returns none if there are no checkpoints saved for this model
        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                    'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
            print ('iteration: ', iteration, 'best_loss: ', best_loss)
        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # By instantiating the augmenter we get a callable that augment images and transform them into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        # Instantiate the class used to read the dataset
        dataset = CoILDataset(full_dataset, transform=augmenter, preload_name=str(g_conf.NUMBER_OF_HOURS)+'hours_'+g_conf.TRAIN_DATASET_NAME)
        print ("Loaded dataset")
        
        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.
        # define the sampling strategy for mini-batch, different samplers can be found in 'splitter.py'
        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)

        # Instatiate the network architecture
        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE) # adabound and adamio can also be used here

        if checkpoint_file is not None or g_conf.PRELOAD_MODEL_ALIAS is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            accumulated_time = checkpoint['total_time']
            loss_window = coil_logger.recover_loss_window('train', iteration)
        else:  
            # We accumulate iteration time and keep the average speed
            accumulated_time = 0
            loss_window = []

        # freeze the perception module weights if required
        # for m in model.perception.parameters():
        #     m.requires_grad = False
        
        # total trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print ('trainable parameters: ', total_params)

        # multi-gpu
        print ('number of gpus: ', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        criterion = Loss(g_conf.LOSS_FUNCTION)

        print ('Start Training')

        st = time.time()
        for data in data_loader:

            # use this for early stopping if the validation loss is not coming down
            if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
                    check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
                break

            """
                ####################################
                    Main optimization loop
                ####################################
            """

            iteration += 1

            if iteration % 1000 == 0:
                adjust_learning_rate_auto(optimizer, loss_window)
            
            # additional learning rate scheduler - cyclic cosine annealing (https://arxiv.org/pdf/1704.00109.pdf)
            # adjust_learning_rate_cosine_annealing(optimizer, loss_window, iteration)

            capture_time = time.time()
            controls = data['directions']
            model.zero_grad()
            branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())
            loss_function_params = {
                'branches': branches,
                'targets': dataset.extract_targets(data).cuda(),
                'controls': controls.cuda(),
                'inputs': dataset.extract_inputs(data).cuda(),
                'branch_weights': g_conf.BRANCH_LOSS_WEIGHT,
                'variable_weights': g_conf.VARIABLE_WEIGHT
            }
            loss, _ = criterion(loss_function_params)
            loss.backward()
            optimizer.step()
            """
                ####################################
                    Saving the model if necessary
                ####################################
            """

            if is_ready_to_save(iteration):
                if torch.cuda.device_count() > 1:
                    state_dict_save = model.module.state_dict()
                else:
                    state_dict_save = model.state_dict()

                state = {
                    'iteration': iteration,
                    'state_dict': state_dict_save,
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            """
                ################################################
                    Adding tensorboard logs.
                    Making calculations for logging purposes.
                    These logs are monitored by the printer module.
                #################################################
            """
            coil_logger.add_scalar('Loss', loss.data, iteration)
            coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data) - 1)

            if torch.cuda.device_count() > 1:
                output = model.module.extract_branch(torch.stack(branches[0:4]), controls)
            else:
                output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data).cuda())

            accumulated_time += time.time() - capture_time

            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                     'Output': output[position].data.tolist(),
                                     'GroundTruth': dataset.extract_targets(data)[
                                         position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(data)[
                                         position].data.tolist()},
                                    iteration)
            loss_window.append(loss.data.tolist())
            coil_logger.write_on_error_csv('train', loss.data)
            print("Iteration: %d  Loss: %f" % (iteration, loss.data))
            st = time.time()

        coil_logger.add_message('Finished', {})
    
    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except RuntimeError as e:

        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})


# for testing and debugging
if __name__ == '__main__':
    merge_with_yaml(os.path.join('configs/nocrash/resnet34imnet10S1.yaml'))
    print (g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    for m in model.perception.parameters():
        print (type(m), m.requires_grad)
    print ('checkpoint loaded')
    checkpoint = torch.load('/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/dagger/resnet34imnet10S1/checkpoints/600000.pth',
        map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    for m in model.perception.parameters():
        m.requires_grad = False
        print (type(m), m.requires_grad)