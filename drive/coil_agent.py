import numpy as np
import scipy
import sys
import os
import glob
import torch

from scipy.misc import imresize
from PIL import Image

# corrputions
# from imagecorruptions import corrupt

import matplotlib.pyplot as plt

try:
    from carla08 import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from carla08.agent import CommandFollower
from carla08.client import VehicleControl

from network import CoILModel
from configs import g_conf
from logger import coil_logger

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass




class CoILAgent(object):

    def __init__(self, checkpoint, town_name, carla_version='0.84'):

        # Set the carla version that is going to be used by the interface
        self._carla_version = carla_version 
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()

        # this entire segment is for loading models for ensemble evaluation - take care for the paths and checkpoints
        # load the felipe model for ensemble evaluation
        # self._model_felipe = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        # checkpoint_felipe = torch.load('/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/S2_working_filtered_sgd/resnet34imnet10S1/checkpoints/420000.pth')
        # self._model_felipe.load_state_dict(checkpoint_felipe['state_dict'])
        # self._model_felipe.cuda()
        # self._model_felipe.eval()
        
        # ensemble
        '''
        self.weights = [0.25, 0.25, 0.25, 0.25] # simple ensemble
        # self.weights = [0.333, 0.333, 0.334]
        # self.weights = [0.5, 0.5]
        # self.weights = [0.42, 0.32, 0.17, 0.09] # swagger weights based on smile ensemble scheme
        # self.weights = [0.17, 0.22, 0.27, 0.34]
        # self.weights = [0.26, 0.33, 0.41]
        # self.weights = [0.44, 0.56]
        self.model_ids = ['660000', '670000', '1070000', '2640000']
        # self.model_ids = ['660000', '670000', '1070000']
        # self.model_ids = ['660000', '670000']
        # self.model_ids = ['660000', '720000', '790000', '910000']
        # self.model_ids = ['120000', '180000', '230000', '280000']
        # self.model_ids = ['660000', '760000', '800000', '830000']
        # self.model_ids = ['120000', '240000', '410000', '450000']
        self.models_dir = '/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/ensemble'
        # self.models_dir = '/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/smile_again'
        # self.models_dir = '/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/smile_scratch_again'
        # self.model_ids = ['670000', '1070000', '2640000']
        # self.models_dir = '/is/rg/avg/aprakash/carla_logs/_logs/dagger'
        self._ensemble_model_list = []
        for i in range(len(self.model_ids)):
            curr_checkpoint = torch.load(self.models_dir+'/resnet34imnet10S1/checkpoints/'+self.model_ids[i]+'.pth')
            self._ensemble_model_list.append(CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION))
            self._ensemble_model_list[i].load_state_dict(curr_checkpoint['state_dict'])
            self._ensemble_model_list[i].cuda().eval()
        '''
        self.latest_image = None
        self.latest_image_tensor = None

        # for image corruptions
        self.corruption_number = None
        self.severity = None
        
        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE: # for evaluating expert
            self.control_agent = CommandFollower(town_name)

    def run_step(self, measurements, sensor_data, directions, target, **kwargs):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: All the float measurements from CARLA ( Just speed is used)
            sensor_data: All the sensor data used on this benchmark
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """
        # only required if using corruptions module
        # self.corruption_number = kwargs.get('corruption_number', None)
        # self.severity = kwargs.get('severity', None)

        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                 directions_tensor)
        # run forward pass using felipe model
        # model_outputs_felipe = self._model_felipe.forward_branch(self._process_sensors(sensor_data), norm_speed,
        #                                           directions_tensor)
        
        # model_outputs[0] = torch.FloatTensor([(model_outputs[0][i].item()+model_outputs_felipe[0][i].item())/2.0 for i in range(3)]).cuda()
        steer, throttle, brake = self._process_model_outputs(model_outputs[0])
        # steer_f, throttle_f, brake_f = self._process_model_outputs(model_outputs_felipe[0])

        # ensemble
        '''
        steer_c = []
        throttle_c = []
        brake_c = []
        for i in range(len(self.model_ids)):
            mo = self._ensemble_model_list[i].forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)
            s, t, b = self._process_model_outputs(mo[0])
            steer_c.append(s)
            throttle_c.append(t)
            brake_c.append(b)
        '''
        if self._carla_version == '0.9':
            import carla
            control = carla.VehicleControl()
        else:
            control = VehicleControl()
        # single model
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # ensemble
        # control.steer = float(np.average(steer_c, weights=self.weights))
        # control.throttle = float(np.average(throttle_c, weights=self.weights))
        # control.brake = float(np.average(brake_c, weights=self.weights))

        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            control.steer, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, sensor_data, target)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(control.steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False

        return control

    # define run step for carla 9
    def run_step_carla9(self, observations):
        norm_speed = np.linalg.norm(observations['velocity'])/g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([int(observations['command'])])
        # print ('rgb: ', observations['big_cam'].shape)
        # print ('velocity: ', observations['velocity'])
        # print ('norm velocity: ', np.linalg.norm(observations['velocity']))
        # print ('norm_speed: ', norm_speed.shape, norm_speed.item())
        # print ('directions_tensor: ', directions_tensor.shape, directions_tensor.item())

        model_outputs = self._model.forward_branch(self._process_sensors(observations), norm_speed,
                                                  directions_tensor)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])

        if self._carla_version == '0.9':
            import carla
            control = carla.VehicleControl()
        else:
            control = VehicleControl()
        # single model
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        
        return control

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
            # attentions.append(np.array(Image.fromarray(sensor).resize((200, 88))))
        return attentions

    def _process_sensors(self, sensors):

        iteration = 0
        for name, size in g_conf.SENSORS.items():

            if self._carla_version == '0.9':
                sensor = sensors[name][g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]
            else:
                sensor = sensors[name].data[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]

            sensor = scipy.misc.imresize(sensor, (size[1], size[2])) # depreciated
            # sensor = np.array(Image.fromarray(sensor).resize((size[2], size[1]))) # for running corruptions
            '''
            # corrupt the image here
            # print ('out of corruption: ', self.corruption_number, self.severity)
            if self.corruption_number is not None and self.severity is not None:
                # print ('in corruption: ', self.corruption_number, self.severity)
                sensor = corrupt(sensor, corruption_number=self.corruption_number, 
                                        severity=self.severity+1)
            '''
            self.latest_image = sensor

            sensor = np.swapaxes(sensor, 0, 1)

            sensor = np.transpose(sensor, (2, 1, 0))

            sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1
    
        image_input = image_input.unsqueeze(0)

        self.latest_image_tensor = image_input

        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        steer, throttle, brake = outputs[0].item(), outputs[1].item(), outputs[2].item()
        # print ('steer: ', steer, 'throttle: ', throttle, 'brake: ', brake)
        
        # these heuristics are a part of the original benchmark, evaluation doesn't run properly without these
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0
        
        # print ('steer after heuristic: ', steer, 'throttle after heuristic: ', throttle, 'brake after heuristic: ', brake)
        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, sensor_data, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _ = self.control_agent.run_step(measurements, sensor_data, [], target)

        return control.steer, control.throttle, control.brake