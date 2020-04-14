"""
Configuration file used to collect the CARLA data.
"""

from carla import sensor
from carla.settings import CarlaSettings

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
POSITIONS = [ [29, 105], [130, 27], [87, 102], [27, 132], [44, 24],
              [26, 96], [67, 34], [1, 28], [134, 140], [9, 105],
              [129, 148], [16, 65], [16, 21], [97, 147], [51, 42],
              [41, 30], [107, 16], [47, 69], [95, 102], [145, 16],
              [64, 111], [47, 79], [69, 84], [31, 73], [81, 37],
              [57, 35], [116, 42], [47, 75], [143, 132], [8, 145],
              [107, 43], [111, 61], [105, 137], [72, 24], [77, 0],
              [80, 17], [32, 12], [64, 3], [32, 146], [40, 33],
              [127, 71], [116, 21], [49, 51], [110, 35], [85, 91],
              [114, 93], [30, 7], [110, 133], [60, 43], [11, 98], [96, 49], [90, 85],
              [27, 40], [37, 74], [97, 41], [110, 62], [19, 2], [138, 114], [131, 76],
              [116, 95], [50, 71], [15, 97], [74, 71], [50, 133],
              [23, 116], [38, 116], [101, 52], [5, 108], [23, 79], [13, 68]
             ]
# town2 positions
# POSITIONS = [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
#              [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
#              [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
#              [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
#              [70, 73], [46, 67], [57, 50], [61, 49], [21, 12]
#              ]

FOV = 100
sensors_frequency = {'CentralRGB': 1, 'CentralDepth': 1, 'CentralSemanticSeg':  1, 'Lidar32': 1,
                     'RightRGB': 1, 'RightDepth': 1, 'RightSemanticSeg': 1,
                     'LeftRGB': 1, 'LeftDepth': 1, 'LeftSemanticSeg': 1}
sensors_yaw = {'CentralRGB': 0, 'CentralDepth': 0,'CentralSemanticSeg': 0, 'Lidar32': 0,
                 'RightRGB': 30.0, 'RightDepth': 30.0, 'RightSemanticSeg': 30.0,
                 'LeftRGB': -30.0, 'LeftDepth': -30.0, 'LeftSemanticSeg': -30.0}

# these noise percentages are used during expert data collection
# these should be set to 0 for DAgger or DART data collection
lat_noise_percent = 20 # latitudinal noise injection (steer)
long_noise_percent = 20 # longitudinal noise injection (throttle/brake)

NumberOfVehicles = [30, 60]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [50, 100]

set_of_weathers = [1, 3, 6, 8] # train weathers

# model checkpoint for DAgger or DART data collection
model_checkpoint = '/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/_logs/eval_S1/resnet34imnet10S1/660000.pth'
yaml_config_file = '/is/sg2/aprakash/Projects/carla_autonomous_driving/code/coiltraine/configs/demo/resnet34imnet10S1.yaml'


def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""

    # The parameters 'NumberOfVehicles', 'NumberOfPedestrians' and 'WeatherID' are set in 'new_episode' function in 'collect.py'
    settings = CarlaSettings()
    settings.set(
        SendNonPlayerAgentsInfo=True,
        SynchronousMode=True,
        NumberOfVehicles=30,
        NumberOfPedestrians=50,
        WeatherId=1)

    settings.set(DisableTwoWheeledVehicles=True)

    settings.randomize_seeds() # IMPORTANT TO RANDOMIZE THE SEEDS EVERY TIME
    
    camera0 = sensor.Camera('CentralRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)
    settings.add_sensor(camera0)
    
    camera0 = sensor.Camera('LeftRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, -30.0, 0)
    settings.add_sensor(camera0)
    
    camera0 = sensor.Camera('RightRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 30.0, 0)
    settings.add_sensor(camera0)

    '''
    # uncommment this section to generate Semantic Segmentation
    camera0 = sensor.Camera('CentralSemanticSeg', PostProcessing='SemanticSegmentation')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)
    settings.add_sensor(camera0)

    camera0 = sensor.Camera('LeftSemanticSeg', PostProcessing='SemanticSegmentation')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, -30.0, 0)
    settings.add_sensor(camera0)

    camera0 = sensor.Camera('RightSemanticSeg', PostProcessing='SemanticSegmentation')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 30.0, 0)
    settings.add_sensor(camera0)
    '''

    '''
    # uncomment this section to generate Depth
    camera0 = sensor.Camera('CentralDepth', PostProcessing='Depth')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)
    settings.add_sensor(camera0)

    camera0 = sensor.Camera('LeftDepth', PostProcessing='Depth')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, -30.0, 0)
    settings.add_sensor(camera0)
    
    camera0 = sensor.Camera('RightDepth', PostProcessing='Depth')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 30.0, 0)
    settings.add_sensor(camera0)
    '''

    '''
    # uncomment this section to generate LIDAR
    lidar = sensor.Lidar('Lidar32')
    lidar.set_position(0, 0, 2.5)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=32,
        Range=50,
        PointsPerSecond=100000,
        RotationFrequency=10,
        UpperFovLimit=10,
        LowerFovLimit=-30)
    settings.add_sensor(lidar)
    '''

    return settings
