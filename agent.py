from keras.models import (load_model, model_from_json)
from carla.agent.agent import Agent
from carla.client import VehicleControl

import pathlib
import cv2
import numpy as np


class DronetAgent(Agent):
    
    def __init__(self, model_directory=None, 
                        architecture_filename="dronet_architecture.json", 
                        weights_filename="dronet_weights.h5",
                        target_size=(200, 200)):
        """
        Initialise the class.
        """

        super().__init__()
        architecture_path = pathlib.Path().joinpath(model_directory, architecture_filename)
        weights_path = pathlib.Path().joinpath(model_directory, weights_filename)

        if not architecture_path.exists() or not weights_path.exists():
            raise FileNotFoundError("Either {} or {} does not exist!"
                                        .format(architecture_path, weights_path))

        self.model = None
        with open(architecture_path, 'r') as fp:
            config = fp.read()
            self.model = model_from_json(config)

        self.model.load_weights(weights_path)
        self.model.compile(loss="mse", optimizer="sgd")
        self.target_size = reversed(target_size)

        self.last_theta = 0
        self.alpha, self.beta = 0.7, 0.5
        self.velocity_max = 30

    def _process_input(sensor_data):
        """
        Preprocess camera image data for use with Dronet.
        """

        assert "CameraRGB" in sensor_data, "No 'CameraRGB' found in sensor stream!"

        camera_img = sensor_data['CameraRGB'].data[:,:,:-1][:, :, ::-1]
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BAYER_BG2GRAY)
        camera_img = cv2.resize(camera_img, self.target_size)
        camera_img = (camera_img[:, :, np.newaxis]).astype(np.float32)

        return camera_img

    def run_step(self, measurements, sensor_data, directions, target):
        """
        Function to run a control step in the CARLA vehicle.
        """
        input_img = self._process_input(sensor_data)
        steer, coll = self.model.predict(input_img[np.newaxis, :, :, :])[0]

        # theta = (1 - beta)*last_theta + beta*pi/2*steer 
        theta = (1 - self.beta)*self.last_theta + beta*np.pi/2*steer

        # throttle = (1 - alpha)*(current_velocity/max_velocity) + alpha*(1 - coll) if current_velocity < max_velocity else 0
        velocity = measurements.forward_speed
        throttle = (1 - self.alpha)*(velocity/self.velocity_max) + self.alpha(1 - coll) if velocity < self.velocity_max else 0.

        self.last_theta = theta

        control = VehicleControl()
        control.throttle = throttle
        control.steer = theta
        return control


