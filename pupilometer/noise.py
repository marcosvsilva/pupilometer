import numpy as np
import cv2
from ellipse import Ellipse


class Noise:
    def __init__(self, min_area):
        self.min_area = min_area
        self.pixel_remove = 255
        self.new_pixel = 0
        self.ellipse = Ellipse()

    def treatment_noise(self, frame, contours):
        if frame is None:
            raise Exception("Frame is none!")

        new_frame = np.copy(frame)

        center, _ = self.ellipse.search_ellipse(frame, contours)

        if frame[center[0], center[1]] == self.pixel_remove:
            new_frame = self.remove_false_center(frame, center)

        return new_frame

    def remove_false_center(self, frame, center):
        new_frame = np.copy(frame)
        i, j = center
        new_frame[i, j] = self.new_pixel

        validate = True
        increment = 0
        while validate:
            increment += 1
            new_frame[i+increment, j] = new_frame[i-increment, j] = self.new_pixel
            new_frame[i, j+increment] = new_frame[i, j-increment] = self.new_pixel

            validate = (new_frame[i+2, j] == self.pixel_remove or new_frame[i-2, j] == self.pixel_remove) or (
                    new_frame[i, j+2] == self.pixel_remove or new_frame[i, j-2] == self.pixel_remove)

        return new_frame
