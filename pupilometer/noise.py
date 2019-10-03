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

        if frame[center[1], center[0]] == self.pixel_remove:
            new_frame = self.remove_false_center(frame, center)

        return new_frame

    def remove_false_center(self, frame, center):
        new_frame = np.copy(frame)
        i, j = center
        lin, col = frame.shape
        new_frame[i, j] = self.new_pixel

        validate = True
        increment = 0
        while validate:
            increment += 1

            if i + increment >= lin or j + increment >= col:
                break

            new_frame[j, i+increment] = new_frame[j, i-increment] = self.new_pixel
            new_frame[j+increment, i] = new_frame[j-increment, i] = self.new_pixel

            validate = (new_frame[i+2, j] == self.pixel_remove or new_frame[i-2, j] == self.pixel_remove) or (
                    new_frame[i, j+2] == self.pixel_remove or new_frame[i, j-2] == self.pixel_remove)

        return new_frame
