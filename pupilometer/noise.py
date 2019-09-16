import numpy as np
import cv2


class Noise:
    def __init__(self, min_area):
        self.min_area = min_area

    def remove_noise(self, frame, contours):
        if frame is None:
            raise Exception("Frame is none!")

        line, col = frame.shape
        new_frame = np.copy(frame)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h < self.min_area:
                while x <= w and x < line:
                    while y <= h and y < col:
                        if new_frame[y, x] == 255:
                            new_frame = 0
                        else:
                            new_frame = 255
                        y += 1
                    x += 1

        return new_frame
