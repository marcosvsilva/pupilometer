import numpy as np
import cv2

semicircle_positions = ('east', 'west', 'south', 'southeast', 'southwest')


class Ellipse:
    def __init__(self):
        # Ellipse parameters
        self.radius_size = (30, 90)
        self.radius_validate_threshold = 3

        self.lin = self.col = 0

    def best_center(self, image, contours):
        self.lin, self.col = image.shape
        center = None
        rad = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            if center[0] < self.lin and center[1] < self.col:
                radius = []
                for direction in semicircle_positions:
                    radio = self.calculate_radius(image=image, center=center, direction=direction)
                    radius.append(radio)

                if self.validate_radius(radius):
                    rad = int(np.array(radius).max())
                    break

        return center, rad

    def calculate_radius(self, image, center, direction):
        x, y = center
        radius = calc_radius = 0
        init = image[y, x]

        while (1 < x < self.col-1) and (1 < y < self.lin-1):
            if direction == 'east':
                x += 1
            elif direction == 'west':
                x -= 1
            elif direction == 'south':
                y += 1
            elif direction == 'southeast':
                x += 1
                y += 1
            elif direction == 'southwest':
                x -= 1
                y += 1

            calc_radius += 1
            if image[y, x] != init:
                radius = calc_radius
                break

        return radius

    def validate_radius(self, radius):
        validate = 0
        for rad in radius:
            if rad in range(self.radius_size[1])[slice(*self.radius_size)]:
                validate += 1
        return validate >= self.radius_validate_threshold
