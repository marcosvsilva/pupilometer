import numpy as np
import cv2

semicircle_positions = ('east', 'west', 'south', 'southeast', 'southwest')


class Ellipse:
    def __init__(self):
        # Ellipse parameters
        self.radius_size = (30, 90)
        self.radius_validate_threshold = 3
        self.rotate_semicircle = False
        self.angle_rotate = 30

        self.lin = self.col = 0

    def select_best_center(self, image, contours):
        self.lin, self.col = image.shape
        center = rad = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            if center[0] < self.lin and center[1] < self.col:
                radius = []
                angles = [x * self.angle_rotate for x in range(int(360 / self.angle_rotate)+1)]
                for angle in angles:
                    for direction in semicircle_positions:
                        radio = self.calculate_radius(image=image, center=center, direction=direction, angle=angle)
                        radius.append(radio)

                if self.validate_radius(radius):
                    rad = int(np.array(radius).max())
                    break

        return center, rad

    def calculate_radius(self, image, center, direction, angle):
        x, y = center
        radius = 0
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

            if image[y, x] != init:
                radius = self.calc_radius(center, (x, y))
                break

        return radius

    def validate_radius(self, radius):
        validate = 0
        for rad in radius:
            if rad in range(self.radius_size[1])[slice(*self.radius_size)]:
                validate += 1
        return validate >= self.radius_validate_threshold

    @staticmethod
    def calc_radius(center, position):
        if position[0] > center[0]:
            return position[0] - center[0]
        else:
            return position[1] - center[1]
