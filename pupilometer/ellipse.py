import numpy as np
import cv2

circle_directions = ('north', 'north-northeast', 'northeast', 'east-northeast', 'east', 'east-southeast',
                     'southeast', 'south-southeast', 'south', 'south-southwest', 'southwest', 'west-southwest',
                     'west', 'west-northwest', 'northwest', 'north-northwest')


class Ellipse:
    def __init__(self):
        # Ellipse parameters
        self.radius_size = (25, 90)
        self.radius_validate_threshold = 40
        self.min_area_size = 30*30
        self.rotate_semicircle = False
        self.angle_rotate = 30

        self.lin = self.col = 0
        self.image = None

    def select_best_center(self, image, contours):
        self.image = image
        self.lin, self.col = image.shape
        center = rad = 0

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            if center[0] < self.lin and center[1] < self.col:
                radius = []
                for direction in circle_directions:
                    radio = self.calculate_radius(center=center, direction=direction)
                    radius.append(radio)

                if self.validate_radius(radius):
                    rad = int(np.array(radius).max())
                    break

        return center, rad

    def calculate_radius(self, center, direction):
        x, y = center
        radius = 0
        init = self.image[y, x]

        while (1 < x < self.col-2) and (1 < y < self.lin-2):
            x, y = self.calculating_coordinates((x, y), direction)

            if self.image[y, x] != init:
                radius = self.calc_radius(center, (x, y))
                break

        return radius

    def validate_radius(self, radius):
        validate = [1 for x in radius if x in range(self.radius_size[1])[slice(*self.radius_size)]]
        return np.sum(validate) >= int((len(circle_directions) * self.radius_validate_threshold / 100))

    @staticmethod
    def calc_radius(center, position):
        if position[0] > center[0]:
            return abs(position[0] - center[0])
        else:
            return abs(position[1] - center[1])

    @staticmethod
    def calculating_coordinates(coordinates, direction):
        x, y = coordinates
        for i in range(str(direction).count('north')):
            y -= 1
        for i in range(str(direction).count('south')):
            y += 1
        for i in range(str(direction).count('east')):
            x += 1
        for i in range(str(direction).count('west')):
            x -= 1
        return x, y
