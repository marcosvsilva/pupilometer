import numpy as np
import cv2
import os
import time

directions_radius = ('east', 'west', 'north', 'south', 'northeast', 'northwest', 'southeast', 'south-west')


def inc(numbers):
    return [x + 1 for x in numbers]


def dec(numbers):
    return [x - 1 for x in numbers]


class Main:
    def __init__(self):
        # Paths parameters
        self.dataset_path = os.getcwd() + "/dataset"
        self.exams = os.listdir(self.dataset_path)

        # Filters parameters
        self.whitening = 0.2
        self.size_filter_gaussian = (9, 9)
        self.type_gaussian = 0
        self.size_median = 3
        self.thresh_threshold = 25
        self.maxvalue_threshold = 255

        # Selection best ellipse parameters
        self.best_area_height = (50, 300)
        self.best_area_width = (50, 300)
        self.radius_size = (35, 65)
        self.edge_threshold = 150
        self.radius_validate_threshold = 5

        # Circle draw parameters
        self.color_circle = (255, 255, 0)
        self.thickness_circle = 3

        # Others parameters
        self.sleep_pause = 3

    def start_process(self):
        for exam in self.exams:
            video = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
            self.pupillary_analysis(video)

    def pupillary_analysis(self, exam):
        while exam.isOpened():
            ret, frame = exam.read()
            rows, cols, _ = frame.shape

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, self.size_filter_gaussian, self.type_gaussian)
            median = cv2.medianBlur(gaussian, self.size_median)
            threshold = cv2.threshold(median, self.thresh_threshold, self.maxvalue_threshold, cv2.THRESH_BINARY_INV)[1]
            final = np.copy(gray)

            contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            center, radius = self.best_center(image=threshold, contours=contours)

            if center is not None and radius > 0:
                cv2.circle(final, center, radius, self.color_circle, self.thickness_circle)

                img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(median)])
                img_final2 = cv2.hconcat([self.resize(gray), self.resize(threshold), self.resize(final)])
                img_final = cv2.vconcat([img_final1, img_final2])

                cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
                cv2.imshow('Training', img_final)
                cv2.sav

                if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                    time.sleep(self.sleep_pause)

        exam.release()
        cv2.destroyAllWindows

    def best_center(self, image, contours):
        center = None
        rad = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            lin, col = image.shape
            if center[0] < lin and center[1] < col:
                radius = []
                for direction in directions_radius:
                    radius.append(self.calculate_radius(image=image, center=center, direction=direction))

                if self.validate_radius(radius):
                    rad = int(np.array(radius).mean())
                    break
        return center, rad

    def validate_radius(self, radius):
        validate = 0
        for rad in radius:
            if rad in range(self.radius_size[1])[slice(*self.radius_size)]:
                validate += 1
        return validate >= self.radius_validate_threshold

    def calculate_radius(self, image, center, direction):
        lin, col = image.shape
        x, y = center
        radius = calc_radius = 0
        init = int(image[x][y])
        while (1 < x < lin-1) and (1 < y < col-1):
            if direction == 'east':
                y += 1
            elif direction == 'west':
                y -= 1
            elif direction == 'north':
                x += 1
            elif direction == 'south':
                x -= 1
            elif direction == 'northeast':
                x += 1
                y += 1
            elif direction == 'northwest':
                x += 1
                y -= 1
            elif direction == 'southeast':
                x -= 1
                y += 1
            elif direction == 'south-west':
                x -= 1
                y -= 1
            else:
                break

            calc_radius += 1
            if abs(int(image[x][y]) - init) > self.edge_threshold:
                radius = calc_radius
                break

        return radius

    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)


if __name__ == '__main__':
    main = Main()
    main.start_process()
