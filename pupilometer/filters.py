import numpy as np
import cv2
from ellipse import Ellipse


class Filters:
    def __init__(self, detail_return):
        self.detail_return = detail_return

        # Filters parameters
        self.whitening = 0.2
        self.size_filter_gaussian = (9, 9)
        self.type_gaussian = 0
        self.size_median = 3
        self.thresh_threshold = 25
        self.maxvalue_threshold = 255
        self.kernel_size_morphology = ((5, 5), (7, 7), (10, 10), (12, 12), (15, 15), (17, 17))
        self.color_circle = (255, 255, 0)
        self.thickness_circle = 3

        self.ellipse = Ellipse()

    def pupil_analysis(self, frame):
        if frame is None:
            raise Exception("Frame is none!")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, self.size_filter_gaussian, self.type_gaussian)
        median = cv2.medianBlur(gaussian, self.size_median)

        final = np.copy(gray)
        for size in self.kernel_size_morphology:
            kernel = np.ones(size, np.uint8)
            erode = cv2.erode(median, kernel=kernel, iterations=1)
            dilate = cv2.dilate(erode, kernel=kernel, iterations=1)
            threshold = cv2.threshold(dilate, self.thresh_threshold, self.maxvalue_threshold, cv2.THRESH_BINARY)[1]

            contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            center, radius = self.ellipse.select_best_center(image=threshold, contours=contours)
            if center is not None and radius > 0:
                cv2.circle(final, center, radius, self.color_circle, self.thickness_circle)
                break

        if self.detail_return:
            img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(erode)])
            img_final2 = cv2.hconcat([self.resize(dilate), self.resize(threshold), self.resize(final)])
            return cv2.vconcat([img_final1, img_final2]), final
        else:
            return cv2.hconcat([self.resize(gray), self.resize(final)]), final

    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)
