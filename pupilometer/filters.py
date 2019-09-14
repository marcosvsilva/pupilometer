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
        self.kernel_morphology_one = np.ones((5, 5), np.uint8)
        self.kernel_morphology_two = np.ones((7, 7), np.uint8)
        self.color_circle = (255, 255, 0)
        self.thickness_circle = 3

        self.ellipse = Ellipse()

    def pupil_analysis(self, frame):
        if frame is None:
            raise Exception("Frame is none!")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, self.size_filter_gaussian, self.type_gaussian)
        median = cv2.medianBlur(gaussian, self.size_median)

        erode_one = cv2.erode(median, kernel=self.kernel_morphology_one, iterations=1)
        dilate_one = cv2.dilate(erode_one, kernel=self.kernel_morphology_one, iterations=1)
        threshold_one = cv2.threshold(dilate_one, self.thresh_threshold,
                                      self.maxvalue_threshold, cv2.THRESH_BINARY_INV)[1]

        erode_two = cv2.erode(threshold_one, kernel=self.kernel_morphology_two, iterations=1)
        dilate_two = cv2.dilate(erode_two, kernel=self.kernel_morphology_two, iterations=1)
        threshold_two = cv2.threshold(dilate_two, self.thresh_threshold,
                                      self.maxvalue_threshold, cv2.THRESH_BINARY_INV)[1]

        final = np.copy(gray)
        contours = cv2.findContours(dilate_two, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        center, radius = self.ellipse.select_best_center(image=threshold_two, contours=contours)

        if center is not None and radius > 0:
            cv2.circle(final, center, radius, self.color_circle, self.thickness_circle)

        if self.detail_return:
            img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(median)])
            img_final2 = cv2.hconcat([self.resize(threshold_one), self.resize(threshold_two), self.resize(final)])
            return cv2.vconcat([img_final1, img_final2]), final
        else:
            return cv2.hconcat([self.resize(gray), self.resize(final)]), final

    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)
