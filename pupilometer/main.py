import numpy as np
import cv2
import os
import time


class Main:
    def __init__(self):
        self.dataset_path = os.getcwd() + "/dataset"
        self.exams = os.listdir(self.dataset_path)

        # Parameter Definition
        self.whitening = 0.2

        self.size_filter_gaussian = (9, 9)
        self.type_gaussian = 0

        self.size_median = 3
        self.thresh_threshold = 25
        self.maxvalue_threshold = 255

        self.color_circle = (255, 255, 0)
        self.thickness_circle = 3

        self.best_area_height = (50, 300)
        self.best_area_width = (50, 300)
        self.radius_size = (50, 100)

        self.sleep_pause = 3

        self.threshold = 150

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

            if center is not None:
                cv2.circle(final, center, radius, self.color_circle, self.thickness_circle)

                img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(median)])
                img_final2 = cv2.hconcat([self.resize(gray), self.resize(threshold), self.resize(final)])
                img_final = cv2.vconcat([img_final1, img_final2])

                cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
                cv2.imshow('Training', img_final)

                if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                    time.sleep(self.sleep_pause)

        exam.release()
        cv2.destroyAllWindows

    def best_center(self, image, contours):
        center = None
        radius = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            lin, col = image.shape
            if center[0] < lin and center[1] < col:
                radius = self.calculate_radius(image=image, center=center)
                if radius in range(self.radius_size[1])[slice(*self.radius_size)]:
                    break
        return center, radius

    def calculate_radius(self, image, center):
        lin, col = image.shape
        x, y = center
        radius = 0
        init = int(image[x][y])
        for i in range(col-y):
            if abs(int(image[x][y+i])-init) > self.threshold:
                radius = i
                break
        return radius



    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)


if __name__ == '__main__':
    main = Main()
    main.start_process()
