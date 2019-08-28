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

        self.canny_threshold1 = 10
        self.canny_threshold2 = 40

        self.hough_dp = 30
        self.hough_minDist = 40
        self.hough_param1 = 100
        self.hough_param2 = 100

        self.hough_minRadius = 10
        self.hough_maxRadius = 100

        self.crop_height = 50
        self.crop_width = 50

        self.best_area_height = (100, 200)
        self.best_area_width = (100, 200)

        self.limiar = 150

    def start_process(self):
        for exam in self.exams:
            video = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
            self.pupillary_analysis(video)

    def pupillary_analysis(self, exam):
        while exam.isOpened():
            ret, frame = exam.read()
            rows, cols, _ = frame.shape

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, (9, 9), 0)
            median = cv2.medianBlur(gaussian, 3)
            threshold = cv2.threshold(median, 25, 255, cv2.THRESH_BINARY_INV)[1]
            final = np.copy(gray)

            contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            center, radius = self.best_center(image=threshold, contours=contours)

            if center is not None:
                cv2.circle(final, center, radius, (255, 255, 0), 3)

                img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(median)])
                img_final2 = cv2.hconcat([self.resize(gray), self.resize(threshold), self.resize(final)])
                img_final = cv2.vconcat([img_final1, img_final2])

                cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
                cv2.imshow('Training', img_final)

                if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                    time.sleep(3)

        exam.release()
        cv2.destroyAllWindows

    def best_center(self, image, contours):
        center = None
        radius = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + int(w / 2), y + int(h / 2))
            if center[0] in range(self.best_area_height[1])[slice(*self.best_area_height)] and \
                    center[1] in range(self.best_area_width[1])[slice(*self.best_area_width)]:
                radius = self.calculate_radius(image=image, center=center)
                if radius > 0:
                    break
        return center, radius

    def calculate_radius(self, image, center):
        lin, col = image.shape
        x, y = center
        radius = 0
        init = int(image[x][y])
        for i in range(col-y):
            if abs(int(image[x][y+i])-init) > self.limiar:
                radius = i
                break
        return radius



    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)


if __name__ == '__main__':
    main = Main()
    main.start_process()
