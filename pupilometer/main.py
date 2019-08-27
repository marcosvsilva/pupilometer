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

            pupil = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, dp=self.hough_dp, minDist=self.hough_minDist,
                                     param1=self.hough_param1, param2=self.hough_param2,
                                     minRadius=self.hough_minRadius, maxRadius=self.hough_maxRadius)

            contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if len(contours) > 0:
                (x, y, w, h) = cv2.boundingRect(contours[0])
                cv2.circle(final, (x + int(w / 2), y + int(h / 2)), int(h / 3), (255, 255, 255), 3)
                cv2.line(final, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
                cv2.line(final, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)

            img_final1 = cv2.hconcat([self.resize(gray), self.resize(gaussian), self.resize(median)])
            img_final2 = cv2.hconcat([self.resize(gray), self.resize(threshold), self.resize(final)])
            img_final = cv2.vconcat([img_final1, img_final2])

            cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
            cv2.imshow('Training', img_final)

            if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                time.sleep(3)

        exam.release()
        cv2.destroyAllWindows

    @staticmethod
    def resize(figure):
        return cv2.resize(figure, (0, 0), fx=0.5, fy=0.5)


if __name__ == '__main__':
    main = Main()
    main.start_process()
