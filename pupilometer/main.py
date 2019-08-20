import numpy as np
import cv2
import os

class Main:
    def __init__(self):
        self.dataset_path = os.getcwd() + "/dataset"
        self.exams = os.listdir(self.dataset_path)

        # Parameter Definition
        self.max_height = 10
        self.max_width = 10

        self.whitening = 0.4

        self.canny_threshold1 = 10
        self.canny_threshold2 = 50

        self.hough_dp = 40
        self.hough_minDist = 40
        self.hough_param1 = 50
        self.hough_param2 = 60
        self.hough_minRadius = 40
        self.hough_maxRadius = 300

    def start_process(self):
        for exam in self.exams:
            video = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
            self.pupillary_analysis(video)

    def pupillary_analysis(self, exam):
        while exam.isOpened():
            ret, frame = exam.read()

            frame = np.array(frame)
            cv2.resize(frame, (self.max_height, self.max_width)).flatten()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            final = np.copy(gray)

            gray = np.array(255 * (gray / 255) ** self.whitening, dtype='uint8')

            edge = cv2.Canny(gray, threshold1=self.canny_threshold1, threshold2=self.canny_threshold2)

            pupil = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=self.hough_dp, minDist=self.hough_minDist,
                                     param1=self.hough_param1, param2=self.hough_param2,
                                     minRadius=self.hough_minRadius, maxRadius=self.hough_maxRadius)

            if pupil is not None:
                pupil = np.uint16(np.around(pupil))

                for i in pupil[0, :]:
                    cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)

            img_final1 = cv2.hconcat([final, edge])
            cv2.resize(final, img_final1.shape).flatten()
            img_final = cv2.hconcat([img_final1, gray])

            cv2.namedWindow('Progress', cv2.WINDOW_NORMAL)
            cv2.imshow('Progress', img_final)
            cv2.waitKey(1)

        exam.release()
        cv2.destroyAllWindows


if __name__ == '__main__':
    main = Main()
    main.start_process()
