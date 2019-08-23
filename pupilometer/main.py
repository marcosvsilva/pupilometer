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
            break

    def pupillary_analysis(self, exam):
        while exam.isOpened():
            ret, frame = exam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            final = np.copy(gray)

            whitening = np.array(255 * (final / 255) ** self.whitening, dtype='uint8')

            edge = cv2.Canny(whitening, threshold1=self.canny_threshold1, threshold2=self.canny_threshold2)

            pupil = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=self.hough_dp, minDist=self.hough_minDist,
                                     param1=self.hough_param1, param2=self.hough_param2,
                                     minRadius=self.hough_minRadius, maxRadius=self.hough_maxRadius)

            if pupil is not None:
                pupil = np.uint16(np.around(pupil))

                for i in pupil[0, :]:
                    cv2.circle(final, (i[0], i[1]), i[2], (255, 255, 0), 1)

            img_final1 = cv2.hconcat([self.resize(gray), self.resize(whitening)])
            img_final2 = cv2.hconcat([self.resize(edge), self.resize(final)])
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
