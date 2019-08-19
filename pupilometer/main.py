import os
import cv2
import numpy as np


class Main:
    def __init__(self):
        self.dataset_path = os.getcwd() + "/dataset"
        self.exams = os.listdir(self.dataset_path)

    def start_process(self):
        for exam in self.exams:
            video = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
            self.pupillary_analysis(video)

    def pupillary_analysis(self, exam):
        while exam.isOpened():
            ret, frame = exam.read()
            b, g, r = cv2.split(frame)

            kSize = 4
            frame_blur = cv2.blur(g, (kSize, kSize))
            frame_laplacian = cv2.Laplacian(frame_blur, cv2.CV_8UC1)
            test, frame_black = cv2.threshold(frame_laplacian, 0, 255, cv2.THRESH_BINARY)

            pupil = cv2.HoughCircles(frame_laplacian,
                                     cv2.HOUGH_GRADIENT,
                                     40, 10,
                                     param1=50,
                                     param2=60,
                                     minRadius=30,
                                     maxRadius=70)

            if pupil is not None:
                pupil = np.uint16(np.around(pupil))

                for i in pupil[0,:]:
                    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                    cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                    break

                cv2.namedWindow('video', cv2.WINDOW_NORMAL)
                cv2.imshow('video', frame)
                cv2.waitKey(1)

        exam.release()
        cv2.destroyAllWindows


if __name__ == '__main__':
    main = Main()
    main.start_process()
