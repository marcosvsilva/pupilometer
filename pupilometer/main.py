import cv2
import os
import time
from filters import Filters


class Main:
    def __init__(self):
        # Main parameters
        self.dataset_path = os.getcwd() + "/dataset"
        self.output_path = os.getcwd() + "/identified"
        self.name_output = "frame"
        self.exams = os.listdir(self.dataset_path)
        self.detail_presentation = True
        self.save_output = False
        self.sleep_pause = 3

        self.filters = Filters(self.detail_presentation)

    def start_process(self):
        for exam in self.exams:
            movie = cv2.VideoCapture("{}/{}".format(self.dataset_path, exam))
            self.pupil_process(movie)

    def pupil_process(self, exam):
        number_frame = 0
        while True:
            ret, frame = exam.read()

            if frame is None:
                break

            presentation, final = self.filters.pupil_analysis(frame)

            cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
            cv2.imshow('Training', presentation)

            if self.save_output:
                name_output = "%s/%s_%03d.png" % (self.output_path, self.name_output, number_frame)
                cv2.imwrite(name_output, final)

            if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
                time.sleep(self.sleep_pause)

        exam.release()
        cv2.destroyAllWindows


if __name__ == '__main__':
    main = Main()
    main.start_process()
