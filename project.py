import cv2
import numpy as np

exam = cv2.VideoCapture('Exams/pupilometer.mp4')

while (1):
    ret, frame  = exam.read()
    b, g, r = cv2.split(frame)

    kSize = 5
    frame_blur = cv2.blur(g, (kSize, kSize))
    frame_laplacian = cv2.Laplacian(frame_blur, cv2.CV_8UC1)
    test, frame_black = cv2.threshold(frame_laplacian, 0, 255, cv2.THRESH_BINARY)
    frame_black = frame_laplacian - frame_black
    #cv2.imshow('test', frame_black)

    pupil = cv2.HoughCircles(frame_black, cv2.HOUGH_GRADIENT, 40, 10, param1=50, param2=60, minRadius=30, maxRadius=70)
    pupil = np.uint16(np.around(pupil))

    for i in pupil[0,:]:
        cv2.circle(frame_black,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(frame_black,(i[0],i[1]),2,(0,0,255),3)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.imshow('video', frame_black)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

exam.release()
cv2.destroyAllWindows