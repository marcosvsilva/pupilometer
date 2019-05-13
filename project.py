import cv2
import numpy as np
import scipy         

exam = cv2.VideoCapture('Exams/pupilometer.mp4')

while (1):
    ret, frame  = exam.read()
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)

    frame = cv2.medianBlur(frame, 5)

    #frame = cv2.cvtColor(frame[2], cv2.COLOR_GRAY2BGR)
    
    #circles = cv2.HoughCircles(frame,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)


    # pupil = np.uint16(np.around(pupil))

    # for i in pupil:
    #     cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
    #     cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)*/*

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

exam.release()
cv2.destroyAllWindows