import cv2
import os
import glob

def camera():
    cap = cv2.VideoCapture(0)
    c = 1
    i = 0
    count = 30
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            exit()
        i += 1
        if i % count == 0:
            cv2.imwrite("./camera_photo/camera" + str(c) + ".jpg", frame)
            i = 0
        print(c)
        c += 1
        cv2.imshow("xxxx", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera()
