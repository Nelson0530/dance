import cv2

cap = cv2.VideoCapture("./mp4/bonbon.mp4")
if not cap.isOpened():
    exit()
c = 1
i = 0
count = 60
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[0:720, 350:950]
    # frame = cv2.resize(frame, (960, 640))
    # frame = cv2.flip(frame, 1)
    if not ret:
        exit()
    i += 1
    if i % count == 0:
        cv2.imwrite("./image/bonbon/" + str(c // 60) + ".jpg", frame)
        i = 0
    print(c)
    c += 1

cap.release()
# cv2.destroyAllWindows()



