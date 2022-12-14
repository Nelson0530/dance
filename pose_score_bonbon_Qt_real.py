import cv2, json
import mediapipe as mp
import math, sys, threading
from PyQt5 import QtWidgets, QtCore, QtMultimedia
from PyQt5.QtGui import QImage, QPixmap, QFont

app = QtWidgets.QApplication(sys.argv)
app_w, app_h = 1920, 1080
path = './mp3/bonbon.mp3'                # 音乐文件路径
url = QtCore.QUrl.fromLocalFile(path)

win = QtWidgets.QWidget()
win.setWindowTitle("Let's_dance")
win.resize(app_w, app_h)
#
view = QtWidgets.QGraphicsView(win)
view.setGeometry(0, 0, app_w, app_h)
scene = QtWidgets.QGraphicsScene()
pict = QPixmap("./UI/ui_5.jpg")
pict = pict.scaled(app_w, app_h)
scene.setSceneRect(0, 0, app_w, app_h-25)
scene.addPixmap(pict)
view.setScene(scene)

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

ocv = True
def closeOpencv():
    global ocv
    ocv = False
win.closeEvent = closeOpencv

label = QtWidgets.QLabel(win)
label.setGeometry(0, 110, 950, 970)

label2 = QtWidgets.QLabel(win)
label2.setGeometry(969, 110, 830, 970)

label3 = QtWidgets.QLabel(win)
label3.setGeometry(1800, 800, 120, 200)

label4 = QtWidgets.QLabel(win)
label4.setGeometry(1600, 10, 320, 80)
# label4.setContentsMargins(0, 0, 0, 0)          # 設定邊界
label4.setAlignment(QtCore.Qt.AlignRight)  # 對齊方式
label4.setStyleSheet('''color:red''')

label5 = QtWidgets.QLabel(win)
label5.setGeometry(150, 10, 700, 80)
label5.setStyleSheet("color:white")

font = QFont()                       # 加入文字設定
font.setFamily('Verdana')                  # 設定字體
font.setPointSize(40)                      # 文字大小

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

def opencv():
    global ocv
    cap = cv2.VideoCapture('./mp4/bonbon_cut.mp4')        #開啟影片檔案
    cap2 = cv2.VideoCapture(0)
    # 啟用姿勢偵測
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    if not cap.isOpened() and cap2.isOpened():
        exit()
    else:
        content = QtMultimedia.QMediaContent(url)  # 载入音樂
        player = QtMultimedia.QMediaPlayer()  # 建立 QMediaPlayer元件
        player.setMedia(content)  # 相連 QMediaPlayer元件與音樂路徑
        player.play()

    c = 1
    i = 0
    count = 60
    score = 0
    while ocv:
        ret, video = cap.read()
        ret2, camera = cap2.read()
        # video = video[0:720, 350:950]
        camera = camera[0:480, 150:550]
        camera = cv2.flip(camera, 1)
        video = cv2.resize(video, (830, 970))
        camera = cv2.resize(camera, (950, 970))
        video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(f"./image/bonbon/{(c // 60) + 1}.jpg")
        img2 = cv2.resize(img2, (120, 200))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if not ret and ret2:
            break
        if c > 240:
            i += 1
            if i % count == 0:
                cv2.imwrite(f"./camera_photo/camera{c}.jpg", camera)
                i = 0
                img = cv2.imread(f"./camera_photo/camera{c}.jpg")
                size = img.shape  # 取得照片影像尺寸
                w = size[1]  # 取得畫面寬度
                h = size[0]  # 取得畫面高度
                results = pose.process(img)  # 取得姿勢偵測結果
                if results.pose_landmarks:
                    x1 = results.pose_landmarks.landmark[11].x * w  # 取得手部末端 x 座標
                    x2 = results.pose_landmarks.landmark[12].x * w
                    x3 = results.pose_landmarks.landmark[13].x * w
                    x4 = results.pose_landmarks.landmark[14].x * w
                    x5 = results.pose_landmarks.landmark[15].x * w
                    x6 = results.pose_landmarks.landmark[16].x * w
                    x7 = results.pose_landmarks.landmark[23].x * w
                    x8 = results.pose_landmarks.landmark[24].x * w
                    x9 = results.pose_landmarks.landmark[25].x * w
                    x10 = results.pose_landmarks.landmark[26].x * w
                    x11 = results.pose_landmarks.landmark[27].x * w
                    x12 = results.pose_landmarks.landmark[28].x * w
                    y1 = results.pose_landmarks.landmark[11].y * h  # 取得手部末端 y 座標
                    y2 = results.pose_landmarks.landmark[12].y * h
                    y3 = results.pose_landmarks.landmark[13].y * h
                    y4 = results.pose_landmarks.landmark[14].y * h
                    y5 = results.pose_landmarks.landmark[15].y * h
                    y6 = results.pose_landmarks.landmark[16].y * h
                    y7 = results.pose_landmarks.landmark[23].y * h
                    y8 = results.pose_landmarks.landmark[24].y * h
                    y9 = results.pose_landmarks.landmark[25].y * h
                    y10 = results.pose_landmarks.landmark[26].y * h
                    y11 = results.pose_landmarks.landmark[27].y * h
                    y12 = results.pose_landmarks.landmark[28].y * h

                    B_C = [x4, y4, x6, y6]
                    B_A = [x4, y4, x2, y2]
                    A_B = [x2, y2, x4, y4]
                    A_G = [x2, y2, x8, y8]
                    G_A = [x8, y8, x2, y2]
                    G_I = [x8, y8, x10, y10]
                    I_G = [x10, y10, x8, y8]
                    I_K = [x10, y10, x12, y12]
                    E_F = [x3, y3, x5, y5]
                    E_D = [x3, y3, x1, y1]
                    D_E = [x1, y1, x3, y3]
                    D_H = [x1, y1, x7, y7]
                    H_D = [x7, y7, x1, y1]
                    H_J = [x7, y7, x9, y9]
                    J_H = [x9, y9, x7, y7]
                    J_L = [x9, y9, x11, y11]

                    ang_1 = angle(B_C, B_A)
                    ang_2 = angle(A_B, A_G)
                    ang_3 = angle(G_A, G_I)
                    ang_4 = angle(I_G, I_K)
                    ang_5 = angle(E_F, E_D)
                    ang_6 = angle(D_E, D_H)
                    ang_7 = angle(H_D, H_J)
                    ang_8 = angle(J_H, J_L)
                    ang_avg = (ang_1 + ang_2 + ang_3 + ang_4 + ang_5 + ang_6 + ang_7 + ang_8) / 8

                    # 根據姿勢偵測結果，標記身體節點和骨架
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    with open(f"./jsonfile/bonbon/sample{c // 60}.json") as f:
                        p = json.load(f)

                    x = int(abs(ang_avg - p["angavg"]))
                    if x < 3:
                        score += 100
                    elif x > 3 and x <= 10:
                        score += 50
                    elif x > 10 and x <= 20:
                        score += 10
                    else:
                        score += 0

        label4.setText(str(score))
        label4.setFont(font)
        label5.setText("Chiki-chiki Ban-ban")
        label5.setFont(font)
        c += 1

        photo = QImage(camera, 950, 970, (950 * 3), QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(photo))

        photo2 = QImage(video, 830, 970, (830 * 3), QImage.Format_RGB888)
        label2.setPixmap(QPixmap.fromImage(photo2))

        photo3 = QImage(img2, 120, 200, (120 * 3),  QImage.Format_RGB888)
        label3.setPixmap(QPixmap.fromImage(photo3))

        cv2.waitKey(24)

    cap.release()
    cap2.release()

if __name__ == '__main__':
    vio = threading.Thread(target=opencv)
    vio.start()
    win.showMaximized()
    sys.exit(app.exec_())
