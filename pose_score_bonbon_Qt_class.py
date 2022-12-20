import cv2, json
import mediapipe as mp
import math, sys, threading, mysql.connector, os
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtMultimedia
from PyQt5.QtGui import QImage, QPixmap, QFont

class Windows(QtWidgets.QWidget):   # 繼承父類別 方法、屬性 (由super().__init()__負責) ， 並可以額外自定義屬於自己的屬性、方法 → 傳承!
    def __init__(self):
        super().__init__()         # 繼承父類別屬性
        self.setWindowTitle("lets_dance")
        self.app_w, self.app_h = 1920, 1080
        self.resize(self.app_w, self.app_h)
        self.path = './mp3/bonbon.mp3'           # 音乐文件路径
        self.url = QtCore.QUrl.fromLocalFile(self.path)
        self.ocv = True
        self.mp_drawing = mp.solutions.drawing_utils     # mediapipe 繪圖方法
        self.mp_drawing_styles = mp.solutions.drawing_styles     # mediapipe 繪圖樣式
        self.mp_pose = mp.solutions.pose     # mediapipe 姿勢偵測
        self.ui()
        self.run()
        self.closeEvent = self.closeOpencv        # 試窗關閉事件觸發後，關閉opencv
        QtCore.QTimer.singleShot(30000, self.close)
        # self.savethescore()
        # self.closeEvent = self.savethescore

    def ui(self):
        # 定義背景UI、建立label函式
        self.view = QtWidgets.QGraphicsView(self)
        self.view.setGeometry(0, 0, self.app_w, self.app_h)
        scene = QtWidgets.QGraphicsScene()
        image = QPixmap("./UI/ui_5.jpg")
        image = image.scaled(self.app_w, self.app_h)
        scene.setSceneRect(0, 0, 1920, 1055)
        scene.addPixmap(image)
        self.view.setScene(scene)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(0, 110, 950, 970)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.setGeometry(969, 110, 830, 970)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setGeometry(1800, 800, 120, 200)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setGeometry(1600, 10, 320, 80)
        # self.label4.setContentsMargins(0, 0, 0, 0)       # 設定邊界
        self.label4.setAlignment(QtCore.Qt.AlignRight)  # 對齊方式(向右)
        self.label4.setStyleSheet('''color:red''')
        self.label4.setFont(QFont("Verdana", 40))

        self.label5 = QtWidgets.QLabel(self)
        self.label5.setGeometry(150, 10, 700, 80)
        self.label5.setStyleSheet("color:white")
        self.label5.setFont(QFont("Verdana", 40))
        self.label5.setText("Chiki-chiki Ban-ban")

    def angle(self, v1, v2):       # 計算角度函式
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

    def closeOpencv(self, event):
        self.ocv = False

    def opencv(self):
        cap = cv2.VideoCapture('./mp4/bonbon_cut.mp4')    # 開啟影片檔案
        cap2 = cv2.VideoCapture(0)
        # 啟用姿勢偵測
        pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        if not cap.isOpened() and cap2.isOpened():
            exit()
        else:
            content = QtMultimedia.QMediaContent(self.url)    # 载入音樂
            player = QtMultimedia.QMediaPlayer()      # 建立 QMediaPlayer元件
            player.setMedia(content)      # 相連 QMediaPlayer元件與音樂路徑
            player.play()

        pointer = 1            # 設定一個指針來計算偵數
        i = 0
        count = 60
        score = 0
        while self.ocv:
            ret, video = cap.read()
            ret2, camera = cap2.read()
            # video = video[0:720, 350:950]
            camera = camera[0:480, 150:550]        # 裁切影像畫面
            camera = cv2.flip(camera, 1)
            video = cv2.resize(video, (830, 970))
            camera = cv2.resize(camera, (950, 970))
            video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
            camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            # 讀取 提示照片 = 關鍵幀照片
            img2 = cv2.imread(f"./image/bonbon/{(pointer // 60) + 1}.jpg")
            img2 = cv2.resize(img2, (120, 200))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if not ret and ret2:
                break
            if pointer > 240:            # 當真數來到240時，開始計算
                i += 1
                if i % count == 0:
                    cv2.imwrite(f"./camera_photo/camera{pointer}.jpg", camera)
                    i = 0
                    img = cv2.imread(f"./camera_photo/camera{pointer}.jpg")
                    size = img.shape  # 取得照片影像尺寸
                    w = size[1]  # 取得畫面寬度
                    h = size[0]  # 取得畫面高度
                    results = pose.process(img)  # 取得姿勢偵測結果
                    if results.pose_landmarks:
                        x1 = results.pose_landmarks.landmark[11].x * w      # 取得特定點位之 x 座標
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
                        y1 = results.pose_landmarks.landmark[11].y * h      # 取得特定點位之 y 座標
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

                        B_C = [x4, y4, x6, y6]          # 指定做座標向量變數
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

                        ang_1 = self.angle(B_C, B_A)
                        ang_2 = self.angle(A_B, A_G)
                        ang_3 = self.angle(G_A, G_I)
                        ang_4 = self.angle(I_G, I_K)
                        ang_5 = self.angle(E_F, E_D)
                        ang_6 = self.angle(D_E, D_H)
                        ang_7 = self.angle(H_D, H_J)
                        ang_8 = self.angle(J_H, J_L)
                        ang_avg = (ang_1 + ang_2 + ang_3 + ang_4 + ang_5 + ang_6 + ang_7 + ang_8) / 8

                        # 根據姿勢偵測結果，標記身體節點和骨架
                        self.mp_drawing.draw_landmarks(
                            img,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                        with open(f"./jsonfile/bonbon/sample{pointer // 60}.json") as f:
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

            self.label4.setText(str(score))
            pointer += 1

            photo = QImage(camera, 950, 970, (950 * 3), QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(photo))

            photo2 = QImage(video, 830, 970, (830 * 3), QImage.Format_RGB888)
            self.label2.setPixmap(QPixmap.fromImage(photo2))

            photo3 = QImage(img2, 120, 200, (120 * 3), QImage.Format_RGB888)
            self.label3.setPixmap(QPixmap.fromImage(photo3))

            cv2.waitKey(25)
        # connection = mysql.connector.connect(host='localhost',
        #                                      database='lets_dance',
        #                                      user='root',
        #                                      password='ab0930769971')
        # my_cursor = connection.cursor(prepared=True)
        # try:
        #     insert_face = "INSERT INTO history(P_NO, SON_NO, POINT, TIME) VALUES (%s, %s, %s, %s)"
        #     data_list = (2, 2, str(score), str(datetime.today()))
        #     my_cursor.executemany(insert_face, (data_list,))
        #     connection.commit()
        # except Exception as err:
        #     print(err)
        #     connection.rollback()

        cap.release()
        cap2.release()
        cv2.destroyAllWindows()

    # def savethescore(self):
    #     connection = mysql.connector.connect(host='localhost',
    #                                          database='lets_dance',
    #                                          user='root',
    #                                          password='ab0930769971')
    #     my_cursor = connection.cursor(prepared=True)
    #     try:
    #         insert_face = "INSERT INTO history(P_NO, SON_NO, POINT, TIME) VALUES (%s, %s, %s, %s)"
    #         data_list = (2, 2, str(self.score), str(datetime.today()))
    #         my_cursor.executemany(insert_face, (data_list,))
    #         connection.commit()
    #     except Exception as err:
    #         print(err)
    #         connection.rollback()
    def run(self):
        self.thread_a = threading.Thread(target=self.opencv)
        self.thread_a.start()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Windows()
    win.showMaximized()
    sys.exit(app.exec_())

