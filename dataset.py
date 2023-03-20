import csv
import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    class_name = "Y"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            num_cords = len(handLms.landmark)
            landmarks = ['class']
            for i in range(1, num_cords + 1):
                landmarks += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i)]

            data = handLms.landmark
            data_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in data]).flatten())
            data_row.insert(0, class_name)

            with open('fDataset_short.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(data_row)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                              landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if k==27:
        break

cap.release()
cv2.destroyAllWindows()