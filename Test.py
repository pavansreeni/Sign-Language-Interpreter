
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


'''while cap.isOpened():

    # Read a frame.
    ok, frame = cap.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)'''

pTime = 0
cTime = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.xw), int(lm.yh)
                #print(id, cx, cy)
                if id == 0:
                    cv2.circle(frame, (cx,cy), 0, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS,
                                  connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0),thickness=3, circle_radius=2),
                                  landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255),thickness=3, circle_radius=2))


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)

'''def countFingers(image, results, draw=True, display=True):
    height, width = image.shape
    output_image = image.copy()

    total_count = {'RIGHT': 0, 'LEFT': 0}

    fingers_tips_ids = [mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                        mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP]

    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        for tip_index in fingers_tips_ids:
            finger_name = tip_index.name.split("_")[0]

            if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True
                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP - 2].x

        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (
                hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            fingers_statuses[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] += 1

    if draw:
        cv2.putText(output_image, " Total Fingers: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_image, str(sum(count.values())), cv2.FONT_HERSHEY_SIMPLEX, 8.9, (20, 255, 155), 10, 10)

    else:
        return output_image, fingers_statuses, count'''