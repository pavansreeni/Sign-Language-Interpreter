import cv2
import mediapipe as mp
import numpy as np

def countFingers(img, results, draw = True, display = True):
    height, width, _ = img.shape

    output_img = img.copy()

    count = {'RIGHT': 0, 'LEFT': 0}

    finger_tips_ids = [mpHands.HandLandmark.INDEX_FINGER_TIP,mpHands.HandLandmark.RING_FINGER_TIP,
                       mpHands.HandLandmark.MIDDLE_FINGER_TIP,mpHands.HandLandmark.PINKY_TIP]

    finger_statuses = {'RIGHT_THUMB': False,'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'LEFT_THUMB': False,'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_PINKY': False}

    for hand_index, hand_info in enumerate(results.multi_handedness):

        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        #print(hand_landmarks.landmark)
        for tip_index in finger_tips_ids:
            finger_name = tip_index.name.split("_")[0]

            if(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                finger_statuses[hand_label.upper() + "_" + finger_name] = True

                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP-2].x

        if(hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):
            finger_statuses[hand_label.upper() + "_THUMB"] = True
            count[hand_label.upper()] +=1

    if draw:
        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_img, "COUNT: ", (1100, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20, 255, 155), 2)
        cv2.putText(output_img, str(sum(count.values())), (1100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (20, 255, 155), 5)

    if display:
        return output_img, finger_statuses, count
    else:
        pass
def recognizeGesture(img, fingers_statuses, count, draw= True, display = True):

    output_img = img.copy()
    hands_labels = ['RIGHT', 'LEFT']
    hands_gestures = {'RIGHT': "UNKNOWN", "LEFT": "UNKNOWN"}

    for hand_index, hand_label in enumerate(hands_labels):
        colour = (0,0,255)

        if count[hand_label] == 2 and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
            hands_gestures[hand_label] = "V SIGN"
            colour = (0,255,0)
        elif count[hand_label] == 3 and fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
            hands_gestures[hand_label] = "SPIDERMAN SIGN"
            colour = (0,255,0)
        elif count[hand_label] == 1 and fingers_statuses[hand_label + '_THUMB']:
            hands_gestures[hand_label] = "OKAY"
            colour = (0, 255, 0)
        elif count[hand_label] == 4 and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_THUMB']:
            hands_gestures[hand_label] = "GANG SHEET"
            colour = (0,255,0)
        elif count[hand_label] == 5:
            hands_gestures[hand_label] = "HIGH-FIVE SIGN"
            colour = (0,255,0)

        if draw:
            cv2.putText(output_img, hand_label + ": " + hands_gestures[hand_label], (10, (hand_index+1)*60), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2)
    if display:
        return output_img, hands_gestures
    else:
        pass

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
    class_name = "J"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #print(handLms.landmark)

            num_chords = len(handLms.landmark)
            landmarks = ['class']
            for i in range(1, num_chords + 1):
                landmarks += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i)]

            data = handLms.landmark
            data_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in data]).flatten())
            data_row.insert(0,class_name)

            #print(data_row)
            '''with open('dataset.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(data_row)'''




        for id, lm in enumerate(handLms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            #print(lm.x,lm.y,lm.z)

            if id == 0:
                cv2.circle(img, (cx, cy), 0, (255, 0, 255), cv2.FILLED)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                              landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}

    if results.multi_hand_landmarks:
        img, fingers_statuses, count = countFingers(img, results, display = True)
        img, gestures = recognizeGesture(img, fingers_statuses, count, display=True)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if(k==27):
        break

cap.release()
cv2.destroyAllWindows()