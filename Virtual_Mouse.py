import cv2
import numpy as np
import Virtual_Mouse_Helper as helper
import time
import autopy

camWidth, camHeight = 640, 480
frameRate = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = helper.process_image_find_hands(max_num_hands=1)
screenWidth, screenHeight = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameRate, frameRate), (camWidth - frameRate, camHeight - frameRate), (255, 0, 255), 2)

    if fingers[1] == 1 and fingers[2] == 0:
        x3 = np.interp(x1, (frameRate, camWidth - frameRate), (0, screenWidth))
        y3 = np.interp(y1, (frameRate, camHeight - frameRate), (0, screenHeight))
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        autopy.mouse.move(camWidth - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    if fingers[1] == 1 and fingers[2] == 1:
        length, img, lineInfo = detector.findDistance(8, 12, img)
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)