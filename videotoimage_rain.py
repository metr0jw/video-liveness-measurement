import cv2

# Opens the Video file
cap = cv2.VideoCapture('video_RAIN.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite("RAIN/rain_image"+str(i)+".png", frame)
    i += 1

cap.release()
cv2.destroyAllWindows()