import cv2
vidcap = cv2.VideoCapture('video_ITZY.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("itzy/itzy_image"+str(count)+".png", image)     # save frame as PNG file
    return hasFrames


sec = 0
frameRate = 1/24
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)