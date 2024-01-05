import cv2
import matplotlib.pyplot as plt

print('Welcome to Bike Detection System by Nitin Kumar Singh')

cascade_src = 'two_wheeler.xml'
video_src = 'two_wheeler2.mp4'
cap = cv2.VideoCapture(video_src)

bike_cascade = cv2.CascadeClassifier(cascade_src)

fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()

    if (type(frame) == type(None)):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bikes = bike_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1)

    for (x, y, w, h) in bikes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 215), 2)

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)
