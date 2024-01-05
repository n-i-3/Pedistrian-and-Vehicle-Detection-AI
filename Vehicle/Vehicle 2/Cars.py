import cv2
import matplotlib.pyplot as plt

print('Cars detection AI')
print('By Nitin Kumar Singh')

cascade_src = 'cars.xml'
video_src = 'sample_video.avi'
cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)

fig, ax = plt.subplots()

while True:
    ret, img = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)

