import cv2
import matplotlib.pyplot as plt

print('Welcome to Pedestrian Detection AI by Nitin Kumar Singh')

video_source = 'pedestrians.avi'
video_capture = cv2.VideoCapture(video_source)

pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')


while True:
    
    ret, frame = video_capture.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrians = pedestrian_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2)
   
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 210), 4)

    
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)  
