import os

import cv2

pallet_cascade = cv2.CascadeClassifier()
pallet_cascade.load(cv2.samples.findFile('europallet/europallet_9_classifier/cascade.xml'))


def detect(frame):
    # Apply filters before recognition, turn into gray and equalize hist
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect pallet
    pallets = pallet_cascade.detectMultiScale(frame_gray, 1.01, 3)
    return pallets


def detect_and_display(frame):
    pallets = detect(frame)
    for (x, y, w, h) in pallets:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    cv2.imshow('Pallet detection', frame)
    cv2.waitKey(0)


def display_test_images():
    for filename in os.listdir('test_images'):
        filepath = f'test_images/{filename}'
        if os.path.isfile(filepath):
            frame = cv2.imread(filepath)
            detect_and_display(frame)


if __name__ == '__main__':
    display_test_images()
