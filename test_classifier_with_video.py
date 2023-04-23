import cv2
from test_pallet_classifier import detect

STEP = 7

video_filepath = "./YDXJ0465-001.MP4"
pallets = ()
counter = 0

cap = cv2.VideoCapture(video_filepath)
while not cap.isOpened():
    cap = cv2.VideoCapture(video_filepath)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    counter += 1
    if flag:
        height, width, layers = frame.shape
        new_h = int(height / 4)
        new_w = int(width / 4)
        frame = cv2.resize(frame, (new_w, new_h))
        if counter % STEP == 0:
            cv2.imwrite(f'europallet/video_positives/{int(counter / 7)}.jpg', frame)
            counter = 0
            pallets = detect(frame)
            # TODO detect pola odkładcze
        for (x, y, w, h) in pallets:
            center = (x + w // 2, y + h // 2)
            frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break

# print(counter, counter)
