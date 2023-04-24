import argparse

import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classificator-path', dest='classificator_path', type=str, default='europallet/video_positives_classifier/cascade.xml')
    parser.add_argument('--video-path', dest='video_path', type=str, default='./YDXJ0465-001.MP4')
    parser.add_argument('--step', dest='step', type=int, default=24)
    args = parser.parse_args()
    return args.classificator_path, args.video_path, args.step


def detect(frame, pallet_cascade):
    # Apply filters before recognition, turn into gray and equalize hist
    _frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _frame_gray = cv2.equalizeHist(_frame_gray)

    # Detect pallet
    _pallets = pallet_cascade.detectMultiScale(_frame_gray, 1.01)
    return _pallets


def play_video():
    classificator_path, video_path, step = get_args()
    pallets = ()
    counter = 0
    pallet_cascade = cv2.CascadeClassifier()
    pallet_cascade.load(cv2.samples.findFile(classificator_path))

    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
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
            if counter % step == 0:
                cv2.imwrite(f'europallet/video_positives/{int(counter / 7)}.jpg', frame)
                pallets = detect(frame, pallet_cascade)
                # TODO detect pola odkładcze
            for (x, y, w, h) in pallets:
                center = (x + w // 2, y + h // 2)
                frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame) + " frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


if __name__ == '__main__':
    play_video()
