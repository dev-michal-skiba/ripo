import argparse

import cv2
import numpy as np

PALLET_CLASSIFICATORS_INFO = [
    {
        "name": "Paleta",
        "key": "front",
        "filepath": "europallet/33_24_0p8_2p0_front_ann_cropped_classifier/cascade.xml",
        "colour": (0, 0, 255),
        "classificator": None
    },
    {
        "name": "Paleta",
        "key": "angle",
        "filepath": "europallet/43_24_0p9_2p6_mixed_ann_cropped_classifier/cascade.xml",
        "colour": (0, 0, 255),
        "classificator": None
    },
    {
        "name": "Paleta",
        "key": "angle",
        "filepath": "europallet/51_24_0p9_4p7_side_ann_classifier/cascade.xml",
        "colour": (0, 0, 255),
        "classificator": None
    },
]

# OFFSET = 0  # for full/front testing
# OFFSET = 1750  # for mixed testing
# OFFSET = 6000  # for side testing
OFFSET = 2700  # for blue dot testing

EXPECTED_BLUE_DOT_AREA = {
    "MIN_X": 0.3,
    "MAX_X": 0.6,
    "MIN_Y": 0.5,
    "MAX_Y": 0.8,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', dest='video_path', type=str, default='./YDXJ0465-001.MP4')
    parser.add_argument('--step', dest='step', type=int, default=12)
    args = parser.parse_args()
    return args.video_path, args.step


def load_classificators():
    for pallet_classificator_info in PALLET_CLASSIFICATORS_INFO:
        pallet_classificator = cv2.CascadeClassifier()
        pallet_classificator.load(cv2.samples.findFile(pallet_classificator_info["filepath"]))
        pallet_classificator_info["classificator"] = pallet_classificator


def detect(frame, pallet_classificator):
    # Apply filters before recognition, turn into gray and equalize hist
    _frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _frame_gray = cv2.equalizeHist(_frame_gray)

    # Detect pallet
    _pallets = pallet_classificator.detectMultiScale(_frame_gray, 1.03, 3)
    return _pallets


def detect_blue_dot(frame):
    height, width, _ = frame.shape
    min_y = int(EXPECTED_BLUE_DOT_AREA["MIN_Y"] * height)
    max_y = int(EXPECTED_BLUE_DOT_AREA["MAX_Y"] * height)
    min_x = int(EXPECTED_BLUE_DOT_AREA["MIN_X"] * width)
    max_x = int(EXPECTED_BLUE_DOT_AREA["MAX_X"] * width)
    frame = frame[min_y:max_y, min_x:max_x]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 75, 140])
    upper_blue = np.array([130, 215, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 5, param1=50, param2=20, minRadius=0,
                               maxRadius=20)
    if circles is None:
        return None
    cords = circles[0][0]
    x = min_x + int(cords[0])
    y = min_y + int(cords[1])
    r = int(cords[2])
    return x, y, r


def play_video():
    video_path, step = get_args()
    pallets_info = []
    blue_dot_cords = None
    blue_dot_counter = 0
    counter = 0
    load_classificators()

    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Wait for the header")
    cap.set(cv2.CAP_PROP_POS_FRAMES, OFFSET)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        counter += 1
        if flag:
            height, width, layers = frame.shape
            new_h = int(height / 4)
            new_w = int(width / 4)
            frame = cv2.resize(frame, (new_w, new_h))

            # Zanjdź blue dot
            # Jeżeli nie wykrywa blue dota
            # poczekaj cały krok bez wykrycia żeby przestać go wyświetlać
            new_blue_dot_cords = detect_blue_dot(frame)
            if new_blue_dot_cords is None:
                blue_dot_counter += 1
                if blue_dot_counter == step:
                    blue_dot_counter = 0
                    blue_dot_cords = None
            elif blue_dot_cords is None:
                blue_dot_counter = 0
                blue_dot_cords = new_blue_dot_cords
            else:
                blue_dot_counter = 0

            if counter == step:
                counter = 0
                pallets_info = []
                for pallet_classificator_info in PALLET_CLASSIFICATORS_INFO:
                    pallets_cords = detect(
                        frame, pallet_classificator=pallet_classificator_info["classificator"]
                    )
                    pallets_info.append({
                        "cords": pallets_cords,
                        "colour": pallet_classificator_info["colour"]
                    })
                #  TODO detect pola odkładcze
            pallets_counter = 0
            x_sum = 0
            y_sum = 0
            w_sum = 0
            h_sum = 0
            for pallets in pallets_info:
                for (x, y, w, h) in pallets["cords"]:
                    pallets_counter += 1
                    x_sum += x
                    y_sum += y
                    w_sum += w
                    h_sum += h
            if pallets_counter:
                x = int(x_sum/pallets_counter)
                y = int(y_sum/pallets_counter)
                w = int(w_sum/pallets_counter)
                h = int(h_sum/pallets_counter)
                frame = cv2.putText(frame, "Paleta", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, pallets["colour"], 1, cv2.LINE_AA)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), pallets["colour"], 2)
            if blue_dot_cords is not None:
                x, y, r = blue_dot_cords
                frame = cv2.putText(
                    frame, "Blue dot", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA
                )
                frame = cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            if cv2.waitKey(1000) == 27:
                break
        if cv2.waitKey(40) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


if __name__ == '__main__':
    play_video()
