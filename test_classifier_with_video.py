import argparse

import cv2
import numpy as np

PALLET_CLASSIFICATORS_INFO = [
    {
        "text": "Paleta",
        "key": "front",
        "filepath": "europallet/front.xml",  # 33 24
        "colour": (0, 0, 255),
        "classificator": None
    },
    {
        "text": "Paleta",
        "key": "angle",
        "filepath": "europallet/angle.xml",  # 43 24
        "colour": (0, 0, 255),
        "classificator": None
    },
    {
        "text": "Paleta",
        "key": "side",
        "filepath": "europallet/side.xml",  # 51 24
        "colour": (0, 0, 255),
        "classificator": None
    },
]

OFFSET = 0  # for full/front testing
# OFFSET = 1750  # for angle testing
# OFFSET = 6000  # for side testing
# OFFSET = 2700  # for blue dot testing

EXPECTED_BLUE_DOT_AREA = {
    "MIN_X": 0.3,
    "MAX_X": 0.6,
    "MIN_Y": 0.5,
    "MAX_Y": 0.8,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', dest='video_path', type=str, default='./YDXJ0465-001.MP4')
    parser.add_argument('--step', dest='step', type=int, default=6)
    args = parser.parse_args()
    return args.video_path, args.step


def load_classificators():
    for pallet_classificator_info in PALLET_CLASSIFICATORS_INFO:
        pallet_classificator = cv2.CascadeClassifier()
        pallet_classificator.load(cv2.samples.findFile(pallet_classificator_info["filepath"]))
        pallet_classificator_info["classificator"] = pallet_classificator


def get_resized_frame(frame, reduce_factor=1):
    height, width, layers = frame.shape
    new_h = int(height / reduce_factor)
    new_w = int(width / reduce_factor)
    return cv2.resize(frame, (new_w, new_h))


def get_pallets_info(frame):
    pallets_info = []
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    for pallet_classificator_info in PALLET_CLASSIFICATORS_INFO:
        pallet_classificator = pallet_classificator_info["classificator"]
        pallet_cords = pallet_classificator.detectMultiScale(frame, 1.03, 3)
        pallets_info.append({
            "text": pallet_classificator_info["text"],
            "cords": pallet_cords,
            "colour": pallet_classificator_info["colour"]
        })
    return pallets_info


def get_blue_dot_info(frame):
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
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 5, param1=50, param2=20, minRadius=0, maxRadius=20
    )
    if circles is None:
        return []
    cords = circles[0][0]
    x = min_x + int(cords[0])
    y = min_y + int(cords[1])
    r = int(cords[2])
    return [
        {
            "text": "Blue dot",
            "cords": [[x - r, y - r, 2 * r, 2 * r]],
            "colour": (255, 0, 0)
        }
    ]


def get_empty_spot_info(res_frame):
    res_frame = res_frame.copy()
    hsv = cv2.cvtColor(res_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 75, 140])
    upper_blue = np.array([150, 215, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    frame = cv2.bitwise_and(res_frame, res_frame, mask=mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty_spots_info = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) >= 1000:
            x, y, w, h = cv2.boundingRect(approx)
            empty_spots_info.append({
                "text": "Puste pole",
                "cords": [[x, y, w, h]],
                "colour": (0, 255, 0)
            })
    return empty_spots_info


def play_video():
    video_path, step = get_args()
    objects_info = []
    frame_counter = 0
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
        if flag:
            frame = get_resized_frame(frame, reduce_factor=4)

            # Wykrywaj obiekty tylko na co którejś klatce
            if frame_counter % step == 0:
                objects_info = []
                objects_info += get_pallets_info(frame)
                objects_info += get_blue_dot_info(frame)
                objects_info += get_empty_spot_info(frame)

            # Zaznaczaj obiekty na każdej klatce
            for object_info in objects_info:
                for (x, y, w, h) in object_info["cords"]:
                    frame = cv2.putText(
                        frame, object_info["text"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        object_info["colour"], 1, cv2.LINE_AA
                    )
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), object_info["colour"], 2)
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            if cv2.waitKey(1000) == 27:
                break
        if cv2.waitKey(40) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        frame_counter += 1


if __name__ == '__main__':
    play_video()
