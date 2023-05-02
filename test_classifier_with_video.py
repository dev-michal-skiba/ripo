import argparse

import cv2

PALLET_CLASSIFICATORS_INFO = [
    {
        "name": "Pallet seen from front",
        "key": "front",
        "filepath": "europallet/33_24_0p8_2p0_front_ann_classifier/cascade.xml",
        "colour": (255, 0, 0),
        "classificator": None
    },
    {
        "name": "Pallet seen from angle",
        "key": "angle",
        "filepath": "europallet/44_24_0p5_3p0_mixed_ann_classifier/cascade.xml",
        "colour": (0, 255, 0),
        "classificator": None
    },
    {
        "name": "Pallet seen from side",
        "key": "angle",
        "filepath": "europallet/51_24_0p9_4p7_side_ann_classifier/cascade.xml",
        "colour": (0, 0, 255),
        "classificator": None
    },
]

OFFSET = 0  # for full/front testing
# OFFSET = 6000  # for side testing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', dest='video_path', type=str, default='./YDXJ0465-001.MP4')
    parser.add_argument('--step', dest='step', type=int, default=24)
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
    _pallets = pallet_classificator.detectMultiScale(_frame_gray, 1.01, 3)
    return _pallets


def play_video():
    video_path, step = get_args()
    pallets_info = []
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
            if counter == step:
                counter = 0
                pallets_info = []
                for pallet_classificator_info in PALLET_CLASSIFICATORS_INFO:
                    pallets_cords = pallets = detect(
                        frame, pallet_classificator=pallet_classificator_info["classificator"]
                    )
                    pallets_info.append({
                        "cords": pallets_cords,
                        "colour": pallet_classificator_info["colour"]
                    })
                #  TODO detect pola odkładcze
            for pallets in pallets_info:
                for (x, y, w, h) in pallets["cords"]:
                    center = (x + w // 2, y + h // 2)
                    frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, pallets["colour"], 4)
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print(str(pos_frame) + " frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(5) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


if __name__ == '__main__':
    play_video()
