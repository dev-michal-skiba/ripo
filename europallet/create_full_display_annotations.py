import os

import cv2

KEY = "mixed"

IMAGES_PATH = f"video_positives/{KEY}/cropped"
ANNOTATIONS_FILENAME = f"video_positives_{KEY}_cropped_annotations.txt"


def create_full_display_annotations():
    lines = []
    filenames = sorted([
        filename
        for filename in os.listdir(IMAGES_PATH)
        if os.path.isfile(f"{IMAGES_PATH}/{filename}")
    ])
    for filename in filenames:
        filepath = f"{IMAGES_PATH}/{filename}"
        img = cv2.imread(filepath)
        height = img.shape[0]
        width = img.shape[1]
        lines.append(f"{filepath} 1 0 0 {width - 1} {height - 1}\n")
    with open(ANNOTATIONS_FILENAME, 'w') as annotations_file:
        annotations_file.writelines(lines)


if __name__ == "__main__":
    create_full_display_annotations()
