import os

import cv2
from matplotlib import pyplot as plt

KEY = "side"

POSITIVES_FILEPATH = f"video_positives/{KEY}/"
OUTPUT_FILENAME = f"ratio_plot_video_positives_{KEY}.png"


if __name__ == "__main__":
    ratio_sum = 0
    ratio_max = -1
    ratio_min = 1000
    ratio_counter = {}
    for x in range(0, 61):
        key = float(f'{x / 10:.1f}')
        ratio_counter[key] = 0

    filenames = os.listdir(POSITIVES_FILEPATH)
    for filename in filenames:
        filepath = f'{POSITIVES_FILEPATH}/{filename}'
        if os.path.isdir(filepath):
            continue
        img = cv2.imread(filepath)
        height = img.shape[0]
        width = img.shape[1]
        ratio = float(f'{width / height:.1f}')
        ratio_counter[ratio] += 1
        ratio_sum += ratio
        if ratio < ratio_min:
            ratio_min = ratio
        if ratio > ratio_max:
            ratio_max = ratio

    print(f"Mean: {ratio_sum / len(filenames):.1f}")
    print(f"Median: {max(ratio_counter, key=ratio_counter.get)}")

    x = list(ratio_counter.keys())
    y = list(ratio_counter.values())

    plt.scatter(x, y, marker='o')
    plt.xlabel('x - ratio')
    plt.ylabel('y - count')
    plt.title(
        f'Mean: {ratio_sum / len(filenames):.1f} '
        f'Median: {max(ratio_counter, key=ratio_counter.get)}\n'
        f'Min: {ratio_min} '
        f'Max: {ratio_max}'
    )
    plt.savefig(OUTPUT_FILENAME)

