import os

import cv2
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ratio_sum = 0
    ratio_counter = {}
    for x in range(0, 61):
        key = float(f'{x / 10:.1f}')
        ratio_counter[key] = 0

    filenames = os.listdir('positives')
    for filename in filenames:
        filepath = f'positives/{filename}'
        img = cv2.imread(filepath)
        height = img.shape[0]
        width = img.shape[1]
        ratio = float(f'{width / height:.1f}')
        ratio_counter[ratio] += 1
        ratio_sum += ratio

    print(f"Mean: {ratio_sum / len(filenames):.1f}")
    print(f"Median: {max(ratio_counter, key=ratio_counter.get)}")

    x = list(ratio_counter.keys())
    y = list(ratio_counter.values())

    plt.scatter(x, y, marker='o')
    plt.xlabel('x - ratio')
    plt.ylabel('y - count')
    plt.title(
        f'Mean: {ratio_sum / len(filenames):.1f}\n'
        f'Median: {max(ratio_counter, key=ratio_counter.get)}'
    )
    plt.savefig('ratio_plot.png')

