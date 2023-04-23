"""
Used combinations:
    - `python train.py --width 38 --height 24 --min-ratio 1.1 --max-ratio 2.1`
    - `python train.py --width 32 --height 24 --min-ratio 1.2 --max-ratio 1.4`
    - `python train.py --width 40 --height 30 --min-ratio 1.2 --max-ratio 1.4`
"""


import argparse
import math
import os

import cv2

TOTAL_SAMPLES_NUMBER = 2500


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', dest='width', type=int, required=True)
    parser.add_argument('--height', dest='height', type=int, required=True)
    parser.add_argument('--min-ratio', dest='min_ratio', type=float, required=True)
    parser.add_argument('--max-ratio', dest='max_ratio', type=float, required=True)
    args = parser.parse_args()
    return args.width, args.height, args.min_ratio, args.max_ratio


def get_filepaths(min_ratio, max_ratio):
    filepaths = []
    for filename in os.listdir('positives'):
        filepath = f'positives/{filename}'
        img = cv2.imread(filepath)
        height = img.shape[0]
        width = img.shape[1]
        ratio = float(f'{width / height:.1f}')
        if min_ratio <= ratio <= max_ratio:
            filepaths.append(filepath)
    return filepaths


def main():
    width, height, min_ratio, max_ratio = get_args()
    name = f"{width}_{height}_{str(min_ratio).replace('.', 'p')}" \
           f"_{str(max_ratio).replace('.', 'p')}"
    filepaths = get_filepaths(min_ratio, max_ratio)
    if os.path.isdir(name) is False:
        os.mkdir(name)
    samples_number = math.ceil(TOTAL_SAMPLES_NUMBER / len(filepaths))
    for i, filepath in enumerate(filepaths):
        os.system(
            f'opencv_createsamples -vec {name}/{i}.vec -img {filepath} -bg negatives.txt '
            f'-w {width} -h {height} -num {samples_number}')
    os.system(f'python mergevec.py -v {name}/ -o {name}.vec')
    if os.path.isdir(f'{name}_classifier') is False:
        os.mkdir(f'{name}_classifier')
    os.system(
        f'opencv_traincascade -data {name}_classifier/ -vec {name}.vec -bg negatives.txt '
        f'-w {width} -h {height}'
    )


if __name__ == '__main__':
    main()
