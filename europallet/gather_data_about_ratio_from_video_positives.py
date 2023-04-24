from matplotlib import pyplot as plt

if __name__ == "__main__":
    ratio_sum = 0
    ratio_counter = {}
    counter = 0
    for x in range(0, 101):
        key = float(f'{x / 10:.1f}')
        ratio_counter[key] = 0

    with open('video_positives_annotations.txt') as annotations_file:
        for line in annotations_file.readlines():
            line = line.strip('\n')
            if not line:
                continue
            values = line.split(' ')
            samples_count = int(values[1])
            annotations = [values[2 + i * 4: 2 + (i + 1) * 4] for i in range(samples_count)]
            for i, annotation in enumerate(annotations):
                counter += 1
                width = int(annotation[2])
                height = int(annotation[3])
                ratio = float(f'{width / height:.1f}')
                ratio_counter[ratio] += 1
                ratio_sum += ratio

    x = list(ratio_counter.keys())
    y = list(ratio_counter.values())

    plt.scatter(x, y, marker='o')
    plt.xlabel('x - ratio')
    plt.ylabel('y - count')
    plt.title(
        f'Mean: {ratio_sum / counter:.1f}\n'
        f'Median: {max(ratio_counter, key=ratio_counter.get)}'
    )
    plt.savefig('ratio_plot_for_video_positives.png')
