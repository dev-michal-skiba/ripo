import cv2

ANNOTATIONS_FILEPATH = "positives_raw.txt"
IMAGES_PATH_FORMAT = "positives/img_{counter}.jpg"


if __name__ == '__main__':
    counter = 0
    with open(ANNOTATIONS_FILEPATH) as file:
        for line in file.readlines():
            line = line.strip('\n')
            if not line:
                continue
            values = line.split(' ')
            filepath = values[0]
            samples_count = int(values[1])
            img = cv2.imread(filepath)
            max_height = img.shape[0]
            max_width = img.shape[1]
            annotations = [values[2 + i * 4: 2 + (i + 1) * 4] for i in range(samples_count)]
            for i, annotation in enumerate(annotations):
                x = int(annotation[0])
                y = int(annotation[1])
                width = int(annotation[2])
                height = int(annotation[3])
                if width < 24:
                    f'Rejecting {i}st/nd/rd/th sample from {filepath}, width is less than 24'
                    continue
                if height < 24:
                    f'Rejecting {i}st/nd/rd/th sample from {filepath}, height is less than 24'
                if x + width >= max_width:
                    width = max_width - x - 1
                if y + height >= max_height:
                    height = max_height - y - 1
                cropped_img = img[y:y + height, x:x + width].copy()
                cv2.imwrite(IMAGES_PATH_FORMAT.format(counter=counter), cropped_img)
                counter += 1
