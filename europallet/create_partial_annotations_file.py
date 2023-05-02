import os

BASE_ANNOTATIONS_FILEPATH = "video_positives_annotations.txt"
OUTPUT_ANNOTATIONS_FILENAME = "video_positives_side_annotations.txt"
IMAGES_DIRECTORY = "video_positives/side/raw/"
ANNOTATION_LINE_FORMAT = "{file_path}{filename} {value}\n"


def create_annotations_mapping():
    annotations_mapping = {}
    with open(BASE_ANNOTATIONS_FILEPATH) as base_annotations_file:
        for line in base_annotations_file.readlines():
            line = line.strip(" \n")
            if not line:
                continue
            values = line.split(' ')
            filepath = values[0]
            name = os.path.basename(filepath)
            value = " ".join(values[1:])
            annotations_mapping[name] = value
    return annotations_mapping


def create_annotations_file():
    annotations_mapping = create_annotations_mapping()
    lines = []
    for filename in os.listdir(IMAGES_DIRECTORY):
        lines.append(ANNOTATION_LINE_FORMAT.format(
            file_path=IMAGES_DIRECTORY, filename=filename, value=annotations_mapping[filename])
        )
    with open(OUTPUT_ANNOTATIONS_FILENAME, "w") as output_annotations_file:
        output_annotations_file.writelines(lines)


if __name__ == "__main__":
    create_annotations_file()

