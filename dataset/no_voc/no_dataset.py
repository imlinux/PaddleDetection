import os
import pathlib

with open("train.txt", "w") as output:
    for f in pathlib.Path("/home/dong/tmp/dataset/voc/no/no-PascalVOC-export/ImageSets/Main").glob("**/*train.txt"):

        with open(f) as input:
            for line in input:
                filename, _ = line.split(" ")

                annotation_file = os.path.splitext(filename)[0] + ".xml"
                print(f"/home/dong/tmp/dataset/voc/no/no-PascalVOC-export/JPEGImages/{filename} /home/dong/tmp/dataset/voc/no/no-PascalVOC-export/Annotations/{annotation_file}", file=output)