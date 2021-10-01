import os
import pathlib

root_dir = "/home/dong/tmp/dataset/voc/no"


def txt(mode="train"):
    with open(f"{mode}.txt", "w") as output:
        for f in pathlib.Path(f"{root_dir}/no-PascalVOC-export/ImageSets/Main").glob(f"**/*{mode}.txt"):

            with open(f) as input:
                for line in input:
                    filename, _ = line.split(" ")

                    annotation_file = os.path.splitext(filename)[0] + ".xml"
                    print(f"{root_dir}/no-PascalVOC-export/JPEGImages/{filename} {root_dir}/no-PascalVOC-export/Annotations/{annotation_file}", file=output)


txt(mode="train")
txt(mode="val")