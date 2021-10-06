import shutil
import pathlib
import os
import json
from tools.x2coco import MyEncoder

input_path = "/home/dong/tmp/zuowen_resize/"
output_path = "/home/dong/tmp/test/"

os.makedirs(output_path, exist_ok=True)

for i in pathlib.Path(input_path).glob("**/*.json"):
    *_, filename = str(i).split("/")
    filename, ext = os.path.splitext(str(filename))
    img_path = f'{input_path}/{filename}.jpg'
    ann_path = f'{input_path}/{filename}.json'

    with open(ann_path) as f:
        ann_obj = json.load(f)
        for c in range(50):
            img_file = output_path + filename + str(c) + ".jpg"
            shutil.copyfile(img_path, img_file)
            ann_obj["imagePath"] = img_file
            json.dump(ann_obj, open(output_path + filename + str(c) +".json", "w"), indent=4, cls=MyEncoder)