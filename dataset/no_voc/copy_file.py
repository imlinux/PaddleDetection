import shutil
import pathlib
import os

input_path = "/home/dong/tmp/zuowen_resize/"
output_path = "/home/dong/tmp/test/"

os.makedirs(output_path, exist_ok=True)

for i in pathlib.Path(input_path).glob("**/*.json"):
    *_, filename = str(i).split("/")
    filename, ext = os.path.splitext(str(filename))
    img_path = f'{input_path}/{filename}.jpg'
    ann_path = f'{input_path}/{filename}.json'
    shutil.copyfile(img_path, output_path + filename + ".jpg")
    shutil.copyfile(ann_path, output_path + filename + ".json")