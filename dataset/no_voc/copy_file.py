import shutil
import pathlib
import os

input_path = "/home/dong/tmp/dataset/voc/zuowen"
output_path = "/home/dong/tmp/no_train/"

os.makedirs(output_path, exist_ok=True)

for i in pathlib.Path(input_path).glob("**/*.xml"):
    *_, filename = str(i).split("/")
    file_path, ext = os.path.splitext(str(i))
    img_path = file_path + ".jpg"
    ann_path = file_path + ".xml"
    shutil.copyfile(img_path, output_path + filename + ".jpg")
    shutil.copyfile(ann_path, output_path + filename + ".xml")