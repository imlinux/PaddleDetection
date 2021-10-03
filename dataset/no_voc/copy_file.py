import shutil
import pathlib
import os

for i in pathlib.Path('/home/dong/tmp/dataset/voc/zuowen').glob("**/*.xml"):
    *_, filename = str(i).split("/")
    file_path, ext = os.path.splitext(str(i))
    file_path = file_path + ".jpg"
    shutil.copyfile(file_path, "/home/dong/tmp/test/" + filename + ".jpg")