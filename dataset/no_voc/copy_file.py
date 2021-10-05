import shutil
import pathlib
import os

for i in pathlib.Path('/home/dong/tmp/zuowen_resize').glob("**/*.xml"):
    *_, filename = str(i).split("/")
    file_path, ext = os.path.splitext(str(i))
    img_path = file_path + ".jpg"
    ann_path = file_path + ".xml"
    shutil.copyfile(img_path, "/home/dong/tmp/test/" + filename + ".jpg")
    shutil.copyfile(ann_path, "/home/dong/tmp/test/" + filename + ".xml")