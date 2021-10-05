import pathlib
import os

with open("train.txt", "w") as f:
    for i in pathlib.Path('/home/dong/tmp/zuowen_resize').glob("**/*.xml"):
        file_path, ext = os.path.splitext(str(i))

        print(f"{file_path}.jpg {i}", file=f)