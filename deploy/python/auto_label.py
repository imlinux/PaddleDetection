import pathlib
import cv2
import os
from predict import Predict


def xml(file_name, file_path, width, height, depth, obj_name, xmin, ymin, xmax, ymax):

    return f"""
        <annotation>
        <folder>student_account</folder>
        <filename>{file_name}</filename>
        <path>{file_path}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>{depth}</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>{obj_name}</name>
            <pose>Unspecified</pose>
            <truncated>1</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{xmin}</xmin>
                <ymin>{ymin}</ymin>
                <xmax>{xmax}</xmax>
                <ymax>{ymax}</ymax>
            </bndbox>
        </object>
    </annotation>
    """


model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no"
predict = Predict(model_dir)


def main():
    file_path = "/home/dong/tmp/no"
    for f in pathlib.Path(file_path).glob("**/*.jpg"):
        *_, filename = str(f).split("/")
        filename, ext = os.path.splitext(filename)
        xml_file_path = file_path + "/" + filename + ".xml"

        if os.path.exists(xml_file_path): continue

        img = cv2.imread(str(f))

        result = predict([img])
        boxes = result["boxes"]
        if len(boxes) > 0:
            r = boxes[boxes[:, 1].argsort()[::-1]]
            dt = r[0]
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
            xmin, ymin, xmax, ymax = bbox

            if score >= 0.6:
                xml_str = xml(filename + ext, f, img.shape[1], img.shape[0], len(img.shape), str(clsid), xmin, ymin, xmax, ymax)
                with open(xml_file_path, "w") as xml_file:
                    print(xml_str, file=xml_file)
            print(f'{f=} {clsid=} {score=}')


if __name__ == '__main__':
    main()