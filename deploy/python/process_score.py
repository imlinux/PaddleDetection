import time
from urllib.parse import quote_plus
import pathlib
import cv2
import fitz
import gridfs
import numpy as np
from bson import ObjectId
from bson.json_util import loads
from imutils import perspective
from pymongo import MongoClient

from keypoint_predict import KeyPointPredict, split
from predict import Predict
import os

model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"
predict = Predict(model_dir)


def sort_contours(cnts):

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(zip(cnts, boundingBoxes), key=lambda b: (b[1][1], b[1][0]), reverse=False)


def predict_img(img_list):
    ret = ""
    info = []
    c = []
    for img in img_list:

        if img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] == 0:
            ret += " "
            info.append([])
            continue

        result = predict([img])
        boxes = result["boxes"]

        if len(boxes) > 0:
            r = boxes[boxes[:, 1].argsort()[::-1]]
            info.append(r.tolist())
            c.append(r[0][1])
            if r[0][1] >= 0.6 :
                ret += str(int(r[0][0]))
            else:
                ret += " "
        else:
            info.append([])
            ret += " "
            c.append(0)
    print(c)
    return ret, info


def process_img(img_raw):
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    thresh, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    kernel_length = np.array(img_bin).shape[1] // 40
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    verticle_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.morphologyEx(verticle_lines_img, cv2.MORPH_CLOSE, verticle_kernel, iterations=100)

    horizontal_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.morphologyEx(horizontal_lines_img, cv2.MORPH_CLOSE, hori_kernel, iterations=100)

    alpha = 0.5
    beta = 1.0 - alpha

    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)

    _, img_final_bin = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    table_cells = []
    for c, b in sort_contours(contours):

        x, y, w, h = b

        if 80 < w < 170 and 80 < h < 170:
            new_img = img_raw[y:y + h, x:x + w]
            table_cells.append(new_img)

    table_rows = []

    for i in range(0, len(table_cells), 9):
        table_rows.append(table_cells[i: i + 9])

    return table_rows


def process_one(pdf_file_path):
    with fitz.open(pdf_file_path) as pdf_file:

        for page_num in range(0, len(pdf_file), 2):

            table_rows = []

            page0_img = pdf_file[page_num].getImageList()[0][0]
            page1_img = pdf_file[page_num + 1].getImageList()[0][0]

            base_image = pdf_file.extractImage(page0_img)
            image_bytes = base_image["image"]
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

            table_rows += process_img(img)

            base_image = pdf_file.extractImage(page1_img)
            image_bytes = base_image["image"]
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

            table_rows += process_img(img)

            for table_row in table_rows:
                print(predict_img(table_row))


def main():

    for path in pathlib.Path("/home/dong/tmp/score").glob("**/*.pdf"):
        print(path)
        process_one(str(path))


if __name__ == "__main__":
    main()
