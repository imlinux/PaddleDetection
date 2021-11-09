import pathlib
import cv2
from imutils import perspective
import time
import os
import numpy as np
from predict import Predict
from my_common import extract_img_from_pdf
import json

model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"
predict = Predict(model_dir)

def sort_contours(cnts):

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(zip(cnts, boundingBoxes), key=lambda b: (b[1][1], b[1][0]), reverse=False)


def edge_detection(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # blur = ~blur
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # verticle_lines_img = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=10)
    # verticle_lines_img = ~verticle_lines_img

    edge = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        ((x, y), (w, h), angle) = cv2.minAreaRect(c)
        if 500 < w and 500 < h:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                t = np.array(approx)
                t = t.reshape(-1, 2)
                print("透视转换")
                processed_img = perspective.four_point_transform(image, t)
                processed_img = cv2.rectangle(processed_img, (0, 0), (processed_img.shape[1], processed_img.shape[0]), (0, 0, 0), thickness=10)
                return image, processed_img
            else:
                os.makedirs("./output", exist_ok=True)
                cv2.imwrite(f"./output/透视{time.time()}.jpg",
                            cv2.drawContours(image.copy(), [c], -1, (0, 0, 255), 3))

    print("未做透视转换")
    return image, image


def process_img(img_raw, pdf_name, page_num):
    img_raw = cv2.resize(img_raw, (img_raw.shape[1] // 2, img_raw.shape[0] // 2))
    img_raw, img = edge_detection(img_raw)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    items = []
    for c, b in sort_contours(contours):

        x, y, w, h = b

        if 40 < w < 600 and 40 < h < 170:
            new_img = img[y:y + h, x:x + w]
            result = predict([new_img])
            boxes = result["boxes"]

            for box in boxes:
                clsid, bbox, score = int(box[0]), box[2:], box[1]
                if score >= 0.7:
                    xmin, ymin, xmax, ymax = bbox
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)
                    x = int(x + xmin)
                    y = int(y + ymin)
                    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
                    # cv2.imshow("", img)
                    # cv2.waitKey(0)
                    items.append({
                        "transcription": str(clsid),
                        "points": [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ],
                        "difficult": False
                    })

    os.makedirs("/home/dong/tmp/tmp3", exist_ok=True)
    img_file_name = f"/home/dong/tmp/tmp3/{pdf_name}-{page_num}.jpg"
    cv2.imwrite(img_file_name, img)
    with open("/home/dong/tmp/tmp3/Label.txt", "a") as f:
        print("tmp3/" + img_file_name.split("/")[-1] + "\t" + json.dumps(items), file=f)


def main():
    for path in pathlib.Path("/home/dong/tmp/score").glob("**/*.pdf"):
        path_str = str(path)
        *_, file_name = path_str.split("/")

        for idx, img in enumerate(extract_img_from_pdf(path_str)):
            process_img(img, file_name, idx)


if __name__ == "__main__":
    main()