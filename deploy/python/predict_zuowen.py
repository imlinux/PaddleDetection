import cv2
import fitz
import numpy as np
import os
from predict import Predict

model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no"
tmp_pdf_dir = "/home/dong/tmp/tmp_pdf"


def image_list(pdf_file):
    *_, filename = pdf_file.split("/")
    doc = fitz.open(pdf_file)

    os.makedirs(tmp_pdf_dir, exist_ok=True)

    for page_idx in range(1, len(doc), 2):
        page = doc[page_idx]
        for img in page.getImageList():
            base_image = doc.extractImage(img[0])
            image_bytes = base_image["image"]

            new_doc = fitz.Document()
            new_doc.insert_pdf(doc, from_page= page_idx - 1, to_page = page_idx)
            tmp_pdf_file = f"{tmp_pdf_dir}/{filename}{page_idx -1}-{page_idx}.pdf"
            new_doc.save(tmp_pdf_file)

            yield *split(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)), tmp_pdf_file


def split(img):

    row = 790
    col = 1716
    w = 100
    h = 76
    account = []

    for i in range(6):
        account.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    row = 3230
    col = 232
    w = 414 - col
    h = 3351 - row
    score = []

    for i in range(8):
        score.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    return account, score


predict = Predict(model_dir)
for (account, score, file_path) in image_list("/home/dong/tmp/zuowen/JUYE_F_00007.pdf"):
    for img in account:
        result = predict([img])
        boxes = result["boxes"]
        print(boxes[boxes[:, 1].argsort()[::-1]][0:2])
        cv2.imshow("", img)
        cv2.waitKey(0)
        print("*"*20)
