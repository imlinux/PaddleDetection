from pymongo import MongoClient
from urllib.parse import quote_plus
import fitz
from bson.json_util import loads
import numpy as np
import cv2
from keypoint_predict import KeyPointPredict
from predict import Predict
from imutils import perspective

file_infos = [

    ("JUYE_F_00007.pdf", "5ea43fc7ebf5f3a540e44f7d"),
    ("JUYE_F_00008.pdf", "5ea43fc7ebf5f3a540e44f7e")
]

key_point_model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no"


keyPointPredict = KeyPointPredict(key_point_model_dir)
predict = Predict(model_dir)

def split(img):

    row = 80
    col = 1507
    w = 100
    h = 76
    account = []

    for i in range(6):
        account.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    row = 2498
    col = 0
    w = 184
    h = 122
    score = []

    for i in range(8):
        score.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    return account, score


def db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
    client = MongoClient(uri)
    return client["sigmai"]


def predict_img(img_list):
    ret = ""
    info = []
    for img in img_list:
        result = predict([img])
        boxes = result["boxes"]

        if len(boxes) > 0:
            r = boxes[boxes[:, 1].argsort()[::-1]]
            info.append(r)
            ret += str(int(r[0][0]))
        else:
            info.append([])
            ret += " "

    return ret, info


def process_one(pdf_file_path):
    with fitz.open(pdf_file_path) as pdf_file:
        for page_num in range(1, len(pdf_file), 2):
            pdf_doc = fitz.Document()
            pdf_doc.insert_pdf(pdf_file, from_page=page_num - 1, to_page=page_num)
            pdf_doc.save("/home/dong/tmp/tmp.pdf")

            page = pdf_file[page_num]
            for img in page.getImageList():
                base_image = pdf_file.extractImage(img[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                result = keyPointPredict(img)
                skeletons, scores = result["keypoint"]
                pts = np.array(
                    [skeletons[0][0][0: 2], skeletons[0][1][0: 2], skeletons[0][2][0: 2], skeletons[0][3][0: 2]])

                img_result = perspective.four_point_transform(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1), pts)
                account, score = split(img_result)

                account_str, _ = predict_img(account)
                score_str, _ = predict_img(score)

                print(f"{account_str=} {score_str=}")


process_one("/home/dong/tmp/zuowen/JUYE_F_00007.pdf")