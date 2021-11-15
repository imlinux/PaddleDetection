import cv2

from my_common import extract_img_from_pdf
from predict import Predict
from keypoint_predict import KeyPointPredict
import numpy as np
from imutils import perspective
import os
import json
import pathlib

key_point_model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"
keyPointPredict = KeyPointPredict(key_point_model_dir)
predict = Predict(model_dir)


def split():

    row = 38
    col = 750
    w = 50
    h = 38
    account = []

    for i in range(6):
        #account.append(img[row: row + h, col + i * w: col + (i + 1) * w])
        account.append([(col + i * w, row), (col + (i + 1) * w, row + h)])

    # 张江
    # row = 2470
    # col = 1
    # w = 184
    # h = 140

    row = 1255
    col = 1
    w = 92
    h = 61
    score = []

    for i in range(8):
        #score.append(img[row: row + h, col + i * w: col + (i + 1) * w])
        score.append([(col + i * w, row), (col + (i + 1) * w, row + h)])

    return account, score


def process_img(img, pdf_name, page_num):
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    result = keyPointPredict(img)
    skeletons, scores = result["keypoint"]
    pts = np.array(
        [skeletons[0][0][0: 2], skeletons[0][1][0: 2], skeletons[0][2][0: 2], skeletons[0][3][0: 2]])

    img_result = perspective.four_point_transform(img, pts)
    accounts, scores = split()

    items = []
    for i in range(len(accounts)):
        (account_xmin, account_ymin), (account_xmax, account_ymax) = accounts[i]
        # img_result = cv2.rectangle(img_result, (account_xmin, account_ymin), (account_xmax, account_ymax), (0, 0, 255), thickness=1)
        result = predict([img_result[account_ymin: account_ymax, account_xmin: account_xmax]])
        boxes = result["boxes"]
        if len(boxes) > 0:
            box = boxes[boxes[:, 1].argsort()[::-1]][0]
            clsid, bbox, score = int(box[0]), box[2:], box[1]
            if score >= 0.7:
                xmin, ymin, xmax, ymax = bbox
                items.append({
                    "transcription": str(clsid),
                    "points": [
                        [account_xmin + xmin, account_ymin + ymin],
                        [account_xmin + xmax, account_ymin + ymin],
                        [account_xmin + xmax, account_ymin + ymax],
                        [account_xmin + xmin, account_ymin + ymax]
                    ],
                    "difficult": False
                })

    for i in range(len(scores)):
        (score_xmin, score_ymin), (score_xmax, score_ymax) = scores[i]
        # img_result = cv2.rectangle(img_result, (score_xmin, score_ymin), (score_xmax, score_ymax), (0, 0, 255), thickness=1)
        result = predict([img_result[score_ymin: score_ymax, score_xmin: score_xmax]])
        boxes = result["boxes"]
        if len(boxes) > 0:
            box = boxes[boxes[:, 1].argsort()[::-1]][0]
            clsid, bbox, score = int(box[0]), box[2:], box[1]
            if score >= 0.7:
                xmin, ymin, xmax, ymax = bbox
                items.append({
                    "transcription": str(clsid),
                    "points": [
                        [score_xmin + xmin, score_ymin + ymin],
                        [score_xmin + xmax, score_ymin + ymin],
                        [score_xmin + xmax, score_ymin + ymax],
                        [score_xmin + xmin, score_ymin + ymax]
                    ],
                    "difficult": False
                })

    # cv2.imshow("", img_result)
    # cv2.waitKey(0)

    os.makedirs("/home/dong/tmp/no/zuowen", exist_ok=True)
    img_file_name = f"/home/dong/tmp/no/zuowen/{pdf_name}-{page_num}.jpg"
    cv2.imwrite(img_file_name, img_result)
    with open("/home/dong/tmp/no/zuowen/Label.txt", "a") as f:
        print("zuowen/" + img_file_name.split("/")[-1] + "\t" + json.dumps(items), file=f)


def process_one_pdf(file):

    imgs = list(extract_img_from_pdf(file))
    imgs = [imgs[i] for i in range(1, len(imgs), 2)]
    file_name = file.split("/")[-1]
    for idx, img in enumerate(imgs):
        process_img(img, file_name, idx)


for path in pathlib.Path("/home/dong/tmp/zuowen").glob("**/*.pdf"):
    process_one_pdf(str(path))