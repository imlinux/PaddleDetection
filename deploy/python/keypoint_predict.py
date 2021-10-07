import numpy as np
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
import cv2
from imutils import perspective
import os
import pathlib


class KeyPointPredict:

    def __init__(self, model_dir):
        pred_config = PredictConfig_KeyPoint(model_dir)
        self.detector = KeyPoint_Detector(
            pred_config,
            model_dir,
            use_gpu=False,
            cpu_threads=1)

    def __call__(self, *args, **kwargs):
        return self.detector.predict(*args)


def split(img):

    row = 85
    col = 1510
    w = 100
    h = 76
    account = []

    for i in range(6):
        account.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    row = 2510
    col = 1
    w = 184
    h = 122
    score = []

    for i in range(8):
        score.append(img[row: row + h, col + i * w: col + (i + 1) * w])

    return account, score


def main():
    model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
    img_path = "/home/dong/tmp/zuowen/img/0/JUYE_F_00015.pdf-51.jpg"
    img = np.array(cv2.imread(img_path))

    keyPointPredict = KeyPointPredict(model_dir)
    result = keyPointPredict(img)
    skeletons, scores = result["keypoint"]

    pts = np.array([skeletons[0][0][0: 2], skeletons[0][1][0: 2], skeletons[0][2][0: 2], skeletons[0][3][0: 2]])
    print(pts)
    img_result = perspective.four_point_transform(cv2.imread(img_path), pts)
    cv2.imwrite("/home/dong/tmp/tmp.jpg", img_result)


def main1():
    model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
    keyPointPredict = KeyPointPredict(model_dir)

    os.makedirs("/home/dong/tmp/tmp2", exist_ok=True)

    for img_path in pathlib.Path("/home/dong/tmp/zuowen/img/0").glob("**/*.jpg"):
        imgs = np.array(cv2.imread(str(img_path)))
        result = keyPointPredict(imgs)
        skeletons, scores = result["keypoint"]
        pts = np.array([skeletons[0][0][0: 2], skeletons[0][1][0: 2], skeletons[0][2][0: 2], skeletons[0][3][0: 2]])
        img_result = perspective.four_point_transform(cv2.imread(str(img_path)), pts)

        account, score = split(img_result)
        *_, filename = str(img_path).split("/")

        for idx, a in enumerate(account):
            cv2.imwrite(f"/home/dong/tmp/tmp2/{filename}-account{idx}.jpg", a)
            pass

        for idx, s in enumerate(score):
            cv2.imwrite(f"/home/dong/tmp/tmp2/{filename}-score{idx}.jpg", s)

        cv2.imwrite(f"/home/dong/tmp/tmp2/{filename}", img_result)


if __name__ == "__main__":
    main1()


