import os
import pathlib
import time
from my_common import open_db, examine_info, qr_code_extract, save_file, horizontal_merge_img

import cv2
import fitz
import numpy as np
from bson import ObjectId
from imutils import perspective
from paddleocr import PaddleOCR, PPStructure

from predict import Predict

model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"
predict = Predict(model_dir)
ocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False)
table_engine = PPStructure(show_log=True, use_gpu=False)


def sort_contours(cnts):

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    return sorted(zip(cnts, boundingBoxes), key=lambda b: (b[1][1], b[1][0]), reverse=False)


def predict_img(img_list):
    ret = ""
    info = []
    c = []

    no = ocr.ocr(img_list[0], cls=True)
    name = ocr.ocr(img_list[1], cls=True)

    if len(no) < 1:
        no = [[[], ("", 1)]]

    if len(name) < 1:
        name = [[[], ("", 1)]]

    for img in img_list[3:]:

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

    return *no, *name, ret, info


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
        if 1000 < w and 1000 < h:
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


def process_img(img_raw):
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

    table_cells = []
    for c, b in sort_contours(contours):

        x, y, w, h = b

        if 80 < w < 600 and 80 < h < 170:
            new_img = img[y:y + h, x:x + w]
            table_cells.append(new_img)

    table_rows = []

    for i in range(0, len(table_cells), 11):
        table_rows.append(table_cells[i: i + 11])

    os.makedirs("./output", exist_ok=True)
    cv2.imwrite(f"./output/{time.time()}.jpg", cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3))
    return table_rows


def process_one(pdf_file_path):
    with fitz.open(pdf_file_path) as pdf_file:

        for page_num in range(1, len(pdf_file), 2):

            table_rows = []

            page0_img = pdf_file[page_num - 1].getImageList()[0][0]
            page1_img = pdf_file[page_num].getImageList()[0][0]

            base_image = pdf_file.extractImage(page0_img)
            image_bytes = base_image["image"]
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            qr_code = qr_code_extract(img)
            (examine_id, clazz_id, subject_id) = qr_code.split("/")
            table_rows += process_img(img)

            base_image = pdf_file.extractImage(page1_img)
            image_bytes = base_image["image"]
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

            # result = table_engine(img)
            # img = result[0]["img"]
            table_rows += process_img(img)

            for table_row in table_rows:
                [no_location, (no, no_score)], [name_location, (name, name_score)], score, info = predict_img(table_row)
                print(f"{no=}, {name=}, {score=}, {qr_code=}")
                save_score(no, name, score, examine_id, clazz_id, subject_id, table_row)


def score_item(s, max_score):

    if s >= max_score * 0.9:
        return 4
    elif s >= max_score * 0.8:
        return 3
    elif s >= max_score * 0.6:
        return 2
    elif s > 0:
        return 1
    else:
        return 0


def save_score(no, name, score, examine_id, clazz_id, subject_id, imgs):

    db, _ = open_db()
    student = db.user.find_one({"no": no, "clazz": clazz_id})
    if student is None:
        print(f"未找到{no}")
        return

    examine = list(db.examine.aggregate(examine_info(examine_id, clazz_id, subject_id)))
    if len(examine) > 0:
        [examine] = examine

    scores = [score[0:2], score[3:5], score[6:]]

    score_info = []
    for idx, s in enumerate(scores):
        kpi_info = examine["studentKpi"][idx]
        starNumber = kpi_info["starNumber"]

        input_score_value = int(s) if not s.isspace() else 0
        input_score_value = input_score_value if not score.isspace() else starNumber
        final_score = starNumber - input_score_value

        score_info.append({
            "key": kpi_info["name"] + "分值",
            "value": final_score
        })
        score_info.append({
            "key": kpi_info["name"],
            "value": score_item(final_score, starNumber)
        })

    score_doc = {
        "examineId": ObjectId(examine_id),
        "clazzId": ObjectId(clazz_id),
        "subjectId": ObjectId(subject_id),
        "studentId": student["_id"],
        "className": examine["clazzName"],
        "createUserId": examine["teacherId"],
        "createUserName": examine["teacherName"],
        "create_time": int(round(time.time(), 0) * 1000),
        "update_time": int(round(time.time(), 0) * 1000),
        "modifiedUserId": examine["teacherId"],
        "modifiedUserName": examine["teacherName"],
        "no": examine["clazzNo"],
        "school": examine["clazzSchool"],
        "startyear": examine["clazzStartyear"],
        "score": score_info
    }
    if db.score.count({"examineId": ObjectId(examine_id), "clazzId": ObjectId(clazz_id), "subjectId": ObjectId(subject_id), "studentId": student["_id"]}) == 0:
        cnn_info = {
            "score_img": save_file("score.jpg", cv2.imencode(".jpg", horizontal_merge_img(imgs))[1].tobytes()),
            "score_str": score,
            "name": name
        }
        print(f"{score_doc=}")
        score_doc["cnn_info"] = cnn_info
        db.score.insert_one(score_doc)
        db.examine.update_one({"_id": ObjectId(examine_id)}, {
            "$set": {
                "clazzSnapshots.$[elem].teachers.$[s].submit": True
            }
        }, array_filters = [
            {"elem.clazzId": ObjectId(clazz_id)},
            {
                "s.teacherId": examine["teacherId"],
                "s.subjectId": student["_id"]
            }
        ])


def main():

    for path in pathlib.Path("/home/dong/tmp/score2").glob("**/*.pdf"):
        print(path)
        process_one(str(path))


if __name__ == "__main__":
    main()
