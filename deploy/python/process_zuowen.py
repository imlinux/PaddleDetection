import os
import time

import cv2
import fitz
import numpy as np
from bson import ObjectId
from imutils import perspective

from keypoint_predict import KeyPointPredict, split
from my_common import zuowen_query, qr_code_extract, open_db
from predict import Predict

key_point_model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"


keyPointPredict = KeyPointPredict(key_point_model_dir)
predict = Predict(model_dir)


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


def remove_red(image):

    blue_c, green_c, red_c = cv2.split(image)

    row_range = image.shape[0] // 2

    red_part = red_c[-row_range:]

    thresh, ret = cv2.threshold(red_part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter_condition = int(thresh * 0.95)
    _, red_thresh = cv2.threshold(red_part, filter_condition, 255, cv2.THRESH_BINARY)

    red_c[-row_range:] = red_thresh
    blue_c[-row_range:] = red_thresh
    green_c[-row_range:] = red_thresh

    result_blue = np.expand_dims(blue_c, axis=2)
    result_greep = np.expand_dims(green_c, axis=2)
    result_red = np.expand_dims(red_c, axis=2)
    result_img = np.concatenate((result_blue, result_greep, result_red), axis=-1)

    return result_img

def remove_blue(image):

    blue_c, green_c, red_c = cv2.split(image)

    row_range = image.shape[0] // 2

    blue_part = blue_c[-row_range:]

    thresh, ret = cv2.threshold(blue_part, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter_condition = int(thresh * 0.95)
    _, blue_thresh = cv2.threshold(blue_part, filter_condition, 255, cv2.THRESH_BINARY)

    red_c[-row_range:] = blue_thresh
    blue_c[-row_range:] = blue_thresh
    green_c[-row_range:] = blue_thresh

    result_blue = np.expand_dims(blue_c, axis=2)
    result_greep = np.expand_dims(green_c, axis=2)
    result_red = np.expand_dims(red_c, axis=2)
    result_img = np.concatenate((result_blue, result_greep, result_red), axis=-1)

    return result_img

def student_pdf(pdf_file, page_num):

    img_xref = pdf_file.get_page_images(page_num)
    base_image = pdf_file.extractImage(img_xref[0][0])
    image_bytes = base_image["image"]
    image_ext = base_image["ext"]
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    img = remove_red(img)

    img_bytes = cv2.imencode(".jpg", img)[1].tobytes()

    rect = pdf_file[page_num].bound()
    *_, width, height = rect

    pdf_doc = fitz.Document()
    pdf_doc.insert_pdf(pdf_file, from_page=page_num - 1, to_page=page_num - 1)

    new_page = pdf_doc.new_page(width=width, height=height)
    new_page.insert_image(rect, stream=img_bytes)

    return pdf_doc.tobytes()


def raw_pdf(pdf_file, page_num):
    pdf_doc = fitz.Document()
    pdf_doc.insert_pdf(pdf_file, from_page=page_num - 1, to_page=page_num)
    return pdf_doc.tobytes()


def save_file(filename, data):
    db, fs = open_db()
    return fs.put(data, filename=filename)


def process_one(pdf_file_path, clazz_id):
    db, fs = open_db()
    data = list(db.user.aggregate(zuowen_query(clazz_id)))
    total = len(data)
    account_match_cnt = 0
    with fitz.open(pdf_file_path) as pdf_file:
        for page_num in range(1, len(pdf_file), 2):
            page = pdf_file[page_num]
            for img in page.getImageList():
                base_image = pdf_file.extractImage(img[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                qrcode = qr_code_extract(img)

                if qrcode == "":
                    print(f"{pdf_file_path}第{page_num + 1}页，未识别到二维码")
                    return

                teach_topic_name, topic_id = qrcode.split("/")
                topic_id = ObjectId(topic_id)
                topic_doc = db.teach_topic.find_one({"_id": topic_id})
                composition_type = topic_doc["compositionType"] if "compositionType" in topic_doc else ""

                print(f"{teach_topic_name=}, {topic_id=}, {composition_type=}")

                if composition_type != "ACTIVITY_WORD" and composition_type != "ACTIVITY":
                    result = keyPointPredict(img)
                    skeletons, scores = result["keypoint"]
                    pts = np.array(
                        [skeletons[0][0][0: 2], skeletons[0][1][0: 2], skeletons[0][2][0: 2], skeletons[0][3][0: 2]])

                    img_result = perspective.four_point_transform(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1), pts)
                    account, score, all_account, all_score = split(img_result)

                    account_str, account_info = predict_img(account)
                    score_str, score_info = predict_img(score)

                    relevanceAndMaterial = score_str[0:2]
                    thinkAndFeel = score_str[2:4]
                    languageAndExpress = score_str[4:6]
                    writeAndCount = score_str[6:8]
                else:
                    score_str = " "
                    account_str = " "
                    account = []
                    score = []
                    all_account = np.empty((0, 0, 0))
                    all_score = np.empty((0, 0, 0))
                    account_info = []
                    score_info = []

                current_time = int(round(time.time()*1000))
                # record = list(db.user.aggregate(loads(pipeline(account_str))))
                record = [data[page_num // 2]]
                if len(record) == 0:
                    record = {}
                    query = {
                        "cnn_info.filename": pdf_file_path,
                        "cnn_info.page": page_num,
                        "cnn_info.topic_id": topic_id
                    }
                else:
                    [record] = record
                    query = {
                        "clazz.clazzId": record["clazzId"],
                        "author.userId": record["id"],
                        "topicId": ObjectId(topic_id)
                    }

                account_match = record["account"] == account_str

                if account_match:
                    account_match_cnt += 1
                else:
                    os.makedirs("/home/dong/tmp/tmp2", exist_ok=True)
                    for idx, a in enumerate(account):
                        shape = a.shape
                        if shape[0] > 0 and shape[1] > 0:
                            cv2.imwrite(f"/home/dong/tmp/tmp2/{page_num}-a-{idx}.jpg", a)

                    for idx, s in enumerate(score):
                        if len(s) > 0:
                            cv2.imwrite(f"/home/dong/tmp/tmp2/{page_num}-s-{idx}.jpg", s)

                    if all_account.shape[0] > 0:
                        cv2.imwrite(f"/home/dong/tmp/tmp2/{page_num}-account.jpg", all_account)
                    if all_score.shape[0] > 0:
                        cv2.imwrite(f"/home/dong/tmp/tmp2/{page_num}-score.jpg", all_score)

                print(f"{record['no']} {record['account']} {record['name']} {account_str=} {score_str=} {account_match=} {account_match_cnt}/{total}={account_match_cnt/total}")

                if db.clazzcircle.count_documents(query) == 0:

                    student_pdf_filename = f"{teach_topic_name}.pdf"

                    db_obj = {}
                    files = [
                        {
                            "fileId": save_file(student_pdf_filename, student_pdf(pdf_file, page_num)),
                            "name": student_pdf_filename,
                        },
                    ]
                    if not score_str.isspace():
                        db_obj["scoreInfos"] = [{
                            "_id": ObjectId(),
                            "sendTime": current_time,
                            "score": {
                                "relevanceAndMaterial": int(relevanceAndMaterial) if not relevanceAndMaterial.isspace() else 0,
                                "thinkAndFeel": int(thinkAndFeel) if not thinkAndFeel.isspace() else 0,
                                "languageAndExpress": int(languageAndExpress) if not languageAndExpress.isspace() else 0,
                                "writeAndCount": int(writeAndCount) if not writeAndCount.isspace() else 0
                            },
                            "author": {
                                "userId": record["teacherId"] if "teacherId" in record else "",
                                "userName": record["teacherName"] if "teacherName" in record else "",
                                "userPhoto": str(record["teacherPhoto"] if "teacherPhoto" in record else "")
                            },
                            "self": False
                        }]

                    db_obj.update({
                        "rawPdf": save_file(student_pdf_filename, raw_pdf(pdf_file, page_num)),
                        "clazz": {
                            "clazzId": record["clazzId"] if "clazzId" in record else "",
                            "clazzName": record["clazzNameabbr"] if "clazzNameabbr" in record else ""
                        },
                        "contentType": "COMPOSITION",
                        "topicId": ObjectId(topic_id),
                        "author": [{
                            "userId": record["id"] if "id" in record else "",
                            "userName": record["name"] if "name" in record else "",
                            "userPhoto": record["photo"] if "photo" in record else None
                        }],
                        "files": files,
                        "create_time": current_time,
                        "sendTime": current_time,
                        "update_time": current_time,
                        "status": "PENDING",
                        "subjectId": ObjectId("5fa8b14de54cbd0a21b61c45"),
                        "type": "OFFICE",
                        "cnn_info": {
                            "filename": pdf_file_path,
                            "page": page_num,
                            "topic_id": topic_id,
                            "account": account_str,
                            "account_img": save_file("account_img.jpg", cv2.imencode(".jpg", all_account)[1].tobytes()) if len(all_score) > 0 else None,
                            "score": score_str,
                            "score_img": save_file("score_img.jpg", cv2.imencode(".jpg", all_score)[1].tobytes()) if len(all_score) > 0 else None,
                            "account_info": account_info[:5],
                            "score_info": score_info[:5],
                        }
                    })
                    db.clazzcircle.insert_one(db_obj)


def main():

    file_infos = [
        ("/home/dong/tmp/zuowen3/JUYE_F_00093.pdf", "170004", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00094.pdf", "170086", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00095.pdf", "170125", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00097.pdf", "190043", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00098.pdf", "190045", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00099.pdf", "190130", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00100.pdf", "190577", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00102.pdf", "190401", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00103.pdf", "190454", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00104.pdf", "180035", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00105.pdf", "190200", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00106.pdf", "180004", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00107.pdf", "190188", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00108.pdf", "190250", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00109.pdf", "190269", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00110.pdf", "190097", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00111.pdf", "190362", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00112.pdf", "190495", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00113.pdf", "190538", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00115.pdf", "170412", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00116.pdf", "170492", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00118.pdf", "170548", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00119.pdf", "170630", True),
        ("/home/dong/tmp/zuowen3/JUYE_F_00120.pdf", "170648", True),
    ]

    db, _ = open_db()

    for file, student_account, process in file_infos:
        if process: continue
        user_doc = db.user.find_one({"account": student_account})
        print(file)
        process_one(file, user_doc["clazz"])


if __name__ == "__main__":
    main()