import time
from urllib.parse import quote_plus

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


def pipeline(account):
    return  '''
        [
            {
                "$match": {
                    "account": ''' + '"' + account + '",' + '''
                    "status": {
                        "$in": ["在籍在读", "借读"]
                    }
                }
            },
            {
                "$addFields": {
                    "sortNo": {
                        "$toInt": {
                            "$cond": {
                                "if": {
                                    "$eq": ["", { "$ifNull": ["$no", ""] }]
                                },
                                "then": "1000000",
                                "else": "$no"
                            }
                        }
                    },
                    "clazzId": {
                        "$toObjectId": "$clazz"
                    }
                }
            },
    
            {
                "$lookup": {
                    "from": "clazz",
                    "foreignField": "_id",
                    "localField": "clazzId",
                    "as": "clazz"
                }
            },
            
            {
                "$unwind": "$clazz"
            },
            
            {
                "$addFields": {
                    "clazz.teachers": {
                        "$filter": {
                            "input": "$clazz.teachers",
                            "as": "item",
                            "cond": {
                                "$eq": ["$$item.subjectId", {"$toObjectId": "5fa8b14de54cbd0a21b61c45"}]
                            }
                        }
                    }
                }
            },
            {
                "$unwind": "$clazz.teachers"
            },
            {
                "$lookup": {
                    "from": "user",
                    "localField": "clazz.teachers.teacherId",
                    "foreignField": "_id",
                    "as": "teacher"
                }
            },
            {
                "$unwind": "$teacher"
            },
    
            {
                "$unwind": "$clazz"
            },
            {
                "$sort": {
                    "sortNo": 1
                }
            },
            {
                "$project": {
                    "id": "$_id",
                    "name": 1,
                    "photo": 1,
                    "school": "$clazz.school",
                    "schoolabbr": "$clazz.schoolabbr",
                    "clazzId": "$clazz._id",
                    "clazzName": "$clazz.name",
                    "clazzNameabbr": "$clazz.nameabbr",
                    "grade": "$clazz.grade",
                    "startyear": "$clazz.startyear",
                    "teacherName": "$teacher.name",
                    "teacherId": "$teacher._id",
                    "teacherPhoto": "$teacher.photo",
                    "no": 1,
                    "account": 1
                }
            }
        ]
    '''


key_point_model_dir = "/home/dong/dev/PaddleDetection/inference_model/higherhrnet_hrnet_w32_512_lo"
model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no_20211018"


keyPointPredict = KeyPointPredict(key_point_model_dir)
predict = Predict(model_dir)


def open_db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "27019")
    client = MongoClient(uri)
    db = client["sigmai"]

    return db, gridfs.GridFS(db)


def predict_img(img_list):
    ret = ""
    info = []
    c = []
    for img in img_list:

        if img.shape[0] == 0:
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


def save_cnn_info(filename, page, topic_id, account, account_info, account_img, score, score_info, score_img):
    db, _ = open_db()
    query = {"filename": filename, "page": page, "topic_id": topic_id}

    doc = db.cnn_info.find_one(query)
    if doc is None:
        return db.cnn_info.insert_one({
            "filename": filename,
            "page": page,
            "topic_id": topic_id,
            "account": account,
            "account_img": save_file("account_img.jpg", cv2.imencode(".jpg", account_img)[1].tobytes()),
            "score": score,
            "score_img": save_file("score_img.jpg", cv2.imencode(".jpg", score_img)[1].tobytes()),
            "col": "clazzcircle",
            "account_info": account_info[:5],
            "score_info": score_info[:5],
        }).inserted_id
    return doc["_id"]


def update_cnn_info(id, reference_id):
    db, _ = open_db()
    db.cnn_info.update_one({"_id": id}, {"$set": {"reference_id": reference_id}})


def process_one(pdf_file_path, topic_id):
    db, fs = open_db()
    topic_doc = db.teach_topic.find_one({"_id": ObjectId(topic_id)})
    teach_topic_name = topic_doc["name"]

    with fitz.open(pdf_file_path) as pdf_file:
        for page_num in range(1, len(pdf_file), 2):
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
                account, score, all_account, all_score = split(img_result)

                account_str, account_info = predict_img(account)
                score_str, score_info = predict_img(score)

                cnn_info_id = save_cnn_info(pdf_file_path, page_num, topic_id, account_str, account_info, all_account, score_str, score_info, all_score)

                relevanceAndMaterial = score_str[0:2]
                thinkAndFeel = score_str[2:4]
                languageAndExpress = score_str[4:6]
                writeAndCount = score_str[6:8]

                print(f"{account_str=} {score_str=}")
                current_time = int(round(time.time()*1000))
                record = list(db.user.aggregate(loads(pipeline(account_str))))

                if len(record) == 0: continue

                [record] = record

                query = {
                    "clazz.clazzId": record["clazzId"],
                    "author.userId": record["id"],
                    "topicId": ObjectId(topic_id)
                }
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
                                "userId": record["teacherId"],
                                "userName": record["teacherName"],
                                "userPhoto": str(record["teacherPhoto"] if "teacherPhoto" in record else "")
                            },
                            "self": False
                        }]

                    db_obj.update({
                        "rawPdf": save_file(student_pdf_filename, raw_pdf(pdf_file, page_num)),
                        "clazz": {
                            "clazzId": record["clazzId"],
                            "clazzName": record["clazzNameabbr"]
                        },
                        "contentType": "COMPOSITION",
                        "topicId": ObjectId(topic_id),
                        "author": [{
                            "userId": record["id"],
                            "userName": record["name"],
                            "userPhoto": record["photo"] if "photo" in record else None
                        }],
                        "files": files,
                        "create_time": current_time,
                        "sendTime": current_time,
                        "update_time": current_time,
                        "status": "PRIVATE",
                        "subjectId": ObjectId("5fa8b14de54cbd0a21b61c45"),
                        "type": "OFFICE",
                        "cnn_info": cnn_info_id
                    })
                    cc_id = db.clazzcircle.insert_one(db_obj).inserted_id
                    update_cnn_info(cnn_info_id, cc_id)


def main():

    file_infos = [
        ("/home/dong/Downloads/JUYE_F_00065.pdf", "60534fd6c87b3f72ac4ea8d2", False),
        ("/home/dong/Downloads/JUYE_F_00066.pdf", "60534fd6c87b3f72ac4ea8d2", False),
        ("/home/dong/Downloads/JUYE_F_00067.pdf", "60534fd6c87b3f72ac4ea8d2", True),
        ("/home/dong/Downloads/JUYE_F_00068.pdf", "60534fd6c87b3f72ac4ea8d2", True),
    ]

    db, _ = open_db()

    for file, topicId, process in file_infos:
        if process: continue

        print(file)
        process_one(file, topicId)


if __name__ == "__main__":
    main()