from bson import ObjectId
from pymongo import MongoClient
import gridfs
from urllib.parse import quote_plus
import fitz
import time
import numpy as np
import cv2
from keypoint_predict import KeyPointPredict, split
from predict import Predict
from imutils import perspective
from bson.json_util import loads
import string

file_infos = [

    ("JUYE_F_00007.pdf", "5ea43fc7ebf5f3a540e44f7d"),
    ("JUYE_F_00008.pdf", "5ea43fc7ebf5f3a540e44f7e"),
    ("JUYE_F_00010.pdf", "5ea43fc7ebf5f3a540e44f7f")
]

def pipeline(clazzId):
    return  '''
        [
            {
                "$match": {
                    "clazz": ''' + '"' + clazzId + '",' + '''
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
model_dir = "/home/dong/dev/PaddleDetection/inference_model/yolov3_mobilenet_v1_no"


keyPointPredict = KeyPointPredict(key_point_model_dir)
predict = Predict(model_dir)


def open_db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
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


def teacher_pdf(pdf_file, page_num):
    pdf_doc = fitz.Document()
    pdf_doc.insert_pdf(pdf_file, from_page=page_num - 1, to_page=page_num)
    return pdf_doc.tobytes()


def save_pdf(filename, data):
    db, fs = open_db()
    return fs.put(data, filename=filename)


def process_one(pdf_file_path, class_id, topic_id):
    db, fs = open_db()
    topic_doc = db.teach_topic.find_one({"_id": ObjectId(topic_id)})
    teach_topic_name = topic_doc["name"]
    print(pipeline(class_id))
    data = list(db.user.aggregate(loads(pipeline(class_id))))

    with fitz.open(pdf_file_path) as pdf_file:
        for page_num in range(1, len(pdf_file), 2):

            record = data[page_num//2]
            page = pdf_file[page_num]
            print(record["no"], record["account"], record["name"])
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

                account_str, account_info = predict_img(account)
                score_str, score_info = predict_img(score)

                relevanceAndMaterial = score_str[0:2]
                thinkAndFeel = score_str[2:4]
                languageAndExpress = score_str[4:6]
                writeAndCount = score_str[6:8]

                print(f"{account_str=} {score_str=}")
                current_time = int(round(time.time()*1000))

                query = {
                    "clazz.clazzId": ObjectId(class_id),
                    "author.userId": record["id"],
                    "topicId": ObjectId(topic_id)
                }
                if db.clazzcircle.count_documents(query) == 0:

                    student_pdf_filename = f"{teach_topic_name}.pdf"
                    teacher_pdf_filename = f"{teach_topic_name}[已批改].pdf"

                    db_obj = {}
                    files = [
                        {
                            "fileId": save_pdf(student_pdf_filename, student_pdf(pdf_file, page_num)),
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
                                "userPhoto": str(record["teacherPhoto"])
                            },
                            "self": False
                        }]
                        files += [
                            {
                                "fileId": save_pdf(teacher_pdf_filename, student_pdf(pdf_file, page_num)),
                                "name": teacher_pdf_filename,
                            },
                        ]

                    db_obj.update({
                        "clazz": {
                            "clazzId": ObjectId(class_id),
                            "clazzName": record["clazzName"]
                        },
                        "contentType": "COMPOSITION",
                        "topicId": ObjectId(topic_id),
                        "author": [{
                            "userId": record["id"],
                            "userName": record["name"],
                            "userPhoto": record["photo"]
                        }],
                        "files": files,
                        "create_time": current_time,
                        "sendTime": current_time,
                        "update_time": current_time,
                        "status": "PRIVATE",
                        "subjectId": ObjectId("5fa8b14de54cbd0a21b61c45"),
                        "type": "OFFICE",
                        "cnn_info": {
                            "account": account_str,
                            "account_info": account_info,
                            "score": score_str,
                            "score_info": score_info
                        }
                    })
                    db.clazzcircle.insert_one(db_obj)


process_one("/home/dong/tmp/zuowen/JUYE_F_00017.pdf", "5ea43fc7ebf5f3a540e44f86", "60534fd6c87b3f72ac4ea853")