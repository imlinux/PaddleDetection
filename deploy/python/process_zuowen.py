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

file_infos = [

    ("JUYE_F_00007.pdf", "5ea43fc7ebf5f3a540e44f7d"),
    ("JUYE_F_00008.pdf", "5ea43fc7ebf5f3a540e44f7e")
]

pipeline = '''
    [
        {
            "$match": {
                "clazz": "{}",
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
                "teacherPhoto": "$teacher.photo"
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
        result = predict([img])
        boxes = result["boxes"]

        if len(boxes) > 0:
            r = boxes[boxes[:, 1].argsort()[::-1]]
            info.append(r)
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


def student_pdf(pdf_file, page_num):

    img_xref = pdf_file.get_page_images(page_num)
    base_image = pdf_file.extractImage(img_xref[0][0])
    image_bytes = base_image["image"]
    image_ext = base_image["ext"]
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    img[-250:, 100:2000] = [255, 255, 255]
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

    f = fs.find_one({"filename": filename})
    if f is None:
        return fs.put(data, filename = filename)
    else:
        return f._id

def process_one(pdf_file_path, class_id, topic_id):
    db, fs = open_db()
    topic_doc = db.teach_topic.findOne({"_id": ObjectId(topic_id)})

    data = db.user.aggregate(loads(pipeline.format(class_id)))

    with fitz.open(pdf_file_path) as pdf_file:
        for page_num in range(1, len(pdf_file), 2):

            record = data[page_num]
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
                    "author.userId": record.id,
                    "topicId": ObjectId(topic_id)
                }
                if db.clazzcircle.count_documents(query) == 0:

                    student_pdf_filename = f"{topic_doc.name}.pdf"
                    teacher_pdf_filename = f"{topic_doc.name}[已批改].pdf"

                    db_obj = {}
                    files = [
                        {
                            "fileId": save_pdf(student_pdf_filename, student_pdf(pdf_file, page_num)),
                            "name": student_pdf_filename,
                        },
                    ]
                    if not account_str.isspace():
                        db_obj["scoreInfos"] = {
                            "_id": ObjectId(),
                            "sendTime": current_time,
                            "score": {
                                "relevanceAndMaterial": int(relevanceAndMaterial),
                                "thinkAndFeel": int(thinkAndFeel),
                                "languageAndExpress": int(languageAndExpress),
                                "writeAndCount": int(writeAndCount)
                            },
                            "author": {
                                "userId": record.teacherId,
                                "userName": record.teacherName,
                                "userPhoto": str(record.teacherPhoto)
                            },
                            "self": False
                        }
                        files += [
                            {
                                "fileId": save_pdf(teacher_pdf_filename, teacher_pdf(pdf_file, page_num)),
                                "name": teacher_pdf_filename,
                            },
                        ]


                    db_obj = {
                        "clazz": {
                            "clazzId": ObjectId(class_id),
                            "clazzName": record.clazzName
                        },
                        "contentType": "COMPOSITION",
                        "topicId": ObjectId(topic_id),
                        "author": [{
                            "userId": record.id,
                            "userName": record.name,
                            "userPhoto": record.photo
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
                    }
                    db.clazzcircle.insert_one(db_obj)


process_one("/home/dong/tmp/zuowen/JUYE_F_00007.pdf", "", "")