from mako.template import Template
from pymongo import MongoClient
from urllib.parse import quote_plus
import gridfs
from bson.json_util import loads
import requests
import cv2
import numpy as np
import fitz

def open_db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
    client = MongoClient(uri)
    db = client["sigmai"]

    return db, gridfs.GridFS(db)


def save_file(filename, data):
    db, fs = open_db()
    return fs.put(data, filename=filename)


def qr_code_extract(img):

    url = "https://dualive.com:8443/tools/qr_code_extract"
    files = {'img': cv2.imencode(".jpg", img)[1].tobytes()}
    r = requests.post(url, files=files)
    json = r.json()
    data = json["data"]
    if len(data) > 0:
        return data[0]["message"]
    else :
        return ""


def horizontal_merge_img(imgs):

    total_w = sum([img.shape[1] for img in imgs])
    max_h = max([img.shape[0] for img in imgs])

    ret = np.zeros((max_h, total_w, 3), np.uint8)

    w = 0
    for img in imgs:
        h = img.shape[0]
        w += img.shape[1]
        ret[:h, w - img.shape[1]:w, :3] = img

    return ret


def extract_img_from_pdf(file, start = 0):
    with fitz.open(file) as pdf_file:
        for page_num in range(start, len(pdf_file)):
            page = pdf_file[page_num]
            for img in page.getImageList():
                base_image = pdf_file.extractImage(img[0])
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                yield img


def examine_info(examineId, clazzId, subjectId):

    query = '''
        [
                {
                    "$match": {
                        "_id": {
                            "$oid": "${examineId}"
                        }
                    }
                },
    
                {
                    "$addFields": {
                        "clazzSnapshots": {
                            "$filter": {
                                "input": "$clazzSnapshots",
                                "as": "item",
                                "cond": {
                                    "$eq": ["$$item.clazzId", {"$oid": "${clazzId}"}]
                                }
                            }
                        }
                    }
                },
                {
                    "$unwind": "$clazzSnapshots"
                },
                {
                    "$addFields": {
                        "clazzSnapshots.teachers": {
                            "$filter": {
                                "input": "$clazzSnapshots.teachers",
                                "as": "item",
                                "cond": {
                                    "$eq": ["$$item.subjectId", {"$oid": "${subjectId}"}]
                                }
                            }
                        }
                    }
                },
                {
                    "$unwind": "$clazzSnapshots.teachers"
                },
                {
                    "$lookup": {
                        "from": "student_kpi",
                        "let": {
                            "kpiIds": {
                                "$ifNull": ["$clazzSnapshots.teachers.kpi", []]
                            }
                        },
                        "pipeline": [
                            {
                                "$match": {
                                    "$expr": {
                                        "$and": [
                                            { "$in": ["$_id", "$$kpiIds"] }
                                        ]
                                    }
                                }
                            },
                            {
                                "$sort": {
                                    "order": 1
                                }
                            },
                            {
                                "$project": {
                                    "name": 1,
                                    "_id": 1,
                                    "starNumber": 1
                                }
                            }
                        ],
                        "as": "studentKpi"
                    }
                },
    
                {
                    "$lookup": {
                        "from": "user",
                        "localField": "clazzSnapshots.teachers.teacherId",
                        "foreignField": "_id",
                        "as": "teacher"
                    }
                },
                {
                    "$unwind": "$teacher"
                },
                {
                    "$project": {
                        "clazzClazzId": "$clazzSnapshots.clazzId",
                        "clazzStartyear": "$clazzSnapshots.startyear",
                        "clazzGrade": "$clazzSnapshots.grade",
                        "clazzNo": "$clazzSnapshots.no",
                        "clazzSchool": "$clazzSnapshots.school",
                        "clazzName": "$clazzSnapshots.name",
                        "teacherId": "$teacher._id",
                        "teacherName": "$teacher.name",
                        "studentKpi": 1
                    }
                }
        ]
    '''
    q = Template(query).render(examineId=examineId, clazzId=clazzId, subjectId=subjectId)
    return loads(q)


def zuowen_query(clazz_id):
    q = '''
        [
            {
                "$match": {
                    "clazz": ''' + '"' + clazz_id + '",' + '''
                    "status": {
                        "$in": ["在籍在读", "借读"]
                    },
                    "name": {
                        "$not": {
                            "$in": ["野智美", "周佳礼", "朱敏嘉", "苏子涵", "张彦辰"]
                        }
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
    return loads(q)