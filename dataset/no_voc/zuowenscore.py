from bson import ObjectId
from bson.json_util import loads
from pymongo import MongoClient
from urllib.parse import quote_plus


query = '''
[
    {
        "$match": {
            "cnn_info": {
                "$exists": true
            }
        }
    },

    {
        "$lookup": {
            "from": "clazz",
            "localField": "clazz.clazzId",
            "foreignField": "_id",
            "as": "clazz"
        }
    },
    
    {
        "$unwind": "$clazz"
    },
    
    {
        "$match": {

            "clazz.school": {
                "$regex": "巨野"
            },
            "clazz.grade": "FOUR",
            "clazz.clazzName": {
                "$ne": "四7"
            }
        }
    },
    
    {
        "$unwind": "$scoreInfos"
    },
    
    {
        "$project": {
            "author": 1,
            "school": "$clazz.school",
            "clazzId": "$clazz.clazzId",
            "score": "$scoreInfos.score",
            "predict": "$cnn_info.score"
        }
    }
]    
'''


def open_db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
    client = MongoClient(uri)
    db = client["sigmai"]

    return db


def main():
    db = open_db()
    data = list(db.clazzcircle.aggregate(loads(query)))

    r = 0
    total = 0

    for row in data:

        predict = row["predict"].replace(" ", "")
        score_info = row["score"]
        score = "%s%s%s%s" % (str(score_info["relevanceAndMaterial"]).strip(), str(score_info["thinkAndFeel"]).strip(), str(score_info["languageAndExpress"]).strip(), str(score_info["writeAndCount"]).strip())

        total = total + 1
        if score == predict:
            r = r + 1
        else:
            print(f'{score}, {predict}', row["author"])

    print(f'{r}/{total}, {r/total}')


if __name__ == "__main__":
    main()