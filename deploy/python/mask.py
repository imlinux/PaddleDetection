from my_common import open_db

db, gridfs = open_db("ah_school")


def common_mask(value, scale=0.5):

    if value == "" or value is None: return value

    l = len(value)
    mask_l = int(l * scale)

    return value[: l - mask_l] + "*" * mask_l



def mask_dict_field(data, key, scale=0.5):
    v = data[key] if key in data else ""
    data[key] = common_mask(v, scale)
    return data


for u in db.user.find({"name": {"$not": {"$in": ["喻晓波"]}}}):

    mask_dict_field(u, "name")
    mask_dict_field(u, "mobile")
    mask_dict_field(u, "idnumber")
    mask_dict_field(u, "company")
    mask_dict_field(u, "modifiedUserName")
    mask_dict_field(u, "createUserName")
    mask_dict_field(u, "residence2")
    mask_dict_field(u, "address2")
    mask_dict_field(u, "code")
    mask_dict_field(u, "account")

    if "residence" in u:
        residence = u["residence"]
        mask_dict_field(residence, "block")
        mask_dict_field(residence, "community")

    if "address" in u:
        address = u["address"]
        mask_dict_field(address, "address")

    if "residenceaddress" in u:
        residenceaddress = u["residenceaddress"]
        mask_dict_field(residenceaddress, "block")
        mask_dict_field(residenceaddress, "room")

    db.user.find_one_and_replace({"_id": u["_id"]}, u)
