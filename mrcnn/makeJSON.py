import json

#x = [1, 2, 3]
#y = [4, 5, 6]
#size = 8635
#filename = "sample03.jpg"


def MaskJson(filename, size, x, y,i):

    x = list(map(int, x))
    y = list(map(int, y))
    filename_join = filename + str(size)

    json_mask = dict()
    regions = dict()

    shape = dict()
    print(x)
    print(y)

    shape["name"] = "polygon"
    shape["all_points_x"] = x
    shape["all_points_y"] = y

    re_attrib = dict()
    re_attrib["name"] = "hanja"

    num = dict()
    num["shape_attributes"] = shape
    num["region_attributes"] = re_attrib

    regions["%d" % i] = num

    name = dict()
    name["fileref"] = ""
    name["size"] = size
    name["filename"] = filename
    name["base64_img_data"] = ""
    name["file_attributes"] = {}
    name["regions"] = regions

    json_mask[filename_join] = name

    make_file = open('test.json', 'w', encoding="utf-8")
    json.dump(json_mask, make_file, ensure_ascii=False, indent="\t")


